"""Provider-free v4 validation for sealed typed-final completions.

The v3 terminal-answer contract is intentionally left reproducible.  This
module upgrades one authenticated v3 validation contract plus its exact dated
question into a v4 contract and revalidates the already sealed completion.  It
does not edit the prompt, call a provider, read benchmark gold, or mutate a v3
answer artifact.

V4 repairs two generic fail-closed boundaries:

* a deterministic advisory is binding only for an answer shape the typed
  executor can actually decide; direct/synthesis answers still require the
  semantic candidate arbiter, and
* aggregate completeness may narrow an action-wide universe only by a topic
  term that came from the question and is shared by every cited semantic row.
  Candidate-only terms can therefore never hide uncited evidence.

Relative exact-day replacements also have to cite evidence dated on the
question-derived target day.  The target is compiled from question text only.
"""

from __future__ import annotations

import json
import re
from calendar import monthrange
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from memory_condense.domain.discourse import quote_sha256
from memory_condense.domain.text_numbers import NUMBER_WORDS

from .contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from .typed_action_semantics import canonical_action_concepts
from .typed_memory_final_arm import (
    VALIDATION_CONTRACT_FORMAT as LEGACY_VALIDATION_CONTRACT_FORMAT,
    ParsedTypedFinalDecision,
    _rows_in_complete_proof_scope,
    _rows_relevant_to_question,
    _validated_semantic_rows,
    judge_row_projection,
    parse_typed_final_completion,
)
from .typed_operator_spec import AnswerShape, TemporalMode, normalized_terms


FORMAT = "memory-condense-typed-memory-final-validator-v4"
VALIDATION_CONTRACT_FORMAT = (
    "memory-condense-typed-memory-final-arm-v1-"
    "completion-validation-contract-v4"
)
VALIDATOR_POLICY_FORMAT = (
    "memory-condense-typed-memory-final-arm-v1-validator-policy-v4"
)
DECISION_FORMAT = (
    "memory-condense-typed-memory-final-arm-v1-decision-v2"
)
RESULT_ROW_FORMAT = (
    "memory-condense-typed-memory-final-arm-v1-result-row-v2"
)

_AGGREGATE_OPERATIONS = frozenset(
    {"count_or_aggregate", "deduplicated_member_join", "order_or_select"}
)
_INTERNAL_SCOPE_VALIDATED_OPERATION = "v4_question_topic_scope_validated"
_DATED_RE = re.compile(
    r"^\[Question asked at (?P<asked_at>.+?)\]\s*(?P<body>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_RELATIVE_DAY_RE = re.compile(
    r"\b(?P<count>\d+|"
    + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
    + r")\s+days?\s+ago\b",
    re.IGNORECASE,
)
_LOOKBACK_RE = re.compile(
    r"\b(?:last|past)\s+(?P<count>\d+|"
    + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
    + r")\s+(?P<unit>days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_RE = re.compile(r"\b(?:I|me|my|mine)\b", re.IGNORECASE)
_ORDINAL_DAY_RE = re.compile(r"^(?P<day>\d{1,2})(?:st|nd|rd|th)$", re.I)
_MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_WEEKDAY_TERMS = frozenset(
    {
        "mon",
        "monday",
        "tue",
        "tues",
        "tuesday",
        "wed",
        "wednesday",
        "thu",
        "thur",
        "thurs",
        "thursday",
        "fri",
        "friday",
        "sat",
        "saturday",
        "sun",
        "sunday",
    }
)
_QUESTION_SCOPE_NOISE = frozenset(
    {
        "amount",
        "ask",
        "count",
        "day",
        "different",
        "distinct",
        "item",
        "kind",
        "many",
        "number",
        "piece",
        "question",
        "time",
        "total",
        "type",
        "week",
        "year",
        *NUMBER_WORDS,
        *_WEEKDAY_TERMS,
    }
)


class TypedMemoryFinalValidatorV4Error(MatchedEvalContractError):
    """A legacy contract, v4 derivation, or sealed completion changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise TypedMemoryFinalValidatorV4Error(message)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _question_focus_terms(
    dated_question: str,
    question_actions: Sequence[str],
) -> tuple[str, ...]:
    action_terms = set(question_actions)
    return tuple(
        term
        for term in normalized_terms(dated_question)
        if term not in action_terms
        and term not in _QUESTION_SCOPE_NOISE
        and not term.isdigit()
    )


def _asked_datetime(dated_question: str) -> datetime | None:
    match = _DATED_RE.match(dated_question)
    if match is None:
        return None
    raw = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", match.group("asked_at")).strip()
    normalized = raw.replace("/", "-")
    for form in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(normalized, form)
        except ValueError:
            pass
    return None


def _subtract_calendar_months(value: datetime, months: int) -> datetime:
    absolute = value.year * 12 + value.month - 1 - months
    year, month_index = divmod(absolute, 12)
    month = month_index + 1
    day = min(value.day, monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _question_temporal_validation(
    dated_question: str,
    temporal_mode: str,
) -> dict[str, Any]:
    asked = _asked_datetime(dated_question)
    offset_match = _RELATIVE_DAY_RE.search(dated_question)
    lookback_match = _LOOKBACK_RE.search(dated_question)
    if temporal_mode == TemporalMode.RELATIVE_SELECT.value:
        if asked is None or offset_match is None:
            body: dict[str, Any] = {
                "amount": None,
                "format": f"{FORMAT}-temporal-validation-v1",
                "lower_bound_date": None,
                "mode": "unresolved_relative_target",
                "offset_days": None,
                "query_date": asked.date().isoformat() if asked else None,
                "target_date": None,
                "unit": None,
            }
        else:
            raw = offset_match.group("count").casefold()
            offset = int(raw) if raw.isdigit() else int(NUMBER_WORDS[raw])
            target = asked.date() - timedelta(days=offset)
            body = {
                "amount": offset,
                "format": f"{FORMAT}-temporal-validation-v1",
                "lower_bound_date": target.isoformat(),
                "mode": "exact_calendar_day_offset",
                "offset_days": offset,
                "query_date": asked.date().isoformat(),
                "target_date": target.isoformat(),
                "unit": "day",
            }
    elif asked is not None and lookback_match is not None:
        raw = lookback_match.group("count").casefold()
        amount = int(raw) if raw.isdigit() else int(NUMBER_WORDS[raw])
        unit = lookback_match.group("unit").casefold().rstrip("s")
        if unit == "day":
            lower = asked - timedelta(days=amount)
        elif unit == "week":
            lower = asked - timedelta(weeks=amount)
        elif unit == "month":
            lower = _subtract_calendar_months(asked, amount)
        else:
            lower = _subtract_calendar_months(asked, amount * 12)
        body = {
            "amount": amount,
            "format": f"{FORMAT}-temporal-validation-v1",
            "lower_bound_date": lower.date().isoformat(),
            "mode": "bounded_calendar_lookback",
            "offset_days": None,
            "query_date": asked.date().isoformat(),
            "target_date": None,
            "unit": unit,
        }
    else:
        body: dict[str, Any] = {
            "amount": None,
            "format": f"{FORMAT}-temporal-validation-v1",
            "lower_bound_date": None,
            "mode": "not_applicable",
            "offset_days": None,
            "query_date": None,
            "target_date": None,
            "unit": None,
        }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _deterministic_advisory_policy(
    legacy_contract: Mapping[str, Any],
    temporal_validation: Mapping[str, Any],
) -> dict[str, Any]:
    advisory = legacy_contract.get("deterministic_execution_advisory")
    if advisory is None:
        body: dict[str, Any] = {
            "advisory_sha256": None,
            "binding_required": False,
            "format": f"{FORMAT}-deterministic-advisory-policy-v1",
            "reason": "no_supported_advisory",
            "status": "absent",
        }
    else:
        _require(
            type(advisory) is dict
            and advisory.get("status") == "supported"
            and type(advisory.get("prediction")) is str
            and bool(advisory["prediction"])
            and type(advisory.get("used_handle_ids")) is list
            and bool(advisory["used_handle_ids"])
            and all(
                type(handle) is str and bool(handle)
                for handle in advisory["used_handle_ids"]
            )
            and len(set(advisory["used_handle_ids"]))
            == len(advisory["used_handle_ids"])
            and set(advisory["used_handle_ids"])
            <= set(legacy_contract.get("by_handle", {})),
            "legacy deterministic advisory changed schema",
        )
        semantic_shape = legacy_contract.get("answer_shape") in {
            AnswerShape.DIRECT.value,
            AnswerShape.SYNTHESIS.value,
        }
        unresolved_relative = (
            legacy_contract.get("temporal_mode")
            == TemporalMode.RELATIVE_SELECT.value
            and temporal_validation.get("mode")
            != "exact_calendar_day_offset"
        )
        if semantic_shape:
            binding = False
            reason = "semantic_answer_shape_requires_candidate_arbiter"
        elif unresolved_relative:
            binding = False
            reason = "relative_target_not_executable"
        else:
            binding = True
            reason = "deterministic_answer_shape_and_target_executable"
        body = {
            "advisory_sha256": identity_sha256(advisory),
            "binding_required": binding,
            "format": f"{FORMAT}-deterministic-advisory-policy-v1",
            "reason": reason,
            "status": "eligible" if binding else "ineligible",
        }
    return {**body, "receipt_sha256": identity_sha256(body)}


def upgrade_completion_validation_contract_v4(
    legacy_contract: Mapping[str, Any],
    *,
    dated_question: str,
) -> dict[str, Any]:
    """Derive the exact v4 policy contract from one authenticated v3 row."""

    require_text(dated_question, "v4 dated question")
    _require(type(legacy_contract) is dict, "legacy validation contract changed type")
    legacy = dict(legacy_contract)
    question_terms = legacy.get("question_terms")
    question_actions = legacy.get("question_action_concepts")
    _require(
        legacy.get("format") == LEGACY_VALIDATION_CONTRACT_FORMAT
        and type(question_terms) is list
        and question_terms == list(normalized_terms(dated_question))
        and type(question_actions) is list
        and question_actions == list(canonical_action_concepts(dated_question)),
        "legacy validation contract lost its exact question binding",
    )
    temporal = _question_temporal_validation(
        dated_question, str(legacy.get("temporal_mode"))
    )
    advisory = _deterministic_advisory_policy(legacy, temporal)
    value = {
        **legacy,
        "deterministic_advisory_policy": advisory,
        "format": VALIDATION_CONTRACT_FORMAT,
        "legacy_validation_contract_sha256": identity_sha256(legacy),
        "question_memory_role": (
            "user" if _FIRST_PERSON_RE.search(dated_question) else None
        ),
        "question_focus_terms": list(
            _question_focus_terms(dated_question, question_actions)
        ),
        "temporal_validation": temporal,
    }
    assert_gold_blind(value, path="typed_final_validator_v4_contract")
    return value


def _legacy_contract_from_v4(
    contract: Mapping[str, Any],
    *,
    dated_question: str,
) -> dict[str, Any]:
    _require(type(contract) is dict, "v4 validation contract changed type")
    legacy = dict(contract)
    for key in (
        "deterministic_advisory_policy",
        "legacy_validation_contract_sha256",
        "question_memory_role",
        "question_focus_terms",
        "temporal_validation",
    ):
        _require(key in legacy, f"v4 validation contract omitted {key}")
        legacy.pop(key)
    legacy["format"] = LEGACY_VALIDATION_CONTRACT_FORMAT
    expected = upgrade_completion_validation_contract_v4(
        legacy, dated_question=dated_question
    )
    _require(dict(contract) == expected, "v4 validation contract derivation changed")
    return legacy


def _row_topic_terms(row: Mapping[str, Any]) -> frozenset[str]:
    values: list[str] = []
    for key in ("summary_terms", "entity_terms", "group_terms"):
        terms = row.get(key)
        _require(
            type(terms) is list
            and all(type(term) is str and bool(term) for term in terms),
            "v4 semantic topic terms changed",
        )
        values.extend(terms)
    return frozenset(values)


def _candidate_conditioned_complete_scope(
    allowed_rows: Sequence[Mapping[str, Any]],
    used_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], ...], bool, bool]:
    base = _rows_in_complete_proof_scope(allowed_rows, contract)
    # Compiled slots are stronger than lexical topic inference.  V4 never
    # narrows their proof universe.
    if contract.get("required_slot_ids"):
        return base, False, True
    question_actions = set(contract.get("question_action_concepts", ()))
    if question_actions and used_rows and all(
        row.get("status") == "completed"
        and question_actions & set(row.get("completed_action_concepts", ()))
        for row in used_rows
    ):
        base = tuple(
            row
            for row in base
            if row.get("status") == "completed"
            and question_actions & set(row.get("completed_action_concepts", ()))
        )
    if contract.get("question_memory_role") == "user" and used_rows and all(
        "user" in set(row.get("relation_terms", ())) for row in used_rows
    ):
        base = tuple(
            row
            for row in base
            if "user" in set(row.get("relation_terms", ()))
        )
    temporal = contract.get("temporal_validation")
    _require(type(temporal) is dict, "v4 aggregate temporal scope changed")
    if temporal.get("mode") == "bounded_calendar_lookback":
        if not all(_row_matches_temporal_validation(row, temporal) for row in used_rows):
            return base, False, False
        base = tuple(
            row for row in base if _row_matches_temporal_validation(row, temporal)
        )
    focus = set(contract.get("question_focus_terms", ()))
    if not focus:
        return base, False, True
    if not used_rows:
        return base, False, False
    shared = set(focus)
    for row in used_rows:
        shared &= set(_row_topic_terms(row))
    if not shared:
        return base, False, False
    narrowed = tuple(row for row in base if shared & set(_row_topic_terms(row)))
    if not narrowed:
        return base, False, False
    return narrowed, {
        row["semantic_unit_sha256"] for row in narrowed
    } != {row["semantic_unit_sha256"] for row in base}, True


def _complete_proof_scope_error_v4(
    allowed_rows: Sequence[Mapping[str, Any]],
    used_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[str | None, bool]:
    if contract.get("operation") not in _AGGREGATE_OPERATIONS:
        return None, False
    required, narrowed, topic_resolved = _candidate_conditioned_complete_scope(
        allowed_rows, used_rows, contract
    )
    if not topic_resolved:
        return "aggregate_scope_topic_unresolved", narrowed
    required_units = {row["semantic_unit_sha256"] for row in required}
    used_units = {row["semantic_unit_sha256"] for row in used_rows}
    if not required_units <= used_units:
        return "aggregate_scope_incomplete", narrowed
    return None, narrowed


def _row_calendar_date(row: Mapping[str, Any]) -> str | None:
    raw = row.get("date")
    if type(raw) is not str or not raw:
        return None
    normalized = raw.replace("/", "-")
    try:
        return datetime.fromisoformat(normalized).date().isoformat()
    except ValueError:
        for form in ("%Y-%m-%d", "%Y-%m"):
            try:
                return datetime.strptime(normalized, form).date().isoformat()
            except ValueError:
                pass
    return None


def _row_explicit_calendar_date(
    row: Mapping[str, Any],
    query_date: str,
) -> str | None:
    terms = row.get("summary_terms")
    _require(
        type(terms) is list
        and all(type(term) is str and bool(term) for term in terms),
        "v4 temporal summary terms changed",
    )
    months = tuple(
        dict.fromkeys(_MONTH_NUMBERS[term] for term in terms if term in _MONTH_NUMBERS)
    )
    days = tuple(
        dict.fromkeys(
            int(match.group("day"))
            for term in terms
            if (match := _ORDINAL_DAY_RE.match(term)) is not None
            and 1 <= int(match.group("day")) <= 31
        )
    )
    if len(months) != 1 or len(days) != 1:
        return None
    query = datetime.fromisoformat(query_date)
    year = query.year
    try:
        candidate = datetime(year, months[0], days[0])
    except ValueError:
        return None
    if candidate.date() > query.date():
        candidate = candidate.replace(year=year - 1)
    return candidate.date().isoformat()


def _row_effective_calendar_date(
    row: Mapping[str, Any],
    temporal: Mapping[str, Any],
) -> str | None:
    query_date = temporal.get("query_date")
    if type(query_date) is str and query_date:
        explicit = _row_explicit_calendar_date(row, query_date)
        if explicit is not None:
            return explicit
    return _row_calendar_date(row)


def _row_matches_temporal_validation(
    row: Mapping[str, Any],
    temporal: Mapping[str, Any],
) -> bool:
    moment = _row_effective_calendar_date(row, temporal)
    if moment is None:
        return False
    if temporal.get("mode") == "exact_calendar_day_offset":
        return moment == temporal.get("target_date")
    if temporal.get("mode") == "bounded_calendar_lookback":
        lower = temporal.get("lower_bound_date")
        query = temporal.get("query_date")
        return (
            type(lower) is str
            and type(query) is str
            and lower <= moment <= query
        )
    return True


def _temporal_selection_error_v4(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> str | None:
    if contract.get("temporal_mode") != TemporalMode.RELATIVE_SELECT.value:
        return None
    temporal = contract.get("temporal_validation")
    _require(type(temporal) is dict, "v4 temporal validation changed type")
    if temporal.get("mode") != "exact_calendar_day_offset":
        return "relative_temporal_target_unresolved"
    target = temporal.get("target_date")
    _require(type(target) is str and bool(target), "v4 temporal target changed")
    relevant = _rows_relevant_to_question(rows, contract)
    if not relevant or any(
        _row_effective_calendar_date(row, temporal) != target for row in relevant
    ):
        return "relative_temporal_target_mismatch"
    return None


def _v4_decision(
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
    basis = (
        validation_basis or parsed.validation_basis
        if valid
        else "invalid"
    )
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


def parse_typed_final_completion_v4(
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
    """Revalidate one sealed completion under the derived v4 policy."""

    require_text(dated_question, "v4 completion dated question")
    contract = dict(validation_contract)
    legacy = _legacy_contract_from_v4(contract, dated_question=dated_question)
    adjusted = dict(legacy)
    advisory_policy = contract["deterministic_advisory_policy"]
    if advisory_policy["binding_required"] is False:
        adjusted["deterministic_execution_advisory"] = None
    if legacy.get("operation") in _AGGREGATE_OPERATIONS:
        # V3 performs an action-wide completeness check.  V4 performs the
        # stronger question-anchored check below, so suppress only that legacy
        # check while retaining every other v3 schema/security/entailment gate.
        adjusted["operation"] = _INTERNAL_SCOPE_VALIDATED_OPERATION
    parsed = parse_typed_final_completion(
        completion,
        parent_prediction=parent_prediction,
        allowed_handle_ids=allowed_handle_ids,
        handle_group_by_id=handle_group_by_id,
        story_coherence=story_coherence,
        preservation_requirements=preservation_requirements,
        validation_contract=adjusted,
    )
    if not parsed.valid or parsed.decision != "replace":
        return _v4_decision(parsed, contract=contract)

    by_handle = legacy.get("by_handle")
    _require(type(by_handle) is dict, "v4 by-handle contract changed")
    used_rows = _validated_semantic_rows(by_handle, parsed.used_handle_ids)
    temporal_error = _temporal_selection_error_v4(used_rows, contract)
    if temporal_error is not None:
        return _v4_decision(
            parsed, contract=contract, error_code=temporal_error
        )

    basis = parsed.validation_basis
    if parsed.validation_basis == "model_attested":
        allowed_rows = _validated_semantic_rows(by_handle, allowed_handle_ids)
        scope_error, narrowed = _complete_proof_scope_error_v4(
            allowed_rows, used_rows, contract
        )
        if scope_error is not None:
            return _v4_decision(
                parsed, contract=contract, error_code=scope_error
            )
        if narrowed:
            basis = "model_attested_question_topic_complete_v4"
        if contract["temporal_mode"] == TemporalMode.RELATIVE_SELECT.value:
            basis = "model_attested_relative_exact_day_v4"
    return _v4_decision(parsed, contract=contract, validation_basis=basis)


def dated_question_from_plan_row(plan_row: Mapping[str, Any]) -> str:
    """Extract and authenticate the exact provider-visible dated question."""

    messages = plan_row.get("messages")
    _require(type(messages) is list, "v4 plan messages changed type")
    users = [
        row
        for row in messages
        if type(row) is dict and row.get("role") == "user"
    ]
    _require(len(users) == 1, "v4 plan must contain exactly one user message")
    content = users[0].get("content")
    _require(type(content) is str and bool(content), "v4 user message changed")
    try:
        provider_input = json.loads(content)
    except json.JSONDecodeError as exc:
        raise TypedMemoryFinalValidatorV4Error(
            "v4 provider input is not exact JSON"
        ) from exc
    _require(type(provider_input) is dict, "v4 provider input changed type")
    dated_question = require_text(
        provider_input.get("dated_question"), "v4 provider dated question"
    )
    _require(
        quote_sha256(dated_question)
        == require_sha256(
            plan_row.get("dated_question_sha256"), "v4 plan dated question"
        ),
        "v4 provider question differs from the sealed plan binding",
    )
    return dated_question


def materialize_typed_final_result_row_v4(
    plan_row: Mapping[str, Any],
    completion: str,
    *,
    completion_receipt_sha256: str,
    call_key_sha256: str,
    request_journal_sha256: str,
    response_journal_sha256: str,
) -> dict[str, Any]:
    """Materialize a new v4 row without changing its sealed v3 sources."""

    for value, label in (
        (completion_receipt_sha256, "v4 completion receipt"),
        (call_key_sha256, "v4 completion call key"),
        (request_journal_sha256, "v4 request journal"),
        (response_journal_sha256, "v4 response journal"),
    ):
        require_sha256(value, label)
    parent = require_text(plan_row.get("parent_prediction"), "v4 parent prediction")
    dated_question = dated_question_from_plan_row(plan_row)
    raw_contract = plan_row.get("validation_contract")
    _require(type(raw_contract) is dict, "v4 legacy plan contract changed type")
    contract = upgrade_completion_validation_contract_v4(
        raw_contract, dated_question=dated_question
    )
    parsed = parse_typed_final_completion_v4(
        completion,
        dated_question=dated_question,
        parent_prediction=parent,
        allowed_handle_ids=tuple(plan_row.get("allowed_handle_ids", ())),
        handle_group_by_id=dict(plan_row.get("handle_group_by_id", {})),
        story_coherence=dict(plan_row.get("story_coherence", {})),
        preservation_requirements=dict(
            plan_row.get("preservation_requirements", {})
        ),
        validation_contract=contract,
    )
    valid_replace = parsed.valid and parsed.decision == "replace"
    prediction = parsed.prediction if valid_replace else parent
    if valid_replace:
        source = (
            "typed_final_deterministic_validated_replacement_v4"
            if parsed.validation_basis == "deterministic_execution_agreement"
            else "typed_final_scalar_validated_replacement_v4"
            if parsed.validation_basis == "bounded_positive_scalar_agreement"
            else "typed_final_model_attested_replacement_v4"
        )
        decision = "replace"
        used = parsed.used_handle_ids
    elif parsed.valid:
        source = "typed_final_validated_keep_parent_v4"
        decision = "keep_parent"
        used = ()
    else:
        source = "typed_final_invalid_keep_parent_v4"
        decision = "invalid_keep_parent"
        used = ()
    body = {
        "call_key_sha256": call_key_sha256,
        "changed_from_parent": prediction != parent,
        "completion_receipt_sha256": completion_receipt_sha256,
        "dated_question_sha256": quote_sha256(dated_question),
        "decision": decision,
        "format": RESULT_ROW_FORMAT,
        "legacy_validation_contract_sha256": contract[
            "legacy_validation_contract_sha256"
        ],
        "ordinal": plan_row.get("ordinal"),
        "parent_prediction_sha256": quote_sha256(parent),
        "parse_error_code": parsed.error_code,
        "parse_receipt_sha256": parsed.receipt_sha256,
        "prediction": prediction,
        "prediction_sha256": quote_sha256(prediction),
        "prediction_source": source,
        "prompt_row_receipt_sha256": require_sha256(
            plan_row.get("prompt_row_receipt_sha256"), "v4 prompt row"
        ),
        "question_id": require_text(plan_row.get("question_id"), "v4 question ID"),
        "question_sha256": require_sha256(
            plan_row.get("question_sha256"), "v4 question"
        ),
        "request_journal_sha256": request_journal_sha256,
        "response_journal_sha256": response_journal_sha256,
        "retained_transformer_token_state_bytes": 0,
        "route_id": require_text(plan_row.get("route_id"), "v4 route"),
        "solver_valid": parsed.valid,
        "used_handle_ids": list(used),
        "validation_basis": parsed.validation_basis,
        "validation_contract_sha256": identity_sha256(contract),
        "validator_policy_format": VALIDATOR_POLICY_FORMAT,
    }
    _require(
        type(body["ordinal"]) is int and int(body["ordinal"]) >= 0,
        "v4 result ordinal changed",
    )
    body["source_row_sha256"] = identity_sha256(body)
    assert_gold_blind(body, path="typed_final_validator_v4_result_row")
    return body


__all__ = [
    "DECISION_FORMAT",
    "FORMAT",
    "RESULT_ROW_FORMAT",
    "TypedMemoryFinalValidatorV4Error",
    "VALIDATION_CONTRACT_FORMAT",
    "VALIDATOR_POLICY_FORMAT",
    "dated_question_from_plan_row",
    "judge_row_projection",
    "materialize_typed_final_result_row_v4",
    "parse_typed_final_completion_v4",
    "upgrade_completion_validation_contract_v4",
]
