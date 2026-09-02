from __future__ import annotations

import json
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.specialist_scoped_completion import (
    HARD_COMPLETE_CHAT_TOKEN_CAP,
    OUTPUT_TOKEN_RESERVE,
    SPECIALIST_ADVISORY_FORMAT,
    SPECIALIST_SYSTEM_PROMPT,
    SpecialistProofKind,
    SpecialistScopedCompletionError,
    compile_specialist_validation_scope,
    parse_specialist_scoped_completion,
    render_specialist_scoped_prompt,
)


def _sha(label: str) -> str:
    return quote_sha256(label)


def _semantic_row(
    label: str,
    *,
    terms: tuple[str, ...],
    numeric: float | None = None,
    unit: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    return {
        "action_concepts": [],
        "completed_action_concepts": [],
        "date": date,
        "entity_terms": list(terms),
        "group_terms": [],
        "item_receipt_sha256": _sha(f"item {label}"),
        "numeric_value": numeric,
        "relation_terms": [],
        "semantic_unit_sha256": _sha(f"unit {label}"),
        "status": "completed",
        "summary_terms": list(terms),
        "supported_slot_ids": [],
        "unit": unit,
    }


def _handle_contract(label: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "semantic_rows": [row],
        "status_values": ["completed"],
        "usable_item_receipt_sha256s": [row["item_receipt_sha256"]],
    }


def _compiled(
    advisories: list[dict[str, Any]],
    specialist_rows: dict[str, dict[str, Any]],
    specialist_groups: dict[str, str],
    *,
    question_terms: tuple[str, ...] = ("what", "memory"),
):
    parent_row = _semantic_row("parent", terms=("parent", "context"))
    terminal = ("H001", *specialist_rows)
    groups = {"H001": "G001", **specialist_groups}
    by_handle = {
        "H001": _handle_contract("parent", parent_row),
        **{
            handle: _handle_contract(handle, row)
            for handle, row in specialist_rows.items()
        },
    }
    validation_contract = {
        "by_handle": by_handle,
        "format": (
            "memory-condense-typed-memory-final-arm-v1-"
            "completion-validation-contract-v3"
        ),
        "include_proposed": False,
        "question_terms": list(question_terms),
    }
    provider_input = {
        "dated_question": "[Question asked at 2026/08/28] What does memory say?",
        "protected_parent_fallback": {
            "label": "fallback_not_evidence",
            "prediction": "Parent fallback",
            "prediction_sha256": _sha("Parent fallback"),
        },
        "response_schema": {
            "decision": "keep_parent|replace",
            "prediction": "nonempty exact text",
            "used_handle_ids": ["H001"],
        },
        "specialist_advisories": advisories,
        "typed_evidence": {"handles": list(terminal), "items": []},
    }
    prompt = render_specialist_scoped_prompt(provider_input)
    scope = compile_specialist_validation_scope(
        specialist_advisories=advisories,
        declared_specialist_advisories_sha256=identity_sha256(advisories),
        sealed_source_receipt_sha256=_sha("sealed source"),
        terminal_allowed_handle_ids=terminal,
        handle_group_by_id=groups,
        validation_contract=validation_contract,
        prompt_envelope=prompt,
    )
    return prompt, scope, validation_contract


def _completion(prediction: str, handles: list[str]) -> str:
    return json.dumps(
        {
            "decision": "replace",
            "prediction": prediction,
            "used_handle_ids": handles,
        },
        separators=(",", ":"),
    )


def test_renderer_declares_closed_specialist_scope_and_recounts_hard_budget() -> None:
    advisory = {
        "format": SPECIALIST_ADVISORY_FORMAT,
        "handle_ids": ["H700001"],
        "mechanism_id": "numeric_specialist",
        "operand_groups": [
            {
                "action_class": "buy",
                "entity_key": "feed",
                "handle_ids": ["H700001"],
                "operand_values": [50.0],
                "operation_mode": "sum",
                "source_group_handles": ["G700001"],
                "value_basis": "explicit_numeric_mention",
            }
        ],
        "purpose": "group operands",
    }
    prompt, _scope, _contract = _compiled(
        [advisory],
        {
            "H700001": _semantic_row(
                "feed", terms=("feed",), numeric=50.0, unit="lb"
            )
        },
        {"H700001": "G700001"},
    )

    assert prompt.messages[0]["content"] == SPECIALIST_SYSTEM_PROMPT
    assert "Never expand an aggregate or timeline" in SPECIALIST_SYSTEM_PROMPT
    assert "absence_certificate" in SPECIALIST_SYSTEM_PROMPT
    assert prompt.prompt_token_proxy + OUTPUT_TOKEN_RESERVE <= (
        HARD_COMPLETE_CHAT_TOKEN_CAP
    )
    assert prompt.provider_prompt_count == 0
    assert prompt.retained_transformer_token_state_bytes == 0

    huge = json.loads(prompt.messages[1]["content"])
    huge["padding"] = "token " * 10_000
    with pytest.raises(
        SpecialistScopedCompletionError,
        match="hard-budget contract",
    ):
        render_specialist_scoped_prompt(huge)


def test_numeric_groups_reduce_once_and_never_expand_into_parent_union() -> None:
    first = _sha("first numeric")
    second = _sha("second numeric")
    duplicate = _sha("duplicate second")
    advisory = {
        "candidate_handle_map": {
            first: "H700001",
            second: "H700002",
            duplicate: "H700003",
        },
        "mechanism_id": "numeric_specialist",
        "operand_groups": [
            {
                "action_class": "buy",
                "candidate_ids": [first],
                "entity_key": "layer_feed",
                "operand_values": [50.0],
                "operation_mode": "sum",
                "source_group_handles": ["G700001"],
                "value_basis": "explicit_numeric_mention",
            },
            {
                "action_class": "buy",
                "candidate_ids": [second, duplicate],
                "entity_key": "scratch_grain",
                "operand_values": [20.0],
                "operation_mode": "sum",
                "source_group_handles": ["G700002"],
                "value_basis": "explicit_numeric_mention",
            },
        ],
        "purpose": "group operands",
    }
    _prompt, scope, _contract = _compiled(
        [advisory],
        {
            "H700001": _semantic_row(
                "layer", terms=("layer", "feed"), numeric=50.0, unit="lb"
            ),
            "H700002": _semantic_row(
                "grain", terms=("scratch", "grain"), numeric=20.0, unit="lb"
            ),
            "H700003": _semantic_row(
                "grain duplicate",
                terms=("scratch", "grain"),
                numeric=20.0,
                unit="lb",
            ),
        },
        {
            "H700001": "G700001",
            "H700002": "G700002",
            "H700003": "G700002",
        },
    )

    parsed = parse_specialist_scoped_completion(
        _completion("70 pounds", ["H700001", "H700002"]),
        parent_prediction="Parent fallback",
        scope=scope,
    )
    duplicate_citation = parse_specialist_scoped_completion(
        _completion("70 pounds", ["H700001", "H700002", "H700003"]),
        parent_prediction="Parent fallback",
        scope=scope,
    )
    missing_group = parse_specialist_scoped_completion(
        _completion("50 pounds", ["H700001"]),
        parent_prediction="Parent fallback",
        scope=scope,
    )
    parent_escape = parse_specialist_scoped_completion(
        _completion("70 pounds", ["H001", "H700001", "H700002"]),
        parent_prediction="Parent fallback",
        scope=scope,
    )

    assert parsed.valid and parsed.decision == "replace"
    assert parsed.proof_kind == SpecialistProofKind.NUMERIC_OPERAND_GROUPS.value
    assert duplicate_citation.valid
    assert missing_group.error_code == "specialist_numeric_group_incomplete"
    assert parent_escape.error_code == "specialist_scope_escape"


def test_temporal_order_and_relative_roles_are_advisory_local() -> None:
    candidates = tuple(_sha(f"temporal {index}") for index in range(3))
    candidate_map = {
        candidate: f"H90000{index + 1}"
        for index, candidate in enumerate(candidates)
    }
    rows = {
        "H900001": _semantic_row(
            "woods", terms=("muir", "wood", "hike"), date="2023-03-10"
        ),
        "H900002": _semantic_row(
            "coast", terms=("big", "sur", "monterey"), date="2023-04-20"
        ),
        "H900003": _semantic_row(
            "park", terms=("solo", "yosemite", "camp"), date="2023-05-15"
        ),
    }
    groups = {handle: f"G90000{index + 1}" for index, handle in enumerate(rows)}
    order_advisory = {
        "absence_certificate": None,
        "format": SPECIALIST_ADVISORY_FORMAT,
        "handle_ids": list(rows),
        "mechanism_id": "temporal_specialist",
        "purpose": "order selected events",
        "temporal_bundle": {
            "ordered_handle_ids": list(rows),
            "original_population_count": 80,
            "predecessor_handle_id": "H900002",
            "query_time": "2023-06-01T03:56:00",
            "requested_cardinality": 3,
            "route": "temporal_order",
            "target_date": None,
            "terminal_selection_truncated": True,
            "winner_handle_id": "H900003",
        },
    }
    _prompt, order_scope, _contract = _compiled(
        [order_advisory], rows, groups
    )
    ordered = parse_specialist_scoped_completion(
        _completion(
            "Muir Woods hike; Big Sur and Monterey; solo Yosemite camping.",
            ["H900001", "H900002", "H900003"],
        ),
        parent_prediction="Parent fallback",
        scope=order_scope,
    )
    reordered = parse_specialist_scoped_completion(
        _completion(
            "solo Yosemite camping; Big Sur and Monterey; Muir Woods hike.",
            ["H900003", "H900002", "H900001"],
        ),
        parent_prediction="Parent fallback",
        scope=order_scope,
    )
    assert ordered.valid
    assert ordered.proof_kind == SpecialistProofKind.TEMPORAL_ORDER.value
    assert reordered.error_code == "specialist_temporal_order_scope"

    relative_advisory = json.loads(json.dumps(order_advisory))
    relative_advisory["temporal_bundle"].update(
        {
            "original_population_count": 3,
            "requested_cardinality": None,
            "route": "temporal_relative",
            "target_date": "2023-05-14",
            "terminal_selection_truncated": False,
        }
    )
    _prompt, relative_scope, _contract = _compiled(
        [relative_advisory], rows, groups
    )
    winner = parse_specialist_scoped_completion(
        _completion("You went solo camping in Yosemite.", ["H900003"]),
        parent_prediction="Parent fallback",
        scope=relative_scope,
    )
    wrong_role = parse_specialist_scoped_completion(
        _completion("You visited Big Sur and Monterey.", ["H900002"]),
        parent_prediction="Parent fallback",
        scope=relative_scope,
    )
    predecessor_contamination = parse_specialist_scoped_completion(
        _completion(
            "You went solo camping in Yosemite.",
            ["H900003", "H900002"],
        ),
        parent_prediction="Parent fallback",
        scope=relative_scope,
    )
    assert winner.valid
    assert winner.proof_kind == SpecialistProofKind.TEMPORAL_RELATIVE.value
    assert wrong_role.error_code == "specialist_temporal_role_scope"
    assert predecessor_contamination.error_code == "specialist_temporal_role_scope"


def test_parent_equivalent_replace_is_narrow_and_scope_checked() -> None:
    candidates = tuple(_sha(f"parent equivalent {index}") for index in range(2))
    rows = {
        "H900001": _semantic_row(
            "mesh", terms=("mesh", "network", "setup"), date="2023-03-10"
        ),
        "H900002": _semantic_row(
            "router", terms=("wifi", "router", "setup"), date="2023-04-20"
        ),
    }
    advisory = {
        "absence_certificate": None,
        "candidate_handle_map": {
            candidate: handle
            for candidate, handle in zip(candidates, rows, strict=True)
        },
        "mechanism_id": "temporal_specialist",
        "purpose": "order selected events",
        "temporal_bundle": {
            "ordered_candidate_ids": list(candidates),
            "ordered_handle_ids": list(rows),
            "original_population_count": 20,
            "predecessor_candidate_id": candidates[0],
            "predecessor_handle_id": "H900001",
            "query_time": "2023-05-01T00:00:00",
            "requested_cardinality": 2,
            "route": "temporal_order",
            "target_date": None,
            "terminal_selection_truncated": True,
            "winner_candidate_id": candidates[1],
            "winner_handle_id": "H900002",
        },
    }
    _prompt, scope, _contract = _compiled(
        [advisory],
        rows,
        {"H900001": "G900001", "H900002": "G900002"},
    )
    parent = "The mesh network system was set up first."
    punctuated_parent = (
        "The woman selling jam at the farmer\N{RIGHT SINGLE QUOTATION MARK}s "
        "market."
    )

    exact = parse_specialist_scoped_completion(
        _completion(parent, list(rows)),
        parent_prediction=parent,
        scope=scope,
    )
    apostrophe = parse_specialist_scoped_completion(
        _completion("The woman selling jam at the farmer's market.", list(rows)),
        parent_prediction=punctuated_parent,
        scope=scope,
    )
    unknown = parse_specialist_scoped_completion(
        _completion(parent, ["H999999"]),
        parent_prediction=parent,
        scope=scope,
    )
    changed = parse_specialist_scoped_completion(
        _completion("The coffee maker was purchased first.", list(rows)),
        parent_prediction=parent,
        scope=scope,
    )
    left_quote = parse_specialist_scoped_completion(
        _completion(
            "The woman selling jam at the farmer\N{LEFT SINGLE QUOTATION MARK}s "
            "market.",
            list(rows),
        ),
        parent_prediction=punctuated_parent,
        scope=scope,
    )
    whitespace = parse_specialist_scoped_completion(
        _completion("The mesh network system was set up first. ", list(rows)),
        parent_prediction=parent,
        scope=scope,
    )

    assert exact.valid and exact.decision == "keep_parent"
    assert exact.prediction == parent
    assert exact.used_handle_ids == ()
    assert exact.validation_basis == "normalized_identical_replace"
    assert apostrophe.valid and apostrophe.decision == "keep_parent"
    assert apostrophe.prediction == punctuated_parent
    assert apostrophe.validation_basis == "right_single_quote_equivalent_replace"
    assert unknown.error_code == "unknown_handle"
    assert changed.error_code == "specialist_temporal_order_entailment"
    assert left_quote.error_code == "specialist_temporal_order_entailment"
    assert whitespace.error_code == "replace_contract"


def test_temporal_numeric_winner_requires_exact_value_unit_and_no_conflict() -> None:
    predecessor = _sha("handbag predecessor")
    winner = _sha("handbag winner")
    slot = _sha("handbag spend slot")
    predecessor_row = _semantic_row(
        "older handbag price",
        terms=("designer", "handbag", "purchase", "two", "thousand"),
        numeric=2_000.0,
        unit="$",
        date="2023-01-15",
    )
    winner_row = _semantic_row(
        "latest handbag price",
        terms=(
            "designer",
            "handbag",
            "purchase",
            "pretty",
            "penny",
            "800",
        ),
        numeric=800.0,
        unit="$",
        date="2023-05-23",
    )
    for row in (predecessor_row, winner_row):
        row["numeric_qualifier"] = "exact"
        row["numeric_role"] = "operand"
        row["supported_slot_ids"] = [slot]
    advisory = {
        "absence_certificate": None,
        "candidate_handle_map": {
            predecessor: "H900011",
            winner: "H900012",
        },
        "mechanism_id": "temporal_insufficiency_specialist_v1",
        "purpose": "select latest transaction state",
        "temporal_bundle": {
            "ordered_candidate_ids": [predecessor, winner],
            "ordered_handle_ids": ["H900011", "H900012"],
            "original_population_count": 42,
            "predecessor_candidate_id": predecessor,
            "predecessor_handle_id": "H900011",
            "query_time": "2023-05-30T23:20:00",
            "requested_cardinality": None,
            "route": "temporal_latest",
            "target_date": None,
            "terminal_selection_truncated": True,
            "winner_candidate_id": winner,
            "winner_handle_id": "H900012",
        },
    }
    _prompt, scope, _contract = _compiled(
        [advisory],
        {"H900011": predecessor_row, "H900012": winner_row},
        {"H900011": "G900011", "H900012": "G900012"},
        question_terms=(
            "question",
            "ask",
            "2023",
            "05",
            "30",
            "tue",
            "23",
            "20",
            "much",
            "spend",
            "designer",
            "handbag",
        ),
    )

    exact_symbol = parse_specialist_scoped_completion(
        _completion("The designer handbag cost $800.", ["H900012"]),
        parent_prediction="$2,000",
        scope=scope,
    )
    exact_word_unit = parse_specialist_scoped_completion(
        _completion(
            "You spent 800 dollars on the designer handbag.",
            ["H900012"],
        ),
        parent_prediction="$2,000",
        scope=scope,
    )
    exact_with_date = parse_specialist_scoped_completion(
        _completion(
            "On May 23, 2023, the designer handbag cost $800.",
            ["H900012"],
        ),
        parent_prediction="$2,000",
        scope=scope,
    )
    vague_lexical_anchor = parse_specialist_scoped_completion(
        _completion(
            "The designer handbag cost a pretty penny.",
            ["H900012"],
        ),
        parent_prediction="$2,000",
        scope=scope,
    )
    wrong_value = parse_specialist_scoped_completion(
        _completion(
            "The pretty penny designer handbag cost $2,000.",
            ["H900012"],
        ),
        parent_prediction="$2,000",
        scope=scope,
    )
    contradictory_value = parse_specialist_scoped_completion(
        _completion(
            "The pretty penny handbag cost $800, not $2,000.",
            ["H900012"],
        ),
        parent_prediction="$2,000",
        scope=scope,
    )
    approximate_value = parse_specialist_scoped_completion(
        _completion(
            "The pretty penny handbag cost about $800.",
            ["H900012"],
        ),
        parent_prediction="$2,000",
        scope=scope,
    )
    contextual_currency = parse_specialist_scoped_completion(
        _completion(
            "The pretty penny handbag price was 800.",
            ["H900012"],
        ),
        parent_prediction="$2,000",
        scope=scope,
    )
    wrong_unit = parse_specialist_scoped_completion(
        _completion(
            "The pretty penny handbag weighed 800 pounds.",
            ["H900012"],
        ),
        parent_prediction="$2,000",
        scope=scope,
    )

    assert exact_symbol.valid
    assert exact_word_unit.valid
    assert exact_with_date.valid
    assert (
        vague_lexical_anchor.error_code
        == "specialist_temporal_winner_numeric_entailment"
    )
    assert (
        wrong_value.error_code
        == "specialist_temporal_winner_numeric_disagreement"
    )
    assert (
        contradictory_value.error_code
        == "specialist_temporal_winner_numeric_disagreement"
    )
    assert (
        approximate_value.error_code
        == "specialist_temporal_winner_numeric_prediction_unsafe"
    )
    assert contextual_currency.valid
    assert wrong_unit.error_code == "specialist_temporal_winner_numeric_entailment"


def test_absence_certificate_allows_only_scoped_insufficiency() -> None:
    support_candidate = _sha("tomato support")
    advisory = {
        "absence_certificate": {
            "applicable": True,
            "every_exact_entity_posting_scanned": True,
            "every_scoped_source_row_scanned": True,
            "may_conclude_operator_insufficient": True,
            "physical_content_rows_scanned": 200,
            "physical_sentence_windows_scanned": 900,
            "provider_instruction": (
                "Report insufficient memory evidence for the missing chili count."
            ),
            "scope_definition": "complete immutable entity scope",
            "scoped_content_row_count": 20,
            "scoped_source_count": 3,
            "semantic_absence_may_be_inferred": False,
            "slot_coverage": [
                {
                    "entity_assertion_source_count": 1,
                    "entity_assertion_window_count": 1,
                    "exact_entity_terms": ["tomato"],
                    "explicit_numeric_assertion_source_count": 1,
                    "explicit_numeric_assertion_window_count": 1,
                    "explicit_numeric_operand_missing": False,
                    "scope_has_grounded_predicate_assertion": True,
                    "selected_supporting_handle_ids": ["H900001"],
                    "slot_label": "tomatoes",
                },
                {
                    "entity_assertion_source_count": 0,
                    "entity_assertion_window_count": 0,
                    "exact_entity_terms": ["chili", "pepper"],
                    "explicit_numeric_assertion_source_count": 0,
                    "explicit_numeric_assertion_window_count": 0,
                    "explicit_numeric_operand_missing": True,
                    "scope_has_grounded_predicate_assertion": True,
                    "selected_supporting_handle_ids": [],
                    "slot_label": "chili peppers",
                },
            ],
        },
        "format": SPECIALIST_ADVISORY_FORMAT,
        "handle_ids": ["H900001"],
        "mechanism_id": "temporal_insufficiency_specialist",
        "purpose": "certify scoped missing operand",
        "temporal_bundle": None,
    }
    _prompt, scope, _contract = _compiled(
        [advisory],
        {
            "H900001": _semantic_row(
                "tomatoes", terms=("tomato", "plant"), numeric=5.0
            )
        },
        {"H900001": "G900001"},
    )
    valid = parse_specialist_scoped_completion(
        _completion(
            "There are 5 tomato plants, but memory evidence is insufficient "
            "to determine the chili pepper count.",
            ["H900001"],
        ),
        parent_prediction="Parent fallback",
        scope=scope,
    )
    invented = parse_specialist_scoped_completion(
        _completion(
            "There are 5 tomato plants and 7 chili pepper plants, but evidence "
            "is insufficient.",
            ["H900001"],
        ),
        parent_prediction="Parent fallback",
        scope=scope,
    )
    asserted = parse_specialist_scoped_completion(
        _completion("There are 5 tomato plants.", ["H900001"]),
        parent_prediction="Parent fallback",
        scope=scope,
    )
    hyphenated = parse_specialist_scoped_completion(
        _completion(
            "Insufficient memory evidence for the chili-pepper plant count; "
            "tomatoes: 5.",
            ["H900001"],
        ),
        parent_prediction="Parent fallback",
        scope=scope,
    )
    lookalike_compound = parse_specialist_scoped_completion(
        _completion(
            "Insufficient memory evidence for a chili-pepperoni plant count; "
            "tomatoes: 5.",
            ["H900001"],
        ),
        parent_prediction="Parent fallback",
        scope=scope,
    )
    concatenated = parse_specialist_scoped_completion(
        _completion(
            "Insufficient memory evidence for the chilipepper plant count; "
            "tomatoes: 5.",
            ["H900001"],
        ),
        parent_prediction="Parent fallback",
        scope=scope,
    )

    assert valid.valid
    assert valid.proof_kind == SpecialistProofKind.ABSENCE_CERTIFICATE.value
    assert hyphenated.valid
    assert hyphenated.proof_kind == SpecialistProofKind.ABSENCE_CERTIFICATE.value
    assert invented.error_code == "specialist_absence_unsupported_numeric_value"
    assert asserted.error_code == "specialist_absence_not_expressed"
    assert (
        lookalike_compound.error_code
        == "specialist_absence_missing_slot_not_named"
    )
    assert concatenated.error_code == "specialist_absence_missing_slot_not_named"


def test_temporal_interval_uses_winner_to_query_time_calendar_math() -> None:
    predecessor = _sha("interval predecessor")
    winner = _sha("interval winner")
    advisory = {
        "absence_certificate": None,
        "candidate_handle_map": {
            predecessor: "H900001",
            winner: "H900002",
        },
        "mechanism_id": "temporal_insufficiency_specialist",
        "purpose": "compute scoped interval",
        "temporal_bundle": {
            "ordered_candidate_ids": [predecessor, winner],
            "ordered_handle_ids": ["H900001", "H900002"],
            "original_population_count": 8,
            "predecessor_candidate_id": predecessor,
            "predecessor_handle_id": "H900001",
            "query_time": "2023-03-25T17:18:00",
            "requested_cardinality": None,
            "route": "temporal_interval",
            "target_date": None,
            "terminal_selection_truncated": True,
            "winner_candidate_id": winner,
            "winner_handle_id": "H900002",
        },
    }
    _prompt, scope, _contract = _compiled(
        [advisory],
        {
            "H900001": _semantic_row(
                "comparison", terms=("earlier", "comparison"), date="2022-09-01"
            ),
            "H900002": _semantic_row(
                "start", terms=("trip", "friend"), date="2022-10-20"
            ),
        },
        {"H900001": "G900001", "H900002": "G900002"},
    )

    accepted = parse_specialist_scoped_completion(
        _completion("5 months", ["H900002"]),
        parent_prediction="0 months",
        scope=scope,
    )
    zero = parse_specialist_scoped_completion(
        _completion("0 months", ["H900002"]),
        parent_prediction="0 months",
        scope=scope,
    )
    seven = parse_specialist_scoped_completion(
        _completion("7 months", ["H900002"]),
        parent_prediction="0 months",
        scope=scope,
    )
    predecessor_contamination = parse_specialist_scoped_completion(
        _completion("5 months", ["H900002", "H900001"]),
        parent_prediction="0 months",
        scope=scope,
    )

    assert accepted.valid
    assert accepted.proof_kind == SpecialistProofKind.TEMPORAL_INTERVAL.value
    assert zero.error_code == "specialist_temporal_interval_disagreement"
    assert seven.error_code == "specialist_temporal_interval_disagreement"
    assert (
        predecessor_contamination.error_code
        == "specialist_temporal_interval_scope"
    )


def test_profile_preference_is_grounded_in_one_specialist_cluster() -> None:
    first = _sha("profile first")
    second = _sha("profile second")
    advisory = {
        "format": SPECIALIST_ADVISORY_FORMAT,
        "handle_ids": ["H800001", "H800002"],
        "mechanism_id": "profile_preference_specialist_v1",
        "purpose": "personalize from one coherent first-person cluster",
    }
    standup = _semantic_row(
        "standup", terms=("stand-up", "comedian", "storytell", "mulaney")
    )
    standup["relation_terms"] = ["author", "by", "user"]
    streaming = _semantic_row(
        "streaming", terms=("netflix", "comedy", "special")
    )
    streaming["relation_terms"] = ["author", "by", "user"]
    _prompt, scope, _contract = _compiled(
        [advisory],
        {"H800001": standup, "H800002": streaming},
        {"H800001": "G800001", "H800002": "G800001"},
    )

    accepted = parse_specialist_scoped_completion(
        _completion(
            "Try a John Mulaney comedy special on Netflix for its storytelling.",
            ["H800001", "H800002"],
        ),
        parent_prediction="Parent fallback",
        scope=scope,
    )
    ungrounded = parse_specialist_scoped_completion(
        _completion("Try an unrelated cooking documentary.", ["H800001"]),
        parent_prediction="Parent fallback",
        scope=scope,
    )
    parent_escape = parse_specialist_scoped_completion(
        _completion("Try a comedy special.", ["H001"]),
        parent_prediction="Parent fallback",
        scope=scope,
    )

    assert accepted.valid
    assert accepted.proof_kind == SpecialistProofKind.PROFILE_PREFERENCE.value
    assert ungrounded.error_code == "specialist_profile_text_entailment"
    assert parent_escape.error_code == "specialist_scope_escape"


def test_provenance_tamper_fails_closed_before_completion_parsing() -> None:
    candidate = _sha("candidate")
    advisory = {
        "candidate_handle_map": {candidate: "H700001"},
        "mechanism_id": "numeric_specialist",
        "operand_groups": [
            {
                "action_class": "buy",
                "candidate_ids": [candidate],
                "entity_key": "feed",
                "operand_values": [50.0],
                "operation_mode": "sum",
                "source_group_handles": ["G700001"],
                "value_basis": "explicit_numeric_mention",
            }
        ],
        "purpose": "group operands",
    }
    prompt, _scope, contract = _compiled(
        [advisory],
        {
            "H700001": _semantic_row(
                "feed", terms=("feed",), numeric=50.0, unit="lb"
            )
        },
        {"H700001": "G700001"},
    )

    with pytest.raises(
        SpecialistScopedCompletionError,
        match="advisory seal differs",
    ):
        compile_specialist_validation_scope(
            specialist_advisories=[advisory],
            declared_specialist_advisories_sha256=_sha("tampered"),
            sealed_source_receipt_sha256=_sha("sealed source"),
            terminal_allowed_handle_ids=("H001", "H700001"),
            handle_group_by_id={"H001": "G001", "H700001": "G700001"},
            validation_contract=contract,
            prompt_envelope=prompt,
        )
