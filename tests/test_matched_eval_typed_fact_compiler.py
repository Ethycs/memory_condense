from __future__ import annotations

import json

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from tools.matched_eval.typed_fact_compiler import (
    ANSWER_OUTPUT_TOKEN_RESERVE,
    COMPILER_OUTPUT_TOKEN_RESERVE,
    HARD_PROMPT_TOKEN_CAP,
    MAX_COMPILER_FACTS,
    TypedFactCompilerError,
    build_answer_messages,
    build_compiler_input,
    build_compiler_messages,
    parse_compiler_completion,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


QUESTION = (
    "[Question asked at 2026/05/30 (Sat) 14:18]\n"
    "How much did I spend on the Aurora lamp?"
)


def _source(*, question: str = QUESTION) -> dict:
    spec = compile_typed_operator_spec(question)
    slot_ids = [row.slot_id for row in spec.required_slots]
    return {
        "provider_projection": {
            "provider_input": {
                "dated_question": question,
                "protected_parent_fallback": {
                    "label": "fallback_not_evidence",
                    "prediction": "secret parent value",
                    "prediction_sha256": "0" * 64,
                },
                "response_schema": {
                    "decision": "keep_parent|replace",
                    "prediction": "nonempty exact text",
                    "used_handle_ids": ["H001"],
                },
                "story_coherence": {
                    "group_links": [],
                    "incompatible_group_pairs": [["G001", "G002"]],
                },
                "typed_evidence": {
                    "conflict_policy": "quarantine",
                    "frontier": {
                        "available_handle_ids": ["H001", "H002"],
                        "closed": False,
                        "mode": "bounded",
                    },
                    "handles": [
                        {
                            "group_handle": "G001",
                            "handle_id": "H001",
                            "origin": "map",
                            "provenance_grade": "exact_citation",
                        },
                        {
                            "group_handle": "G002",
                            "handle_id": "H002",
                            "origin": "direct_pointer",
                            "provenance_grade": "direct_pointer",
                        },
                    ],
                    "items": [
                        {
                            "date": "2026-05-03",
                            "entity_key": "Aurora lamp",
                            "handle_ids": ["H001"],
                            "included": True,
                            "kind": "operand",
                            "numeric_value": 30.0,
                            "status": "completed",
                            "summary": (
                                "On May 3, 2026, I completed buying the Aurora "
                                "lamp for 30 USD."
                            ),
                            "supported_slot_ids": slot_ids,
                            "unit": "USD",
                        },
                        {
                            "handle_ids": ["H002"],
                            "included": True,
                            "kind": "direct",
                            "status": "unknown",
                            "summary": "I bought a blue kettle for the kitchen.",
                            "supported_slot_ids": [],
                        },
                    ],
                    "operator_spec": spec.projection(),
                },
            }
        }
    }


def _fact(*, value: float = 30.0, handle: str = "H001", quote: str | None = None) -> dict:
    slot_ids = build_compiler_input(_source())["operator_spec"]["required_slots"]
    return {
        "citations": [
            {
                "handle_id": handle,
                "quote": quote or "completed buying the Aurora lamp for 30 USD",
            }
        ],
        "date": "2026-05-03",
        "entity": "Aurora lamp",
        "kind": "operand",
        "numeric_value": value,
        "slot_ids": [row["slot_id"] for row in slot_ids],
        "status": "completed",
        "text": "I completed buying the Aurora lamp for 30 USD.",
        "unit": "USD",
    }


def test_compiler_prompt_is_bounded_opaque_and_excludes_parent() -> None:
    messages = build_compiler_messages(_source())
    rendered = json.dumps(messages)

    assert "secret parent value" not in rendered
    assert "protected_parent" not in rendered
    assert "H001" in rendered and "G001" in rendered
    assert "Aurora lamp" in rendered
    assert count_chat_prompt_token_proxy(messages) + COMPILER_OUTPUT_TOKEN_RESERVE <= HARD_PROMPT_TOKEN_CAP


def test_unknown_handle_rejects_but_ungrounded_numeric_is_stripped() -> None:
    response = json.dumps(
        {
            "facts": [
                _fact(),
                _fact(handle="H999"),
                _fact(value=99.0),
            ]
        }
    )

    compiled = parse_compiler_completion(_source(), response)

    assert len(compiled.accepted_before_dedup) == 2
    assert len(compiled.rejected) == 1
    assert compiled.accepted_before_dedup[1].numeric_value is None
    assert compiled.packet.valid is True
    assert compiled.packet.retained_handle_ids == ("H001",)
    projected = compiled.packet.typed_evidence_projection()
    assert projected["items"][0]["citations"][0]["quote"] == (
        "completed buying the Aurora lamp for 30 USD"
    )
    assert projected["items"][0]["numeric_value"] == 30.0
    assert projected["handles"] == [
        {
            "group_handle": "G001",
            "handle_id": "H001",
            "origin": "map",
            "provenance_grade": "exact_citation",
        }
    ]

    nonexact = _fact(quote="this quote is absent from the source summary")
    rejected = parse_compiler_completion(
        _source(), json.dumps({"facts": [nonexact]})
    )
    assert rejected.packet.valid is False
    assert "not exact admitted evidence" in rejected.rejected[0].reason


def test_validation_precedes_dedup_and_duplicate_citations_merge() -> None:
    source = _source()
    source["provider_projection"]["provider_input"]["typed_evidence"]["handles"][1][
        "group_handle"
    ] = "G001"
    source["provider_projection"]["provider_input"]["typed_evidence"]["items"].append(
        {
            "date": "2026-05-03",
            "entity_key": "Aurora lamp",
            "handle_ids": ["H002"],
            "included": True,
            "kind": "operand",
            "numeric_value": 30.0,
            "status": "completed",
            "summary": "I completed buying the Aurora lamp for 30 USD.",
            "supported_slot_ids": [
                row["slot_id"]
                for row in source["provider_projection"]["provider_input"]
                ["typed_evidence"]["operator_spec"]["required_slots"]
            ],
            "unit": "USD",
        }
    )
    response = json.dumps(
        {
            "facts": [
                _fact(),
                _fact(handle="H002", quote="completed buying the Aurora lamp for 30 USD"),
                _fact(value=99.0),
            ]
        }
    )

    compiled = parse_compiler_completion(source, response)

    assert len(compiled.accepted_before_dedup) == 3
    assert len(compiled.rejected) == 0
    assert compiled.duplicate_count == 1
    assert len(compiled.packet.facts) == 2
    assert compiled.packet.facts[0].handle_ids == ("H001", "H002")


def test_malformed_completion_becomes_invalid_fallback_packet() -> None:
    compiled = parse_compiler_completion(_source(), '{"facts":[],"facts":[]}')

    assert compiled.packet.valid is False
    assert compiled.packet.facts == ()
    assert compiled.packet.invalid_reason
    assert compiled.rejected[0].source_index == -1
    assert compiled.packet.provider_calls == 0
    assert compiled.packet.retained_transformer_token_state_bytes == 0


def test_empty_slot_route_uses_question_density_to_rank_facts() -> None:
    question = (
        "[Question asked at 2026/05/30 (Sat) 14:18]\n"
        "How many documentaries did I watch on Netflix?"
    )
    source = _source(question=question)
    response = {
        "facts": [
            {
                "citations": [{"handle_id": "H002", "quote": "bought a blue kettle"}],
                "date": None,
                "entity": "blue kettle",
                "kind": "event",
                "numeric_value": None,
                "slot_ids": [],
                "status": "completed",
                "text": "I bought a blue kettle.",
                "unit": None,
            },
            {
                "citations": [
                    {
                        "handle_id": "H001",
                        "quote": "completed buying the Aurora lamp for 30 USD",
                    }
                ],
                "date": None,
                "entity": "Aurora lamp",
                "kind": "event",
                "numeric_value": None,
                "slot_ids": [],
                "status": "completed",
                "text": "I completed buying the Aurora lamp.",
                "unit": None,
            },
        ]
    }
    # Make H001 explicitly about the question while preserving exact citation.
    item = source["provider_projection"]["provider_input"]["typed_evidence"]["items"][0]
    item["summary"] = "I completed watching three documentaries on Netflix."
    item["entity_key"] = "Netflix documentaries"
    response["facts"][1].update(
        {
            "citations": [
                {
                    "handle_id": "H001",
                    "quote": "completed watching three documentaries on Netflix",
                }
            ],
            "entity": "Netflix documentaries",
            "text": "I completed watching three documentaries on Netflix.",
        }
    )

    compiled = parse_compiler_completion(source, json.dumps(response))

    assert not source["provider_projection"]["provider_input"]["typed_evidence"]["operator_spec"]["required_slots"]
    assert compiled.packet.facts[0].fact_id == "F002"
    assert set(compiled.packet.facts[0].question_term_hits) >= {"documentary", "netflix"}


def test_answer_prompt_restores_parent_only_after_compilation() -> None:
    compiled = parse_compiler_completion(_source(), json.dumps({"facts": [_fact()]}))
    messages = build_answer_messages(_source(), compiled.packet)
    rendered = json.dumps(messages)

    assert "secret parent value" in rendered
    assert compiled.packet.receipt_sha256 in rendered
    assert count_chat_prompt_token_proxy(messages) + ANSWER_OUTPUT_TOKEN_RESERVE <= HARD_PROMPT_TOKEN_CAP


def test_incompatible_groups_cannot_share_a_fact_or_cross_group_dedup() -> None:
    source = _source()
    mixed = {
        "citations": [
            {
                "handle_id": "H001",
                "quote": "Aurora lamp",
            },
            {
                "handle_id": "H002",
                "quote": "blue kettle",
            },
        ],
        "date": None,
        "entity": None,
        "kind": "claim",
        "numeric_value": None,
        "slot_ids": [],
        "status": None,
        "text": "Aurora lamp and blue kettle",
        "unit": None,
    }
    rejected = parse_compiler_completion(source, json.dumps({"facts": [mixed]}))
    assert rejected.packet.valid is False
    assert "incompatible story groups" in rejected.rejected[0].reason

    # Identical valid facts from disjoint incompatible groups stay separate.
    second_item = source["provider_projection"]["provider_input"]["typed_evidence"]["items"][1]
    second_item.update(
        {
            "entity_key": "Aurora lamp",
            "kind": "operand",
            "summary": "I completed buying the Aurora lamp for 30 USD.",
            "status": "completed",
            "numeric_value": 30.0,
            "unit": "USD",
            "date": "2026-05-03",
            "supported_slot_ids": [
                row["slot_id"]
                for row in source["provider_projection"]["provider_input"]
                ["typed_evidence"]["operator_spec"]["required_slots"]
            ],
        }
    )
    compiled = parse_compiler_completion(
        source,
        json.dumps(
            {
                "facts": [
                    _fact(),
                    _fact(
                        handle="H002",
                        quote="completed buying the Aurora lamp for 30 USD",
                    ),
                ]
            }
        ),
    )
    assert compiled.duplicate_count == 0
    assert len(compiled.packet.facts) == 2


def test_null_structured_free_text_is_discarded_and_summary_is_exact_quote() -> None:
    arbitrary = _fact()
    arbitrary.update(
        {
            "date": None,
            "entity": None,
            "numeric_value": None,
            "slot_ids": [],
            "status": None,
            "text": "Aliens secretly stole the Aurora lamp.",
            "unit": None,
        }
    )
    compiled = parse_compiler_completion(
        _source(), json.dumps({"facts": [_fact(), arbitrary]})
    )

    assert len(compiled.accepted_before_dedup) == 2
    assert not compiled.rejected
    assert all("Aliens" not in fact.text for fact in compiled.accepted_before_dedup)
    item = compiled.packet.typed_evidence_projection()["items"][0]
    assert item["summary"] == "completed buying the Aurora lamp for 30 USD"
    assert "I completed" not in item["summary"]


def test_uncovered_and_original_unresolved_slots_force_invalid_fallback() -> None:
    source = _source()
    operator = source["provider_projection"]["provider_input"]["typed_evidence"][
        "operator_spec"
    ]
    slot_id = "a" * 64
    operator["required_slots"] = [
        {
            "kind": "predicate",
            "label": "unseen predicate",
            "match_terms": ["unseen", "predicate"],
            "minimum_match_term_count": 2,
            "relation_constraint": None,
            "requires_numeric": False,
            "slot_id": slot_id,
        }
    ]
    for item in source["provider_projection"]["provider_input"]["typed_evidence"]["items"]:
        item["supported_slot_ids"] = []
    fact = _fact()
    fact["slot_ids"] = []

    compiled = parse_compiler_completion(source, json.dumps({"facts": [fact]}))
    assert compiled.packet.facts
    assert compiled.packet.valid is False
    assert compiled.packet.invalid_reason == "required_slots_unresolved"
    assert compiled.packet.typed_evidence_projection()["frontier"][
        "unresolved_slot_ids"
    ] == [slot_id]
    with pytest.raises(TypedFactCompilerError, match="valid fact packet"):
        build_answer_messages(source, compiled.packet)

    source["provider_projection"]["provider_input"]["typed_evidence"]["frontier"][
        "unresolved_slot_ids"
    ] = ["original-unresolved-slot"]
    covered = parse_compiler_completion(source, json.dumps({"facts": [_fact()]}))
    assert covered.packet.valid is False
    assert covered.packet.typed_evidence_projection()["frontier"][
        "unresolved_slot_ids"
    ] == ["original-unresolved-slot", slot_id]


def test_field_salvage_normalizes_status_strips_fields_and_derives_slots() -> None:
    source = _source()
    item = source["provider_projection"]["provider_input"]["typed_evidence"]["items"][0]
    item["summary"] = "I am planning an Aurora lamp purchase for 30 USD."
    item["status"] = "unknown"
    item["supported_slot_ids"] = []
    quote = "planning an Aurora lamp purchase for 30 USD"
    raw = {
        "citations": [{"handle_id": "H001", "quote": quote}],
        "date": "2099-01-01",
        "entity": "Mars colony",
        "kind": "event",
        "numeric_value": 99,
        "slot_ids": ["not-a-real-slot"],
        "status": "planned",
        "text": "An invented provider paraphrase.",
        "unit": "parsecs",
    }
    eligible = {**raw, "status": "eligible"}

    compiled = parse_compiler_completion(
        source, json.dumps({"facts": [raw, eligible]})
    )

    assert not compiled.rejected
    planned, normalized_eligible = compiled.accepted_before_dedup
    assert planned.text == quote
    assert planned.kind == "operand"
    assert planned.status == "proposed"
    assert normalized_eligible.status is None
    assert planned.entity is None
    assert planned.numeric_value is None
    assert planned.unit is None
    assert planned.date is None
    real_slot_ids = {
        row["slot_id"]
        for row in source["provider_projection"]["provider_input"]["typed_evidence"]
        ["operator_spec"]["required_slots"]
    }
    assert set(planned.slot_ids) <= real_slot_ids
    assert "not-a-real-slot" not in planned.slot_ids


def test_final_packet_evicts_lowest_ranked_facts_until_full_projection_fits() -> None:
    facts = []
    for quote in (
        "completed buying",
        "buying the Aurora lamp",
        "Aurora lamp for 30 USD",
        "May 3, 2026",
        "completed buying the Aurora",
        "lamp for 30 USD",
    ):
        fact = _fact()
        fact["citations"][0]["quote"] = quote
        facts.append(fact)
    response = json.dumps({"facts": facts})
    full = parse_compiler_completion(_source(), response)
    constrained_cap = full.packet.packet_token_proxy - 200

    constrained = parse_compiler_completion(
        _source(), response, max_packet_tokens=constrained_cap
    )

    assert constrained.packet.dropped_fact_ids
    assert constrained.packet.packet_truncated is True
    assert constrained.packet.packet_token_proxy <= constrained_cap
    assert constrained.packet.facts == full.packet.facts[: len(constrained.packet.facts)]
    assert MAX_COMPILER_FACTS == 12
    assert COMPILER_OUTPUT_TOKEN_RESERVE == 2_048
