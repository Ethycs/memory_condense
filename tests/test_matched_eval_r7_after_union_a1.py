from __future__ import annotations

import json
import hashlib
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import canonical_json_bytes, identity_sha256
from tools.matched_eval.r7_after_union_a1 import (
    ANSWER_OUTPUT_TOKEN_RESERVE,
    COMPILER_OUTPUTS_FORMAT,
    DISPOSITIONS_FORMAT,
    MAX_TOTAL_TOKENS,
    R7AfterUnionA1Error,
    build_r7_after_union_a1_payload,
    replay_r7_after_union_a1_payload,
)
from tools.matched_eval.selected_evidence_discourse_links import LINK_FORMAT
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec
from tools.run_r7_after_union_a1 import (
    CONSTRUCTION_NAME,
    REPLAY_NAME,
    run,
)


def _sha(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _question(
    question_id: str,
    dated_question: str,
    summaries: tuple[str, ...],
) -> dict[str, Any]:
    spec = compile_typed_operator_spec(dated_question)
    handles: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    groups: list[str] = []
    for index, summary in enumerate(summaries):
        handle = f"H{index + 1:03d}"
        group = f"G{index + 1:03d}"
        groups.append(group)
        handles.append(
            {
                "group_handle": group,
                "handle_id": handle,
                "origin": "map",
                "provenance_grade": "exact_citation",
            }
        )
        items.append(
            {
                "date": f"2025-01-{index + 1:02d}",
                "entity_key": f"Topic {index + 1}",
                "handle_ids": [handle],
                "included": True,
                "kind": "event",
                "relation": "completed event",
                "status": "completed",
                "summary": summary,
                "supported_slot_ids": [],
            }
        )
    typed = {
        "conflict_policy": "quarantine",
        "format": "synthetic-r7-typed-evidence-v1",
        "frontier": {
            "available_handle_ids": [row["handle_id"] for row in handles],
            "closed": False,
            "mode": "bounded",
            "omitted_handle_ids": [],
            "represented_handle_ids": [row["handle_id"] for row in handles],
            "truncated": False,
        },
        "handles": handles,
        "items": items,
        "operator_spec": spec.projection(),
    }
    question_sha = quote_sha256(dated_question)
    return {
        # These source-only fields prove the adapter does not route or emit them.
        "ordinal": 9001,
        "question_id": question_id,
        "dated_question_sha256": question_sha,
        "terminal_answer_plan": {
            "dated_question_sha256": question_sha,
            "parent_prediction": "SOURCE-ONLY-PARENT-SENTINEL",
            "provider_input": {
                "dated_question": dated_question,
                "protected_parent_fallback": "SOURCE-ONLY-PARENT-SENTINEL",
                "story_coherence": {
                    "group_links": (
                        [
                            {
                                "group_handles": groups[:2],
                                "relation": "same entity across event boundary",
                            }
                        ]
                        if len(groups) >= 2
                        else []
                    ),
                    "incompatible_group_pairs": [],
                    "link_overlays": [],
                },
                "typed_evidence": typed,
            },
            "reference_answer": "SOURCE-ONLY-GOLD-SENTINEL",
        },
    }


def _source(*questions: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "memory-condense-reduced-semantic-global-terminal-assay-v2",
        "gold_loaded": False,
        "new_provider_calls": 0,
        "production_ordinal_routing_enabled": False,
        "question_count": len(questions),
        "questions": list(questions),
        "retained_transformer_token_state_bytes": 0,
        "terminal_answer_plan_count": len(questions),
    }


def _preflight(source: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    source_sha = _sha(source)
    return build_r7_after_union_a1_payload(
        source,
        source_sha,
        source_sha,
        expected_question_count=len(source["questions"]),
        **kwargs,
    )


def _dispositions(
    source: dict[str, Any],
    preflight: dict[str, Any],
    decisions: dict[str, str],
    *,
    classifier_id: str = "sealed-topic-neutral-classifier-v1",
) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    for row in preflight["questions"]:
        leaves = row["semantic_selection"]["leaves"]
        questions.append(
            {
                "classifier_request_sha256s": [
                    request["request_sha256"]
                    for request in row["classifier_requests"]
                ],
                "dispositions": [
                    {
                        "disposition": decisions.get(
                            leaf["handle_id"], "unresolved"
                        ),
                        "handle_id": leaf["handle_id"],
                        "leaf_receipt_sha256": leaf["receipt_sha256"],
                    }
                    for leaf in leaves
                ],
                "question_sha256": row["question_sha256"],
                "selected_union_population_sha256": row[
                    "selected_population_sha256"
                ],
            }
        )
    return {
        "classifier_id": classifier_id,
        "format": DISPOSITIONS_FORMAT,
        "provider_calls_performed_by_core": 0,
        "questions": questions,
        "retained_transformer_token_state_bytes": 0,
        "runtime_firewall": {
            "gold_loaded": False,
            "ordinal_routing_enabled": False,
            "protected_parent_loaded": False,
            "reference_loaded": False,
            "semantic_atom_manifest_loaded": False,
            "source_allowlist_loaded": False,
        },
        "source_artifact_sha256": _sha(source),
    }


def _forbidden_runtime_keys(value: object) -> set[str]:
    forbidden = {
        "ordinal",
        "source_allowlist",
        "semantic_atom_manifest",
        "parent_prediction",
        "protected_parent_fallback",
        "reference_answer",
    }
    observed: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden:
                observed.add(key)
            observed |= _forbidden_runtime_keys(child)
    elif isinstance(value, list):
        for child in value:
            observed |= _forbidden_runtime_keys(child)
    return observed


def test_missing_compilation_is_exact_cover_fail_open_and_replayable() -> None:
    source = _source(
        _question(
            "q-alpha",
            "[Question asked at 2025-01-09] What did I buy and where did I travel?",
            (
                "I bought the cobalt kettle.",
                "I traveled to Kyoto after the conference.",
                "The spare receipt concerned a notebook.",
            ),
        )
    )

    payload = _preflight(source, max_leaves_per_shard=2)

    assert payload["selected_leaf_count"] == 3
    assert payload["construction_status"] == (
        "preflight_external_classification_then_compilation_required"
    )
    assert payload["classifier_request_count"] == 1
    assert payload["compiler_request_count"] == 2
    assert payload["actionable_compiler_request_count"] == 0
    assert payload["missing_classifier_call_count"] == 1
    assert payload["missing_compiler_call_count"] == 0
    assert payload["missing_external_call_count"] == 1
    assert payload["provider_calls_performed_by_core"] == 0
    assert payload["runtime_firewall"] == {
        "benchmark_fields_loaded": False,
        "ordinal_routing_enabled": False,
        "protected_parent_loaded": False,
        "semantic_atom_manifest_loaded": False,
        "source_allowlist_loaded": False,
        "topic_labels_have_exclusion_authority": False,
    }
    assert not _forbidden_runtime_keys(payload)

    row = payload["questions"][0]
    selection = row["semantic_selection"]
    assert row["selected_leaf_count"] == 3
    assert row["disposition_counts"] == {
        "definitely_irrelevant": 0,
        "relevant": 0,
        "uncertain": 3,
    }
    assert selection["semantic_result"]["retained_leaf_cell_ids"] == [
        "H001",
        "H002",
        "H003",
    ]
    assert selection["semantic_result"]["pruned_leaf_cell_ids"] == []
    request_handles = [
        handle
        for request in row["compiler_requests"]
        for handle in request["leaf_handle_ids"]
    ]
    assert request_handles == ["H001", "H002", "H003"]
    classifier_handles = [
        handle
        for request in row["classifier_requests"]
        for handle in request["leaf_handle_ids"]
    ]
    assert classifier_handles == ["H001", "H002", "H003"]
    assert row["actionable_compiler_request_count"] == 0
    for request in row["compiler_requests"]:
        assert (
            request["prompt_token_proxy"]
            + request["answer_output_token_reserve"]
            <= request["hard_total_token_cap"]
        )
        assert request["hard_total_token_cap"] == MAX_TOTAL_TOKENS
        assert request["answer_output_token_reserve"] == ANSWER_OUTPUT_TOKEN_RESERVE
    closure = row["fact_closure"]
    assert closure["selected_population_coverage"]["exact_outcome_coverage"] is True
    assert closure["selected_population_coverage"]["selected_population_resolved"] is False
    assert closure["selected_population_coverage"]["unresolved_leaf_ids"] == [
        "H001",
        "H002",
        "H003",
    ]
    assert closure["operator_obligation_coverage"][
        "required_obligations_closed_within_selected_population"
    ] is False
    source_sha = _sha(source)
    assert replay_r7_after_union_a1_payload(
        payload, source, source_sha, source_sha
    ) == payload


def test_only_sealed_irrelevance_prunes_and_topics_still_union() -> None:
    dated = "[Question asked at 2025-01-09] What did I buy and where did I travel?"
    source = _source(
        _question(
            "q-topics",
            dated,
            (
                "I bought the cobalt kettle.",
                "I traveled to Kyoto after the conference.",
                "The weather report said it was windy.",
            ),
        )
    )
    classifier_preflight = _preflight(source)
    dispositions = _dispositions(
        source,
        classifier_preflight,
        {
            "H001": "relevant",
            "H002": "unresolved",
            "H003": "definitely_irrelevant",
        },
    )

    payload = _preflight(
        source,
        disposition_payload=dispositions,
        disposition_artifact_sha256=_sha(dispositions),
        max_leaves_per_shard=1,
    )

    row = payload["questions"][0]
    selection = row["semantic_selection"]
    assert [leaf["handle_id"] for leaf in selection["leaves"]] == [
        "H001",
        "H002",
        "H003",
    ]
    assert selection["semantic_result"]["retained_leaf_cell_ids"] == [
        "H001",
        "H002",
    ]
    assert selection["semantic_result"]["pruned_leaf_cell_ids"] == ["H003"]
    assert len(selection["cross_boundary_edges"]) == 1
    assert {
        tuple(leaf["topic_labels"])
        for leaf in selection["leaves"][:2]
    }
    assert [
        handle
        for request in row["compiler_requests"]
        for handle in request["leaf_handle_ids"]
    ] == ["H001", "H002"]
    assert row["fact_closure"]["selected_population_coverage"][
        "definitely_irrelevant_leaf_ids"
    ] == ["H003"]


def test_pair_form_content_coherence_link_survives_a1_projection() -> None:
    source_question = _question(
        "q-pair-link",
        "[Question asked at 2025-01-09] What related events did I mention?",
        (
            "I attended the first neighborhood exhibition.",
            "I returned to the same gallery for its winter exhibition.",
        ),
    )
    source_question["terminal_answer_plan"]["provider_input"][
        "story_coherence"
    ]["group_links"] = [
        {
            "basis": "content_entity_coherence",
            "left_group": "G001",
            "right_group": "G002",
        }
    ]

    payload = _preflight(_source(source_question))

    edges = payload["questions"][0]["semantic_selection"][
        "cross_boundary_edges"
    ]
    assert len(edges) == 1
    assert edges[0]["kind"] == "entity"
    assert edges[0]["left_handle_id"] == "H001"
    assert edges[0]["relation"] == "content_entity_coherence"
    assert edges[0]["right_handle_id"] == "H002"
    assert set(edges[0]) == {
        "edge_id",
        "format",
        "kind",
        "left_handle_id",
        "receipt_sha256",
        "relation",
        "right_handle_id",
    }


@pytest.mark.parametrize(
    ("link", "message"),
    (
        (
            {
                "basis": "content_entity_coherence",
                "left_group": "G001",
            },
            "requires exact left/right groups",
        ),
        (
            {
                "basis": "content_entity_coherence",
                "left_group": "G001",
                "right_group": "G999",
            },
            "unknown selected group",
        ),
    ),
)
def test_pair_form_story_link_fails_closed(
    link: dict[str, str], message: str
) -> None:
    source_question = _question(
        "q-invalid-pair-link",
        "[Question asked at 2025-01-09] What related events did I mention?",
        (
            "I attended the first neighborhood exhibition.",
            "I returned to the same gallery for its winter exhibition.",
        ),
    )
    source_question["terminal_answer_plan"]["provider_input"][
        "story_coherence"
    ]["group_links"] = [link]

    with pytest.raises(R7AfterUnionA1Error, match=message):
        _preflight(_source(source_question))


def test_handle_level_typed_link_survives_a1_semantic_selection() -> None:
    source_question = _question(
        "q-typed-link",
        "[Question asked at 2025-01-09] What decision did I revise?",
        (
            "I decided to use option A.",
            "I revised that decision and used option B instead.",
        ),
    )
    story = source_question["terminal_answer_plan"]["provider_input"][
        "story_coherence"
    ]
    story["group_links"] = []
    story["typed_links"] = [
        {
            "format": LINK_FORMAT,
            "link_id": "D0123456789abcdef01234567",
            "members": [
                {
                    "evidence_role": "assistant",
                    "handle_id": "H001",
                    "ordinal": 0,
                    "role": "predecessor",
                },
                {
                    "evidence_role": "user",
                    "handle_id": "H002",
                    "ordinal": 1,
                    "role": "successor",
                },
            ],
            "relation": "revises",
        }
    ]

    payload = _preflight(_source(source_question))

    edges = payload["questions"][0]["semantic_selection"][
        "cross_boundary_edges"
    ]
    assert len(edges) == 1
    assert edges[0]["left_handle_id"] == "H001"
    assert edges[0]["relation"] == "revises"
    assert edges[0]["right_handle_id"] == "H002"


@pytest.mark.parametrize(
    ("members", "message"),
    (
        (
            [
                {
                    "evidence_role": "user",
                    "handle_id": "H001",
                    "ordinal": 0,
                    "role": "predecessor",
                },
                {
                    "evidence_role": "user",
                    "handle_id": "H999",
                    "ordinal": 1,
                    "role": "successor",
                },
            ],
            "unknown selected handle",
        ),
        (
            [
                {
                    "evidence_role": "user",
                    "handle_id": "H001",
                    "ordinal": 0,
                    "role": "predecessor",
                },
                {
                    "evidence_role": "user",
                    "handle_id": "H001",
                    "ordinal": 1,
                    "role": "successor",
                },
            ],
            "requires distinct selected handles",
        ),
    ),
)
def test_handle_level_typed_link_fails_closed(
    members: list[dict[str, object]], message: str
) -> None:
    source_question = _question(
        "q-invalid-typed-link",
        "[Question asked at 2025-01-09] What decision did I revise?",
        ("I decided to use option A.", "I revised it to option B."),
    )
    source_question["terminal_answer_plan"]["provider_input"][
        "story_coherence"
    ]["typed_links"] = [
        {
            "format": LINK_FORMAT,
            "link_id": "D0123456789abcdef01234567",
            "members": members,
            "relation": "revises",
        }
    ]

    with pytest.raises(R7AfterUnionA1Error, match=message):
        _preflight(_source(source_question))


def test_exact_cited_compiler_output_reaches_typed_operator() -> None:
    dated = "[Question asked at 2025-01-09] What city did I visit?"
    source = _source(
        _question("q-compiled", dated, ("I visited Kyoto after the conference.",))
    )
    classifier_preflight = _preflight(source)
    dispositions = _dispositions(
        source, classifier_preflight, {"H001": "relevant"}
    )
    preflight = _preflight(
        source,
        disposition_payload=dispositions,
        disposition_artifact_sha256=_sha(dispositions),
    )
    request = preflight["questions"][0]["compiler_requests"][0]
    response = json.dumps(
        {
            "facts": [
                {
                    "citations": [
                        {
                            "handle_id": "H001",
                            "quote": "I visited Kyoto after the conference.",
                        }
                    ],
                    "date": "2025-01-01",
                    "entity": "Kyoto",
                    "kind": "event",
                    "numeric_value": None,
                    "slot_ids": [],
                    "status": "completed",
                    "text": "The user visited Kyoto.",
                    "unit": None,
                }
            ]
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    outputs = {
        "format": COMPILER_OUTPUTS_FORMAT,
        "provider_calls_performed_by_core": 0,
        "responses": [
            {
                "request_sha256": request["request_sha256"],
                "response_sha256": quote_sha256(response),
                "response_text": response,
            }
        ],
        "retained_transformer_token_state_bytes": 0,
    }

    payload = _preflight(
        source,
        disposition_payload=dispositions,
        disposition_artifact_sha256=_sha(dispositions),
        compiler_output_payload=outputs,
        compiler_output_artifact_sha256=_sha(outputs),
    )

    assert payload["construction_status"] == "complete_materialization"
    assert payload["missing_external_call_count"] == 0
    row = payload["questions"][0]
    closure = row["fact_closure"]
    assert closure["selected_population_coverage"]["exact_outcome_coverage"] is True
    assert closure["selected_population_coverage"]["selected_population_resolved"] is True
    assert closure["selected_population_coverage"]["fact_leaf_ids"] == ["H001"]
    assert len(closure["merged_facts"]) == 1
    assert row["operator_packet"] is not None
    assert row["operator_packet"]["provider_payload_token_proxy"] + 768 <= 8_000
    assert row["operator_execution"] is not None


def test_classifier_payload_is_exact_cover_bounded_and_topic_blind() -> None:
    source = _source(
        _question(
            "q-classifier",
            "[Question asked at 2025-01-09] What did I buy and where did I travel?",
            (
                "I bought the cobalt kettle.",
                "I traveled to Kyoto after the conference.",
                "The spare receipt concerned a notebook.",
            ),
        )
    )
    payload = _preflight(
        source, max_leaves_per_classifier_shard=2
    )
    row = payload["questions"][0]
    assert row["classifier_request_count"] == 2
    assert row["missing_classifier_request_sha256s"] == [
        request["request_sha256"] for request in row["classifier_requests"]
    ]
    assert [
        handle
        for request in row["classifier_requests"]
        for handle in request["leaf_handle_ids"]
    ] == ["H001", "H002", "H003"]
    for request in row["classifier_requests"]:
        assert (
            request["prompt_token_proxy"]
            + request["classifier_output_token_reserve"]
            <= request["hard_total_token_cap"]
        )
        provider_bytes = "\n".join(
            message["content"] for message in request["messages"]
        )
        assert "topic_labels_for_scheduling_only" not in provider_bytes
        assert "boundary_labels" not in provider_bytes
        assert "entity:topic" not in provider_bytes
        assert request["topic_labels_for_scheduling_only"]
        assert request["boundary_labels_for_scheduling_only"]
        assert "relevant|definitely_irrelevant|unresolved" in provider_bytes


def test_required_slots_are_not_falsely_closed_by_slotless_fact() -> None:
    dated = (
        "[Question asked at 2023/05/30 (Tue) 14:24]\n"
        "How many plants did I initially plant for tomatoes and chili peppers?"
    )
    source = _source(
        _question("q-required", dated, ("I planted flowers in the garden.",))
    )
    classifier_preflight = _preflight(source)
    dispositions = _dispositions(
        source, classifier_preflight, {"H001": "relevant"}
    )
    compilation_preflight = _preflight(
        source,
        disposition_payload=dispositions,
        disposition_artifact_sha256=_sha(dispositions),
    )
    request = compilation_preflight["questions"][0]["compiler_requests"][0]
    response = json.dumps(
        {
            "facts": [
                {
                    "citations": [
                        {
                            "handle_id": "H001",
                            "quote": "I planted flowers in the garden.",
                        }
                    ],
                    "date": None,
                    "entity": "flowers",
                    "kind": "event",
                    "numeric_value": None,
                    "slot_ids": [],
                    "status": "completed",
                    "text": "The user planted flowers.",
                    "unit": None,
                }
            ]
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    outputs = {
        "format": COMPILER_OUTPUTS_FORMAT,
        "provider_calls_performed_by_core": 0,
        "responses": [
            {
                "request_sha256": request["request_sha256"],
                "response_sha256": quote_sha256(response),
                "response_text": response,
            }
        ],
        "retained_transformer_token_state_bytes": 0,
    }

    payload = _preflight(
        source,
        disposition_payload=dispositions,
        disposition_artifact_sha256=_sha(dispositions),
        compiler_output_payload=outputs,
        compiler_output_artifact_sha256=_sha(outputs),
    )

    row = payload["questions"][0]
    coverage = row["fact_closure"]["operator_obligation_coverage"]
    assert coverage["required_obligations_closed_within_selected_population"] is False
    assert coverage["missing_required_obligation_ids"]
    assert all(
        fact["obligation_ids"] == []
        for merged in row["fact_closure"]["merged_facts"]
        for fact in merged["facts"]
    )
    assert payload["construction_status"] == "materialized_with_unresolved_closure"
    assert row["compiler_request_results"][0]["status"] == (
        "compiled_with_facts_packet_incomplete"
    )


def test_all_rejected_compiler_response_stays_unresolved() -> None:
    dated = "[Question asked at 2025-01-09] What city did I visit?"
    source = _source(_question("q-rejected", dated, ("I visited Kyoto.",)))
    classifier_preflight = _preflight(source)
    dispositions = _dispositions(
        source, classifier_preflight, {"H001": "relevant"}
    )
    compilation_preflight = _preflight(
        source,
        disposition_payload=dispositions,
        disposition_artifact_sha256=_sha(dispositions),
    )
    request = compilation_preflight["questions"][0]["compiler_requests"][0]
    response = json.dumps(
        {
            "facts": [
                {
                    "citations": [
                        {"handle_id": "H001", "quote": "not in evidence"}
                    ],
                    "date": None,
                    "entity": "Kyoto",
                    "kind": "event",
                    "numeric_value": None,
                    "slot_ids": [],
                    "status": "completed",
                    "text": "The user visited Kyoto.",
                    "unit": None,
                }
            ]
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    outputs = {
        "format": COMPILER_OUTPUTS_FORMAT,
        "provider_calls_performed_by_core": 0,
        "responses": [
            {
                "request_sha256": request["request_sha256"],
                "response_sha256": quote_sha256(response),
                "response_text": response,
            }
        ],
        "retained_transformer_token_state_bytes": 0,
    }
    payload = _preflight(
        source,
        disposition_payload=dispositions,
        disposition_artifact_sha256=_sha(dispositions),
        compiler_output_payload=outputs,
        compiler_output_artifact_sha256=_sha(outputs),
    )
    row = payload["questions"][0]
    assert row["compiler_request_results"][0]["status"] == (
        "compiled_no_valid_facts"
    )
    assert row["fact_closure"]["selected_population_coverage"][
        "unresolved_leaf_ids"
    ] == ["H001"]
    assert payload["construction_status"] == "materialized_with_unresolved_closure"


def test_dispositions_are_bound_to_classifier_requests_and_leaf_receipts() -> None:
    source = _source(
        _question(
            "q-disposition-binding",
            "[Question asked at 2025-01-09] What city did I visit?",
            ("I visited Kyoto.",),
        )
    )
    preflight = _preflight(source)
    dispositions = _dispositions(
        source, preflight, {"H001": "definitely_irrelevant"}
    )
    dispositions["questions"][0]["dispositions"][0][
        "leaf_receipt_sha256"
    ] = "e" * 64
    with pytest.raises(R7AfterUnionA1Error, match="leaf receipt"):
        _preflight(
            source,
            disposition_payload=dispositions,
            disposition_artifact_sha256=_sha(dispositions),
        )

    request_changed = _dispositions(
        source, preflight, {"H001": "definitely_irrelevant"}
    )
    request_changed["questions"][0]["classifier_request_sha256s"][0] = (
        "f" * 64
    )
    with pytest.raises(R7AfterUnionA1Error, match="classifier request population"):
        _preflight(
            source,
            disposition_payload=request_changed,
            disposition_artifact_sha256=_sha(request_changed),
        )


def test_direct_api_rehashes_claimed_source_artifact() -> None:
    source = _source(
        _question(
            "q-source-digest",
            "[Question asked at 2025-01-09] What city did I visit?",
            ("I visited Kyoto.",),
        )
    )
    with pytest.raises(R7AfterUnionA1Error, match="sealed gold-blind"):
        build_r7_after_union_a1_payload(
            source,
            "a" * 64,
            "a" * 64,
            expected_question_count=1,
        )


def test_unknown_compiler_output_request_is_rejected() -> None:
    source = _source(
        _question(
            "q-unknown",
            "[Question asked at 2025-01-09] What city did I visit?",
            ("I visited Kyoto.",),
        )
    )
    classifier_preflight = _preflight(source)
    dispositions = _dispositions(
        source, classifier_preflight, {"H001": "relevant"}
    )
    response = "{\"facts\":[]}"
    outputs = {
        "format": COMPILER_OUTPUTS_FORMAT,
        "provider_calls_performed_by_core": 0,
        "responses": [
            {
                "request_sha256": "d" * 64,
                "response_sha256": quote_sha256(response),
                "response_text": response,
            }
        ],
        "retained_transformer_token_state_bytes": 0,
    }
    with pytest.raises(R7AfterUnionA1Error, match="unknown A1 request"):
        _preflight(
            source,
            disposition_payload=dispositions,
            disposition_artifact_sha256=_sha(dispositions),
            compiler_output_payload=outputs,
            compiler_output_artifact_sha256=_sha(outputs),
        )


def test_cli_seals_byte_identical_construction_and_replay(tmp_path: Path) -> None:
    source = _source(
        _question(
            "q-cli",
            "[Question asked at 2025-01-09] What city did I visit?",
            ("I visited Kyoto.",),
        )
    )
    construction, _ = publish_sealed_json(tmp_path / "source.json", source)
    replay, _ = publish_sealed_json(tmp_path / "source-replay.json", source)
    output = tmp_path / "output"
    result = run(
        Namespace(
            compiler_outputs=None,
            dispositions=None,
            expected_question_count=1,
            max_leaves_per_classifier_shard=48,
            max_leaves_per_shard=8,
            output_root=output,
            source_construction=construction.path,
            source_replay=replay.path,
        )
    )

    sealed = read_sealed_json(output / CONSTRUCTION_NAME)
    sealed_replay = read_sealed_json(output / REPLAY_NAME)
    assert result["replay_byte_identical"] is True
    assert sealed.sha256 == sealed_replay.sha256
    assert sealed.payload == sealed_replay.payload
    assert result["selected_leaf_count"] == 1
    assert result["missing_classifier_call_count"] == 1
    assert result["missing_compiler_call_count"] == 0
    assert result["missing_external_call_count"] == 1


def test_compiler_request_population_changes_with_source_bytes() -> None:
    first = _source(
        _question(
            "q-bind",
            "[Question asked at 2025-01-09] What city did I visit?",
            ("I visited Kyoto.",),
        )
    )
    second = json.loads(json.dumps(first))
    second["questions"][0]["terminal_answer_plan"]["provider_input"][
        "typed_evidence"
    ]["items"][0]["summary"] = "I visited Osaka."
    one = _preflight(first)
    two = _preflight(second)
    assert one["selected_population_sha256"] != two["selected_population_sha256"]
    assert (
        one["questions"][0]["compiler_requests"][0]["request_sha256"]
        != two["questions"][0]["compiler_requests"][0]["request_sha256"]
    )
