from __future__ import annotations

import hashlib
import json

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import EvidenceSpan, quote_sha256
from tools.matched_eval.selected_evidence_discourse_links import (
    SelectedEvidenceLinkInput,
    link_selected_evidence,
)
from tools.matched_eval.typed_memory_final_arm import (
    HARD_PROMPT_TOKEN_CAP,
    OUTPUT_TOKEN_RESERVE,
    STORY_LINK_TOKEN_CAP,
    SYSTEM_PROMPT,
    TypedMemoryFinalArmError,
    VALIDATION_CONTRACT_FORMAT,
    VALIDATOR_POLICY_FORMAT,
    _mechanism_priority,
    fit_typed_final_prompt,
    materialize_typed_final_result_row,
    parse_typed_final_completion,
    story_coherence_projection,
)
from tools.matched_eval.typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    FrontierMode,
    ProvenanceGrade,
    build_typed_evidence_packet,
    parse_typed_items,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


QUESTION = (
    "[Question asked at 2026/08/27 12:00] "
    "How many tomato and chili pepper plants did I initially plant?"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _binding(index: int, *, group: int | None = None) -> EvidenceHandleBinding:
    group_index = index if group is None else group
    return EvidenceHandleBinding(
        f"H{index:03d}",
        EvidenceOrigin.MAP,
        ProvenanceGrade.EXACT_CITATION,
        f"G{group_index:03d}",
        _sha("sealed-map"),
        _sha("parent"),
        _sha(f"evidence-{index}"),
        _sha(f"payload-{index}"),
        _sha(f"citation-{index}"),
        42,
        _sha(f"private-source-{group_index}"),
    )


def _packet(
    raw_items: list[dict[str, object]],
    *,
    output_reserve: int = OUTPUT_TOKEN_RESERVE,
    question: str = QUESTION,
):
    spec = compile_typed_operator_spec(question)
    indices = sorted(
        {
            int(handle[1:])
            for row in raw_items
            for handle in row["handle_ids"]  # type: ignore[index]
        }
    )
    bindings = tuple(_binding(index) for index in indices)
    parsed = parse_typed_items(raw_items, operator_spec=spec, bindings=bindings)
    assert not parsed.rejected_items
    return build_typed_evidence_packet(
        spec,
        bindings,
        parsed,
        sealed_input_artifact_sha256s=(_sha("sealed-map"),),
        frontier_mode=FrontierMode.BOUNDED,
        output_token_reserve=output_reserve,
    )


def _linked_input(handle_id: str, ordinal: int, text: str):
    span = EvidenceSpan(
        chunk_id=f"story-chunk-{ordinal}",
        start_char=0,
        end_char=len(text),
        quote_sha256=quote_sha256(text),
        ordinal=ordinal,
        source_id="story-source",
        turn_id=f"story-turn-{ordinal}",
        role="user",
        created_at=f"2026-08-{ordinal:02d}T12:00:00+00:00",
    )
    return SelectedEvidenceLinkInput(
        handle_id=handle_id,
        span=span,
        quote=text,
        source_binding_receipt_sha256=_sha(f"story-binding-{ordinal}"),
        selected_evidence_receipt_sha256=_sha(f"story-selected-{ordinal}"),
    )


def test_story_plane_filters_whole_typed_link_when_endpoint_is_not_fitted() -> None:
    compiled = link_selected_evidence(
        (
            _linked_input("H001", 1, "We decided to use option A."),
            _linked_input(
                "H002",
                2,
                "We revised that decision; instead use option B.",
            ),
        )
    )
    full_packet = _packet(
        [
            {"handle_ids": ["H001"], "kind": "claim", "summary": "Option A."},
            {"handle_ids": ["H002"], "kind": "claim", "summary": "Option B."},
        ]
    )
    fitted_packet = _packet(
        [{"handle_ids": ["H001"], "kind": "claim", "summary": "Option A."}]
    )

    full_story = story_coherence_projection(
        full_packet,
        selected_evidence_discourse_links=compiled,
    )
    fitted_story = story_coherence_projection(
        fitted_packet,
        selected_evidence_discourse_links=compiled,
    )

    assert {row["relation"] for row in full_story["typed_links"]} >= {
        "sequence",
        "revises",
    }
    assert "typed_links" not in fitted_story


def test_final_fit_retains_active_reconstruction_above_passive_full_store() -> None:
    assert _mechanism_priority("active_reconstruction_v1") == 60
    assert _mechanism_priority("active_reconstruction_v1") > _mechanism_priority(
        "full_store_slot_closure_v1"
    )


def test_final_fit_preserves_high_cue_active_connector_before_shallow_slots() -> None:
    high_cue_summary = (
        "Tomato plant memory: "
        + " ".join(f"connector{index}" for index in range(2_000))
    )
    shallow_multi_slot_summary = (
        "Tomato and chili pepper plant memory: "
        + " ".join(f"surface{index}" for index in range(600))
    )
    packet = _packet(
        [
            {
                "handle_ids": ["H600001"],
                "kind": "operand",
                "summary": high_cue_summary,
                "numeric_role": "operand",
                "numeric_value": 6,
            },
            {
                "handle_ids": ["H600002"],
                "kind": "operand",
                "summary": shallow_multi_slot_summary,
                "numeric_role": "operand",
                "numeric_value": 4,
            },
        ],
        output_reserve=1,
    )
    by_handle = {
        item.handle_ids[0]: item for item in packet.items
    }
    assert len(by_handle["H600001"].supported_slot_ids) < len(
        by_handle["H600002"].supported_slot_ids
    )
    high = (1,) * 24
    low = (0,) * 24
    fitted = fit_typed_final_prompt(
        dated_question=QUESTION,
        parent_prediction="10",
        packet=packet,
        mechanism_by_handle={
            "H600001": "active_reconstruction_v1",
            "H600002": "active_reconstruction_v1",
        },
        local_retention_priority_by_handle={
            "H600001": high,
            "H600002": low,
        },
    )
    retained = {row.handle_id for row in fitted.packet.handles}
    assert "H600001" in retained
    assert "H600002" not in retained
    assert fitted.dropped_item_receipt_sha256s


def _validation_contract(
    handles: tuple[str, ...],
    *,
    anchors: dict[str, list[str]] | None = None,
    answer_shape: str = "direct",
    actions_by_handle: dict[str, list[str]] | None = None,
    cardinality: int | None = None,
    comparison_mode: str = "none",
    date_by_handle: dict[str, str] | None = None,
    numeric_by_handle: dict[str, float] | None = None,
    numeric_qualifier_by_handle: dict[str, str] | None = None,
    operation: str = "single_supported_fact",
    question_actions: list[str] | None = None,
    question_terms: list[str] | None = None,
    relation_terms_by_handle: dict[str, list[str]] | None = None,
    required_slots: list[dict[str, object]] | None = None,
    scalar_advisory: dict[str, object] | None = None,
    semantic_unit_by_handle: dict[str, str] | None = None,
    summary_terms_by_handle: dict[str, list[str]] | None = None,
    temporal_mode: str = "none",
) -> dict[str, object]:
    slots = required_slots or []
    return {
        "answer_shape": answer_shape,
        "by_handle": {
            handle: {
                "answer_anchor_terms": (anchors or {}).get(handle, []),
                "numeric_value_rows": (
                    [
                        {
                            "item_receipt_sha256": _sha(f"item-{handle}"),
                            "numeric_value": (numeric_by_handle or {})[handle],
                            "numeric_qualifier": (
                                numeric_qualifier_by_handle or {}
                            ).get(handle, "exact"),
                            "supported_slot_ids": [
                                row["slot_id"]
                                for row in slots
                                if handle in row.get("handle_ids", [])
                            ],
                            "unit": None,
                        }
                    ]
                    if handle in (numeric_by_handle or {})
                    else []
                ),
                "semantic_rows": [
                    {
                        "action_concepts": (actions_by_handle or {}).get(
                            handle,
                            question_actions or [],
                        ),
                        "completed_action_concepts": (actions_by_handle or {}).get(
                            handle,
                            question_actions or [],
                        ),
                        "date": (date_by_handle or {}).get(handle),
                        "entity_terms": [],
                        "explicit_member_count": None,
                        "group_terms": [],
                        "item_receipt_sha256": _sha(f"item-{handle}"),
                        "kind": "direct",
                        "numeric_role": "operand",
                        "numeric_qualifier": (
                            numeric_qualifier_by_handle or {}
                        ).get(handle, "exact"),
                        "numeric_value": (numeric_by_handle or {}).get(handle),
                        "participant_count": None,
                        "relation_terms": (relation_terms_by_handle or {}).get(
                            handle, []
                        ),
                        "semantic_unit_sha256": (semantic_unit_by_handle or {}).get(
                            handle, _sha(f"semantic-{handle}")
                        ),
                        "status": "completed",
                        "summary_terms": (summary_terms_by_handle or {}).get(
                            handle,
                            (anchors or {}).get(handle, ["plant"]),
                        ),
                        "supported_slot_ids": [
                            row["slot_id"]
                            for row in slots
                            if handle in row.get("handle_ids", [])
                        ],
                        "unit": None,
                    }
                ],
                "status_values": ["completed"],
                "supported_slot_ids": [
                    row["slot_id"]
                    for row in slots
                    if handle in row.get("handle_ids", [])
                ],
                "usable_item_receipt_sha256s": [_sha(f"item-{handle}")],
            }
            for handle in handles
        },
        "cardinality": cardinality,
        "comparison_mode": comparison_mode,
        "deterministic_execution_advisory": None,
        "format": VALIDATION_CONTRACT_FORMAT,
        "include_proposed": False,
        "operation": operation,
        "operator_spec_receipt_sha256": _sha("spec"),
        "packet_receipt_sha256": _sha("packet"),
        "question_action_concepts": question_actions or [],
        "question_terms": question_terms or ["question"],
        "required_slot_ids": [row["slot_id"] for row in slots],
        "required_slots": [
            {key: value for key, value in row.items() if key != "handle_ids"}
            for row in slots
        ],
        "requires_all_slots": True,
        "scalar_validation_advisory": scalar_advisory,
        "temporal_mode": temporal_mode,
    }


def _parse_candidate(
    prediction: str,
    handles: tuple[str, ...],
    contract: dict[str, object],
    *,
    allowed_handles: tuple[str, ...] | None = None,
    parent_prediction: str = "protected parent",
    preservation_requirements: dict[str, object] | None = None,
):
    allowed = handles if allowed_handles is None else allowed_handles
    return parse_typed_final_completion(
        json.dumps(
            {
                "decision": "replace",
                "prediction": prediction,
                "used_handle_ids": list(handles),
            }
        ),
        parent_prediction=parent_prediction,
        allowed_handle_ids=allowed,
        handle_group_by_id={
            handle: f"G{index + 1:03d}" for index, handle in enumerate(allowed)
        },
        story_coherence={"incompatible_group_pairs": []},
        preservation_requirements=preservation_requirements
        or {"by_handle": {}, "question_required_terms": []},
        validation_contract=contract,
    )


def test_bounded_frontier_is_not_an_automatic_keep_parent_instruction() -> None:
    assert "BOUNDED is a provenance qualifier, not automatic insufficiency" in SYSTEM_PROMPT
    assert "all explicit required slots" in SYSTEM_PROMPT


def test_complete_chat_budget_trims_weakest_item_and_protects_parent() -> None:
    marker = "PROTECTED-PARENT-ANSWER"
    raw = [
        {
            "handle_ids": ["H001"],
            "kind": "operand",
            "summary": (
                "Tomato plant memory: "
                + " ".join(f"detail{index}" for index in range(2_000))
            ).strip(),
            "numeric_value": 6,
            "numeric_role": "operand",
        },
        {
            "handle_ids": ["H002"],
            "kind": "operand",
            "summary": (
                "Chili plant weaker tail: "
                + " ".join(f"extra{index}" for index in range(600))
            ).strip(),
            "numeric_value": 4,
            "numeric_role": "operand",
        },
    ]
    # Construction uses a one-token reserve, as the fair common merge does.
    # The final fitter must account the complete wrapper and the real 768-token
    # completion allowance, then drop a whole weakest item deterministically.
    packet = _packet(raw, output_reserve=1)
    fitted = fit_typed_final_prompt(
        dated_question=QUESTION,
        parent_prediction=marker,
        packet=packet,
        mechanism_by_handle={row.handle_id: "parent_map" for row in packet.handles},
    )

    assert fitted.prompt_token_proxy == count_chat_prompt_token_proxy(fitted.messages)
    assert fitted.prompt_token_proxy + OUTPUT_TOKEN_RESERVE <= HARD_PROMPT_TOKEN_CAP
    assert fitted.dropped_item_receipt_sha256s
    assert marker in fitted.messages[1]["content"]
    assert len(fitted.packet.items) < len(packet.items)
    assert fitted.dropped_binding_receipt_sha256s


def test_complete_chat_budget_drops_surplus_before_exact_lane_minimum() -> None:
    packet = _packet(
        [
            {
                "handle_ids": ["H001"],
                "summary": " ".join(
                    f"surplus{index}" for index in range(2_000)
                ),
            },
            {
                "handle_ids": ["H002"],
                "summary": " ".join(
                    f"minimum{index}" for index in range(600)
                ),
            },
        ],
        output_reserve=1,
    )
    protected = packet.items[1]
    fitted = fit_typed_final_prompt(
        dated_question=QUESTION,
        parent_prediction="protected parent",
        packet=packet,
        mechanism_by_handle={
            row.handle_id: "parent_map" for row in packet.handles
        },
        protected_item_receipt_sha256s=(protected.receipt_sha256,),
        protection_source_receipt_sha256=_sha("fair merge source"),
    )

    final_receipts = {item.receipt_sha256 for item in fitted.packet.items}
    assert protected.receipt_sha256 in final_receipts
    assert packet.items[0].receipt_sha256 in fitted.dropped_item_receipt_sha256s
    assert fitted.protected_binding_receipt_sha256s == (
        packet.local_bindings[1].receipt_sha256,
    )
    assert fitted.projection()["protection_source_receipt_sha256"] == _sha(
        "fair merge source"
    )


def test_complete_chat_budget_fails_closed_when_exact_minima_cannot_fit() -> None:
    packet = _packet(
        [
            {
                "handle_ids": ["H001"],
                "summary": " ".join(
                    f"first{index}" for index in range(2_000)
                ),
            },
            {
                "handle_ids": ["H002"],
                "summary": " ".join(
                    f"second{index}" for index in range(600)
                ),
            },
        ],
        output_reserve=1,
    )
    with pytest.raises(
        TypedMemoryFinalArmError,
        match="protected minima cannot fit",
    ):
        fit_typed_final_prompt(
            dated_question=QUESTION,
            parent_prediction="protected parent",
            packet=packet,
            mechanism_by_handle={
                row.handle_id: "parent_map" for row in packet.handles
            },
            protected_item_receipt_sha256s=tuple(
                item.receipt_sha256 for item in packet.items
            ),
            protection_source_receipt_sha256=_sha("fair merge source"),
        )


def test_exact_chunk_json_round_trip_preserves_quotes_newlines_and_unicode() -> None:
    exact_chunk = (
        'I said "use the cobalt paper."\n'
        "Mina answered: bring the café receipt and the lantern 🏮."
    )
    packet = _packet(
        [{"handle_ids": ["H001"], "summary": exact_chunk}]
    )
    fitted = fit_typed_final_prompt(
        dated_question=QUESTION,
        parent_prediction="protected parent",
        packet=packet,
        mechanism_by_handle={"H001": "parent_map"},
    )
    provider_input = json.loads(fitted.messages[1]["content"])
    assert provider_input["typed_evidence"]["items"][0]["summary"] == (
        exact_chunk
    )


def test_exact_local_comembership_becomes_opaque_cav_link_with_local_binding() -> None:
    # The ranges model two distinct mechanisms.  Their exact local provenance
    # receipt is equal, but neither that receipt nor a source locator is shown
    # in the provider overlay.
    spec = compile_typed_operator_spec(QUESTION)
    bindings = (
        _binding(1, group=1),
        EvidenceHandleBinding(
            "H100001",
            EvidenceOrigin.MAP,
            ProvenanceGrade.EXACT_CITATION,
            "G100001",
            _sha("sealed-tail"),
            _sha("tail-parent"),
            _sha("tail-evidence"),
            _sha("tail-payload"),
            _sha("tail-citation"),
            30,
            _sha("private-source-shared"),
        ),
    )
    parsed = parse_typed_items(
        [
            {"handle_ids": ["H001"], "summary": "6 tomato plants"},
            {"handle_ids": ["H100001"], "summary": "4 chili plants"},
        ],
        operator_spec=spec,
        bindings=bindings,
    )
    packet = build_typed_evidence_packet(
        spec,
        bindings,
        parsed,
        sealed_input_artifact_sha256s=(_sha("sealed-map"), _sha("sealed-tail")),
        frontier_mode=FrontierMode.BOUNDED,
    )
    local_receipt = _sha("same-exact-local-source-receipt")
    fitted = fit_typed_final_prompt(
        dated_question=QUESTION,
        parent_prediction="10",
        packet=packet,
        mechanism_by_handle={"H001": "parent_map", "H100001": "tail"},
        local_story_keys_by_group={
            "G001": (local_receipt,),
            "G100001": (local_receipt,),
        },
    )
    overlay = fitted.story_coherence["link_overlays"]
    assert overlay == [
        {
            "group_handles": ["G001", "G100001"],
            "link_id": "L001",
            "relation": "exact_local_candidate_comembership",
        }
    ]
    assert fitted.story_link_local_bindings[0][
        "local_story_key_receipt_sha256"
    ] == local_receipt
    assert local_receipt not in fitted.messages[1]["content"]


def test_exact_story_overlays_prioritize_broader_selected_evidence_components() -> None:
    packet = _packet(
        [
            {"handle_ids": [f"H{index:03d}"], "summary": f"Exact chunk {index}"}
            for index in range(1, 6)
        ]
    )
    narrow_key = "0" * 64
    broad_key = "f" * 64
    story = story_coherence_projection(
        packet,
        local_story_keys_by_group={
            "G001": (narrow_key,),
            "G002": (narrow_key,),
            "G003": (broad_key,),
            "G004": (broad_key,),
            "G005": (broad_key,),
        },
    )

    # Hash sorting would put ``narrow_key`` first.  The compact link plane must
    # instead spend its first position on the broader evidence-derived story.
    assert story["link_overlays"][0]["group_handles"] == [
        "G003",
        "G004",
        "G005",
    ]


def test_distinct_set_members_sharing_a_generic_slot_are_not_conflicts() -> None:
    packet = _packet(
        [
            {
                "handle_ids": ["H001"],
                "kind": "member",
                "summary": "Joined an online photography community",
                "entity_key": "photography",
                "relation": "joined community",
                "numeric_value": 1,
            },
            {
                "handle_ids": ["H002"],
                "kind": "member",
                "summary": "Joined an online hiking community",
                "entity_key": "hiking",
                "relation": "joined community",
                "numeric_value": 2,
            },
        ]
    )
    story = story_coherence_projection(packet)
    assert story["incompatible_group_pairs"] == []


def test_linked_same_assertion_with_incompatible_values_remains_a_conflict() -> None:
    packet = _packet(
        [
            {
                "handle_ids": ["H001"],
                "kind": "operand",
                "summary": "Tomato plot initially had 6 plants",
                "entity_key": "tomato plot",
                "relation": "initial plant count",
                "numeric_value": 6,
                "numeric_role": "baseline",
                "status": "completed",
                "date": "2023-03-01",
            },
            {
                "handle_ids": ["H002"],
                "kind": "operand",
                "summary": "Tomato plot initially had 8 plants",
                "entity_key": "tomato plot",
                "relation": "initial plant count",
                "numeric_value": 8,
                "numeric_role": "baseline",
                "status": "completed",
                "date": "2023-03-01",
            },
        ]
    )
    shared = _sha("same-local-story")
    story = story_coherence_projection(
        packet,
        local_story_keys_by_group={"G001": (shared,), "G002": (shared,)},
    )
    assert story["link_overlays"]
    assert len(story["incompatible_group_pairs"]) == 1


def test_same_entity_at_different_event_times_is_a_timeline_not_conflict() -> None:
    packet = _packet(
        [
            {
                "handle_ids": ["H001"],
                "kind": "event",
                "summary": "Tomato plot had 6 plants on 2023-03-01",
                "entity_key": "tomato plot",
                "relation": "plant count",
                "numeric_value": 6,
                "date": "2023-03-01",
            },
            {
                "handle_ids": ["H002"],
                "kind": "event",
                "summary": "Tomato plot had 8 plants on 2023-03-08",
                "entity_key": "tomato plot",
                "relation": "plant count",
                "numeric_value": 8,
                "date": "2023-03-08",
            },
        ]
    )
    assert story_coherence_projection(packet)["incompatible_group_pairs"] == []


def test_raw_locator_keys_and_values_cannot_escape_provider_projection() -> None:
    packet = _packet(
        [
            {
                "handle_ids": ["H001"],
                "kind": "operand",
                "summary": "I initially planted 6 tomato plants.",
                "numeric_value": 6,
                "numeric_role": "operand",
            }
        ]
    )
    with pytest.raises(TypedMemoryFinalArmError, match="raw locator value"):
        fit_typed_final_prompt(
            dated_question=QUESTION,
            parent_prediction="10",
            packet=packet,
            mechanism_by_handle={"H001": "parent_map"},
            forbidden_provider_literals=("initially planted",),
        )


def test_contribution_ranges_must_be_disjoint() -> None:
    packet = _packet(
        [
            {
                "handle_ids": ["H001"],
                "summary": "6 tomato plants",
                "numeric_value": 6,
                "numeric_role": "operand",
            },
            {
                "handle_ids": ["H002"],
                "summary": "4 chili pepper plants",
                "numeric_value": 4,
                "numeric_role": "operand",
            },
        ]
    )
    with pytest.raises(TypedMemoryFinalArmError, match="disjoint H/G ranges"):
        fit_typed_final_prompt(
            dated_question=QUESTION,
            parent_prediction="10",
            packet=packet,
            mechanism_by_handle={"H001": "parent_map", "H002": "tail"},
        )


def test_strict_completion_validation_and_invalid_parent_fallback() -> None:
    packet = _packet(
        [
            {
                "handle_ids": ["H001"],
                "kind": "operand",
                "summary": "I initially planted 6 tomato plants.",
                "numeric_value": 6,
                "numeric_role": "operand",
            },
            {
                "handle_ids": ["H002"],
                "kind": "operand",
                "summary": "I initially planted 4 chili pepper plants.",
                "numeric_value": 4,
                "numeric_role": "operand",
            },
        ]
    )
    fitted = fit_typed_final_prompt(
        dated_question=QUESTION,
        parent_prediction="9",
        packet=packet,
        mechanism_by_handle={"H001": "parent_map", "H002": "parent_map"},
    )
    valid = parse_typed_final_completion(
        json.dumps(
            {
                "decision": "replace",
                "prediction": "10",
                "used_handle_ids": ["H001", "H002"],
            }
        ),
        parent_prediction="9",
        allowed_handle_ids=fitted.allowed_handle_ids,
        handle_group_by_id=fitted.handle_group_by_id,
        story_coherence=fitted.story_coherence,
        preservation_requirements=fitted.preservation_requirements,
        validation_contract=fitted.validation_contract,
    )
    assert valid.valid is True
    assert valid.decision == "replace"

    plan = {
        "allowed_handle_ids": list(fitted.allowed_handle_ids),
        "dated_question_sha256": _sha(QUESTION),
        "handle_group_by_id": dict(fitted.handle_group_by_id),
        "ordinal": 0,
        "parent_prediction": "9",
        "preservation_requirements": dict(fitted.preservation_requirements),
        "validation_contract": dict(fitted.validation_contract),
        "prompt_row_receipt_sha256": fitted.receipt_sha256,
        "question_id": "opaque-local-q0",
        "question_sha256": _sha(QUESTION),
        "route_id": packet.operator_spec.style.value,
        "story_coherence": dict(fitted.story_coherence),
    }
    row = materialize_typed_final_result_row(
        plan,
        '{"decision":"replace","prediction":"oops","used_handle_ids":["H999"]}',
        completion_receipt_sha256=_sha("completion"),
        call_key_sha256=_sha("call"),
        request_journal_sha256=_sha("request"),
        response_journal_sha256=_sha("response"),
    )
    assert row["prediction"] == "9"
    assert row["changed_from_parent"] is False
    assert row["prediction_source"] == "typed_final_invalid_keep_parent_v1"
    assert row["parse_error_code"] == "unknown_handle"


def test_incompatible_story_groups_force_exact_parent() -> None:
    parsed = parse_typed_final_completion(
        json.dumps(
            {
                "decision": "replace",
                "prediction": "9 plants",
                "used_handle_ids": ["H001", "H100001"],
            }
        ),
        parent_prediction="10 plants",
        allowed_handle_ids=("H001", "H100001"),
        handle_group_by_id={"H001": "G001", "H100001": "G100001"},
        story_coherence={
            "incompatible_group_pairs": [
                {
                    "left_group": "G001",
                    "right_group": "G100001",
                    "reason": "overlapping_slot_inconsistent_value",
                }
            ]
        },
        preservation_requirements={"by_handle": {}, "question_required_terms": []},
        validation_contract=_validation_contract(("H001", "H100001")),
    )
    assert parsed.valid is False
    assert parsed.error_code == "incompatible_story_groups"


def test_keep_parent_must_be_byte_exact_and_empty_citation_list() -> None:
    parsed = parse_typed_final_completion(
        '{"decision":"keep_parent","prediction":"parent ","used_handle_ids":[]}',
        parent_prediction="parent",
        allowed_handle_ids=(),
        handle_group_by_id={},
        story_coherence={"incompatible_group_pairs": []},
        preservation_requirements={"by_handle": {}, "question_required_terms": []},
        validation_contract=_validation_contract(()),
    )
    assert parsed.valid is False
    assert parsed.error_code == "keep_parent_contract"


def test_identical_replace_normalizes_to_valid_keep_after_handle_security() -> None:
    contract = _validation_contract(("H001",))
    parsed = parse_typed_final_completion(
        '{"decision":"replace","prediction":"parent","used_handle_ids":["H001"]}',
        parent_prediction="parent",
        allowed_handle_ids=("H001",),
        handle_group_by_id={"H001": "G001"},
        story_coherence={"incompatible_group_pairs": []},
        preservation_requirements={"by_handle": {}, "question_required_terms": []},
        validation_contract=contract,
    )
    assert parsed.valid is True
    assert parsed.decision == "keep_parent"
    assert parsed.prediction == "parent"
    assert parsed.used_handle_ids == ()
    assert parsed.validation_basis == "normalized_identical_replace"

    unknown = parse_typed_final_completion(
        '{"decision":"replace","prediction":"parent","used_handle_ids":["H999"]}',
        parent_prediction="parent",
        allowed_handle_ids=("H001",),
        handle_group_by_id={"H001": "G001"},
        story_coherence={"incompatible_group_pairs": []},
        preservation_requirements={"by_handle": {}, "question_required_terms": []},
        validation_contract=contract,
    )
    assert unknown.valid is False
    assert unknown.error_code == "unknown_handle"

    repeated = parse_typed_final_completion(
        '{"decision":"replace","prediction":"parent","used_handle_ids":["H001","H001"]}',
        parent_prediction="parent",
        allowed_handle_ids=("H001",),
        handle_group_by_id={"H001": "G001"},
        story_coherence={"incompatible_group_pairs": []},
        preservation_requirements={"by_handle": {}, "question_required_terms": []},
        validation_contract=contract,
    )
    assert repeated.valid is False
    assert repeated.error_code == "value_schema"


def test_identical_normalization_is_sealed_in_materialized_output() -> None:
    contract = _validation_contract(("H001",))
    plan = {
        "allowed_handle_ids": ["H001"],
        "dated_question_sha256": _sha("dated"),
        "handle_group_by_id": {"H001": "G001"},
        "ordinal": 0,
        "parent_prediction": "parent",
        "preservation_requirements": {
            "by_handle": {},
            "question_required_terms": [],
        },
        "prompt_row_receipt_sha256": _sha("prompt"),
        "question_id": "opaque-q0",
        "question_sha256": _sha("question"),
        "route_id": "direct",
        "story_coherence": {"incompatible_group_pairs": []},
        "validation_contract": contract,
    }
    row = materialize_typed_final_result_row(
        plan,
        '{"decision":"replace","prediction":"parent","used_handle_ids":["H001"]}',
        completion_receipt_sha256=_sha("completion"),
        call_key_sha256=_sha("call"),
        request_journal_sha256=_sha("request"),
        response_journal_sha256=_sha("response"),
    )
    assert row["decision"] == "keep_parent"
    assert row["prediction_source"] == "typed_final_validated_keep_parent_v1"
    assert row["validation_basis"] == "normalized_identical_replace"
    assert row["validator_policy_format"] == VALIDATOR_POLICY_FORMAT


def test_candidate_preservation_is_scoped_to_used_handles_not_distractors() -> None:
    requirements = {
        "by_handle": {
            "H001": {
                "specificity_terms": ["lager"],
                "personalization_terms": [],
            },
            "H002": {
                "specificity_terms": ["distractor"],
                "personalization_terms": ["unrelated"],
            },
        },
        "question_required_terms": [],
    }
    valid = parse_typed_final_completion(
        '{"decision":"replace","prediction":"A pale lager","used_handle_ids":["H001"]}',
        parent_prediction="beer",
        allowed_handle_ids=("H001", "H002"),
        handle_group_by_id={"H001": "G001", "H002": "G002"},
        story_coherence={"incompatible_group_pairs": []},
        preservation_requirements=requirements,
        validation_contract=_validation_contract(
            ("H001", "H002"), anchors={"H001": ["lager"]}
        ),
    )
    assert valid.valid is True

    invalid = parse_typed_final_completion(
        '{"decision":"replace","prediction":"A pale ale","used_handle_ids":["H001"]}',
        parent_prediction="beer",
        allowed_handle_ids=("H001", "H002"),
        handle_group_by_id={"H001": "G001", "H002": "G002"},
        story_coherence={"incompatible_group_pairs": []},
        preservation_requirements=requirements,
        validation_contract=_validation_contract(
            ("H001", "H002"), anchors={"H001": ["lager"]}
        ),
    )
    assert invalid.valid is False
    assert invalid.error_code == "candidate_preservation"

    keep = parse_typed_final_completion(
        '{"decision":"keep_parent","prediction":"beer","used_handle_ids":[]}',
        parent_prediction="beer",
        allowed_handle_ids=("H001", "H002"),
        handle_group_by_id={"H001": "G001", "H002": "G002"},
        story_coherence={"incompatible_group_pairs": []},
        preservation_requirements=requirements,
        validation_contract=_validation_contract(("H001", "H002")),
    )
    assert keep.valid is True


def test_model_attested_numeric_answer_must_be_supported_by_used_scalars() -> None:
    contract = _validation_contract(
        ("H001",),
        answer_shape="number",
        actions_by_handle={"H001": ["acquire"]},
        numeric_by_handle={"H001": 50.0},
        operation="count_or_aggregate",
        question_actions=["acquire"],
        question_terms=["total", "weight", "feed", "acquire"],
        summary_terms_by_handle={"H001": ["feed", "50", "pound", "acquire"]},
    )

    unsupported = _parse_candidate("70 pounds", ("H001",), contract)
    assert unsupported.valid is False
    assert unsupported.error_code == "typed_numeric_entailment"

    supported = _parse_candidate("50 pounds", ("H001",), contract)
    assert supported.valid is True


def test_scalar_advisory_is_an_optional_fast_path_not_an_exclusive_veto() -> None:
    advisory = {
        "basis": "bounded_positive_scalar_check_no_frontier_upgrade",
        "prediction": "2",
        "used_handle_ids": ["H001"],
    }
    contract = _validation_contract(
        ("H001", "H002"),
        answer_shape="number",
        numeric_by_handle={"H001": 2.0, "H002": 3.0},
        scalar_advisory=advisory,
    )
    exact = _parse_candidate(
        "2",
        ("H001",),
        contract,
        allowed_handles=("H001", "H002"),
    )
    assert exact.valid is True
    assert exact.validation_basis == "bounded_positive_scalar_agreement"

    alternative = _parse_candidate(
        "3",
        ("H002",),
        contract,
        allowed_handles=("H001", "H002"),
    )
    assert alternative.valid is True
    assert alternative.validation_basis == "model_attested"

    unsupported = _parse_candidate(
        "7",
        ("H002",),
        contract,
        allowed_handles=("H001", "H002"),
    )
    assert unsupported.valid is False
    assert unsupported.error_code == "typed_numeric_entailment"

    malformed = _validation_contract(
        ("H001",),
        answer_shape="number",
        numeric_by_handle={"H001": 2.0},
        scalar_advisory={
            "basis": "bounded_positive_scalar_check_no_frontier_upgrade",
            "prediction": "2",
            "used_handle_ids": [{}],
        },
    )
    with pytest.raises(TypedMemoryFinalArmError, match="scalar completion advisory"):
        _parse_candidate("2", ("H001",), malformed)


def test_direct_replacement_cannot_drop_a_protected_parent_url() -> None:
    contract = _validation_contract(
        ("H001",),
        summary_terms_by_handle={"H001": ["manual", "guide", "example"]},
    )
    lossy = _parse_candidate(
        "The setup manual",
        ("H001",),
        contract,
        parent_prediction="The setup manual: https://example.com/guide",
    )
    assert lossy.valid is False
    assert lossy.error_code == "parent_anchor_loss"

    preserved = _parse_candidate(
        "Setup guide: https://example.com/guide",
        ("H001",),
        contract,
        parent_prediction="The setup manual: https://example.com/guide",
    )
    assert preserved.valid is True

    no_url_parent = _parse_candidate(
        "The setup manual",
        ("H001",),
        contract,
        parent_prediction="A manual",
    )
    assert no_url_parent.valid is True


def test_direct_resource_replacement_preserves_selected_evidence_title_and_url() -> None:
    question = (
        "[Question asked at 2026/08/28 10:00] "
        "Can you remind me of the Alpine Clinic desk video you recommended?"
    )
    packet = _packet(
        [
            {
                "handle_ids": ["H001"],
                "kind": "direct",
                "summary": (
                    '1. "Desk Stretch Basics" by the Alpine Clinic: '
                    "<https://resources.example/stretch>\n"
                    '2. "Keyboard Setup Notes" by Harbor Health: '
                    "<https://resources.example/keyboard>"
                ),
                "relation": "authored_by_assistant",
            }
        ],
        question=question,
    )
    fitted = fit_typed_final_prompt(
        dated_question=question,
        parent_prediction="The Alpine Clinic desk video",
        packet=packet,
        mechanism_by_handle={"H001": "terminal_exact_resource"},
    )
    requirements = fitted.preservation_requirements["by_handle"]["H001"]
    assert requirements["exact_identifier_anchors"] == [
        "https://resources.example/stretch"
    ]
    assert requirements["exact_title_anchors"] == ["Desk Stretch Basics"]

    missing_url = _parse_candidate(
        "Desk Stretch Basics by the Alpine Clinic",
        ("H001",),
        dict(fitted.validation_contract),
        parent_prediction="The Alpine Clinic desk video",
        preservation_requirements=dict(fitted.preservation_requirements),
    )
    assert missing_url.valid is False
    assert missing_url.error_code == "evidence_identifier_anchor_loss"

    missing_title = _parse_candidate(
        "Alpine Clinic video: https://resources.example/stretch",
        ("H001",),
        dict(fitted.validation_contract),
        parent_prediction="The Alpine Clinic desk video",
        preservation_requirements=dict(fitted.preservation_requirements),
    )
    assert missing_title.valid is False
    assert missing_title.error_code == "evidence_title_anchor_loss"

    complete = _parse_candidate(
        "Desk Stretch Basics: https://resources.example/stretch",
        ("H001",),
        dict(fitted.validation_contract),
        parent_prediction="The Alpine Clinic desk video",
        preservation_requirements=dict(fitted.preservation_requirements),
    )
    assert complete.valid is True


def test_personalized_synthesis_requires_a_user_grounded_citation() -> None:
    generic = _validation_contract(
        ("H001",),
        answer_shape="synthesis",
        operation="preference_or_causal_synthesis",
        summary_terms_by_handle={"H001": ["denver", "cocktail", "bar"]},
    )
    rejected = _parse_candidate("A Denver cocktail bar", ("H001",), generic)
    assert rejected.valid is False
    assert rejected.error_code == "personalization_citation_missing"

    grounded = _validation_contract(
        ("H001",),
        answer_shape="synthesis",
        operation="preference_or_causal_synthesis",
        relation_terms_by_handle={"H001": ["authored", "user"]},
        summary_terms_by_handle={
            "H001": ["user", "prefer", "denver", "cocktail", "bar"]
        },
    )
    accepted = _parse_candidate("A Denver cocktail bar", ("H001",), grounded)
    assert accepted.valid is True

    anchored = _parse_candidate(
        "A favorite Denver cocktail bar",
        ("H001",),
        generic,
        preservation_requirements={
            "by_handle": {
                "H001": {
                    "personalization_terms": ["favorite"],
                    "specificity_terms": [],
                }
            },
            "question_required_terms": [],
        },
    )
    assert anchored.valid is True

    direct = _validation_contract(
        ("H001",),
        summary_terms_by_handle={"H001": ["denver", "cocktail", "bar"]},
    )
    assert _parse_candidate("A Denver cocktail bar", ("H001",), direct).valid


def test_model_attested_count_requires_all_in_scope_semantic_units() -> None:
    handles = ("H001", "H002", "H003")
    contract = _validation_contract(
        handles,
        answer_shape="number",
        actions_by_handle={handle: ["visit"] for handle in handles},
        operation="count_or_aggregate",
        question_actions=["visit"],
        summary_terms_by_handle={
            handle: ["visit", f"place{index}"]
            for index, handle in enumerate(handles, start=1)
        },
    )
    incomplete = _parse_candidate(
        "2",
        ("H001", "H002"),
        contract,
        allowed_handles=handles,
    )
    assert incomplete.valid is False
    assert incomplete.error_code == "aggregate_scope_incomplete"

    complete = _parse_candidate("3", handles, contract)
    assert complete.valid is True


def test_complete_proof_deduplicates_semantics_and_excludes_slot_distractors() -> None:
    duplicate_sha = _sha("same-semantic-unit")
    duplicates = _validation_contract(
        ("H001", "H002"),
        answer_shape="number",
        operation="count_or_aggregate",
        semantic_unit_by_handle={"H001": duplicate_sha, "H002": duplicate_sha},
        summary_terms_by_handle={"H001": ["visit"], "H002": ["visit"]},
    )
    deduplicated = _parse_candidate(
        "1",
        ("H001",),
        duplicates,
        allowed_handles=("H001", "H002"),
    )
    assert deduplicated.valid is True

    slot_id = _sha("count-target-slot")
    slot = {
        "handle_ids": ["H001", "H002"],
        "kind": "predicate",
        "label_terms": ["museum"],
        "match_terms": ["museum"],
        "relation_constraint": None,
        "requires_numeric": False,
        "slot_id": slot_id,
    }
    scoped = _validation_contract(
        ("H001", "H002", "H003"),
        answer_shape="number",
        actions_by_handle={
            "H001": ["visit"],
            "H002": ["visit"],
            "H003": ["visit"],
        },
        operation="count_or_aggregate",
        question_actions=["visit"],
        required_slots=[slot],
        summary_terms_by_handle={
            "H001": ["visit", "museum", "one"],
            "H002": ["visit", "museum", "two"],
            "H003": ["visit", "unrelated"],
        },
    )
    selected = _parse_candidate(
        "2",
        ("H001", "H002"),
        scoped,
        allowed_handles=("H001", "H002", "H003"),
    )
    assert selected.valid is True


def test_model_attested_order_requires_distinct_chronological_proof() -> None:
    contract = _validation_contract(
        ("H001", "H002"),
        answer_shape="ordered_list",
        actions_by_handle={"H001": ["visit"], "H002": ["visit"]},
        cardinality=2,
        date_by_handle={"H001": "2025-01-01", "H002": "2025-02-01"},
        operation="order_or_select",
        question_actions=["visit"],
        summary_terms_by_handle={
            "H001": ["visit", "museum"],
            "H002": ["visit", "cafe"],
        },
        temporal_mode="order",
    )
    chronological = _parse_candidate("Museum, Cafe", ("H001", "H002"), contract)
    assert chronological.valid is True

    reversed_order = _parse_candidate("Cafe, Museum", ("H001", "H002"), contract)
    assert reversed_order.valid is False
    assert reversed_order.error_code == "typed_order_entailment"

    undated = _validation_contract(
        ("H001", "H002"),
        answer_shape="ordered_list",
        actions_by_handle={"H001": ["visit"], "H002": ["visit"]},
        cardinality=2,
        operation="order_or_select",
        question_actions=["visit"],
        summary_terms_by_handle={
            "H001": ["visit", "museum"],
            "H002": ["visit", "cafe"],
        },
        temporal_mode="order",
    )
    missing_dates = _parse_candidate("Museum, Cafe", ("H001", "H002"), undated)
    assert missing_dates.valid is False
    assert missing_dates.error_code == "typed_order_entailment"


@pytest.mark.parametrize(
    ("actions", "prediction"),
    [
        (("visit", "visit"), "2"),
        (("learn", "cook", "try", "learn"), "4"),
        (("pickup", "return", "pickup"), "3"),
    ],
)
def test_model_attested_open_counts_use_distinct_typed_semantic_units(
    actions: tuple[str, ...],
    prediction: str,
) -> None:
    handles = tuple(f"H{index + 1:03d}" for index in range(len(actions)))
    contract = _validation_contract(
        handles,
        answer_shape="number",
        actions_by_handle={
            handle: [action] for handle, action in zip(handles, actions, strict=True)
        },
        operation="count_or_aggregate",
        question_actions=list(dict.fromkeys(actions)),
        question_terms=["many", "different"],
        summary_terms_by_handle={
            handle: [action, f"member{index + 1}"]
            for index, (handle, action) in enumerate(
                zip(handles, actions, strict=True)
            )
        },
    )

    parsed = _parse_candidate(prediction, handles, contract)
    assert parsed.valid is True
    assert parsed.validation_basis == "model_attested"


def test_explicit_two_member_claim_contributes_two_without_plural_guessing() -> None:
    handles = ("H001", "H002")
    contract = _validation_contract(
        handles,
        answer_shape="number",
        actions_by_handle={"H001": ["acquire"], "H002": ["acquire"]},
        operation="count_or_aggregate",
        question_actions=["acquire"],
        question_terms=["plant", "acquire"],
        summary_terms_by_handle={
            "H001": ["peace", "lily", "succulent", "acquire"],
            "H002": ["snake", "plant", "acquire"],
        },
    )
    contract["by_handle"]["H001"]["semantic_rows"][0][  # type: ignore[index]
        "explicit_member_count"
    ] = 2

    parsed = _parse_candidate("3", handles, contract)
    assert parsed.valid is True


def test_ordered_difference_and_boolean_are_locally_entailed_without_anchors() -> None:
    left_slot = {
        "handle_ids": ["H001"],
        "kind": "comparison_side",
        "label_terms": ["hawaii"],
        "match_terms": ["hawaii"],
        "relation_constraint": None,
        "requires_numeric": True,
        "slot_id": _sha("left-slot"),
    }
    right_slot = {
        "handle_ids": ["H002"],
        "kind": "comparison_side",
        "label_terms": ["tokyo"],
        "match_terms": ["tokyo"],
        "relation_constraint": None,
        "requires_numeric": True,
        "slot_id": _sha("right-slot"),
    }
    difference = _validation_contract(
        ("H001", "H002"),
        answer_shape="number",
        comparison_mode="difference",
        numeric_by_handle={"H001": 300.0, "H002": 30.0},
        operation="compare_or_calculate",
        question_actions=["spend"],
        question_terms=["hawaii", "tokyo", "spend"],
        required_slots=[left_slot, right_slot],
    )
    assert _parse_candidate("$270", ("H001", "H002"), difference).valid

    boolean = _validation_contract(
        ("H001", "H002"),
        answer_shape="boolean",
        comparison_mode="boolean_greater",
        numeric_by_handle={"H001": 40.0, "H002": 20.0},
        operation="compare_or_calculate",
        question_actions=["receive"],
        question_terms=["hellofresh", "ubereat", "discount"],
        required_slots=[left_slot, right_slot],
    )
    yes = _parse_candidate("Yes.", ("H001", "H002"), boolean)
    assert yes.valid is True
    wrong = _parse_candidate("No", ("H001", "H002"), boolean)
    assert wrong.valid is False
    assert wrong.error_code == "typed_boolean_entailment"


def test_q75_model_attested_difference_must_preserve_safe_qualifiers() -> None:
    left_slot = {
        "handle_ids": ["H001"],
        "kind": "comparison_side",
        "label_terms": ["hawaii"],
        "match_terms": ["hawaii"],
        "relation_constraint": None,
        "requires_numeric": True,
        "slot_id": _sha("q75-left-slot"),
    }
    right_slot = {
        "handle_ids": ["H002"],
        "kind": "comparison_side",
        "label_terms": ["tokyo"],
        "match_terms": ["tokyo"],
        "relation_constraint": None,
        "requires_numeric": True,
        "slot_id": _sha("q75-right-slot"),
    }
    contract = _validation_contract(
        ("H001", "H002"),
        answer_shape="number",
        comparison_mode="difference",
        numeric_by_handle={"H001": 300.0, "H002": 30.0},
        numeric_qualifier_by_handle={
            "H001": "lower_bound",
            "H002": "approximate",
        },
        operation="compare_or_calculate",
        question_actions=["spend"],
        question_terms=["hawaii", "tokyo", "spend"],
        required_slots=[left_slot, right_slot],
    )

    exact = _parse_candidate("$270", ("H001", "H002"), contract)
    assert exact.valid is False
    assert exact.error_code == "typed_numeric_approximation_erased"

    qualified = _parse_candidate(
        "Approximately more than $270",
        ("H001", "H002"),
        contract,
    )
    assert qualified.valid is True
    assert qualified.validation_basis == "model_attested"


def test_set_member_requires_the_question_action_in_its_typed_claim() -> None:
    handles = ("H001", "H002")
    unsupported = _validation_contract(
        handles,
        actions_by_handle={"H001": [], "H002": ["join"]},
        answer_shape="set_list",
        cardinality=2,
        operation="deduplicated_member_join",
        question_actions=["join"],
        question_terms=["two", "hobby", "join", "online", "community"],
        summary_terms_by_handle={
            "H001": ["photography", "online", "community", "recommend"],
            "H002": ["cook", "online", "community", "join"],
        },
    )
    false_join = _parse_candidate(
        "photography and cooking",
        handles,
        unsupported,
    )
    assert false_join.valid is False
    assert false_join.error_code == "typed_list_entailment"

    supported = _validation_contract(
        handles,
        actions_by_handle={"H001": ["join"], "H002": ["join"]},
        answer_shape="set_list",
        cardinality=2,
        operation="deduplicated_member_join",
        question_actions=["join"],
        question_terms=["two", "hobby", "join", "online", "community"],
        summary_terms_by_handle={
            "H001": ["photography", "online", "community", "join"],
            "H002": ["cook", "online", "community", "join"],
        },
    )
    assert _parse_candidate("photography and cooking", handles, supported).valid
