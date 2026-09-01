from __future__ import annotations

import hashlib

from tools.matched_eval.typed_operator_adapter import (
    EvidenceHandleBinding,
    EvidenceOrigin,
    FrontierMode,
    ProvenanceGrade,
    build_typed_evidence_packet,
    parse_typed_items,
)
from tools.matched_eval.typed_downstream_operator import (
    compile_downstream_operator_overlay,
    execute_downstream_typed_operator,
)
from tools.matched_eval.typed_operator_executor import (
    ExecutionStatus,
    assess_candidate_preservation,
    build_evidence_consensus,
    execute_numeric,
    execute_set,
    execute_time,
    execute_typed_operator,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


Q8 = """[Question asked at 2023/05/30 (Tue) 22:48]
I'm looking back at our previous conversation about the Seco de Cordero recipe from Ancash. You mentioned using a light or medium-bodied beer, but I was wondering if you could remind me what type of beer you specifically recommended?"""
Q16 = """[Question asked at 2023/10/15 (Sun) 08:39]
How long have I been living in my current apartment in Harajuku?"""
Q28 = """[Question asked at 2023/03/20 (Mon) 23:57]
How many bikes did I service or plan to service in March?"""
Q47 = """[Question asked at 2023/06/17 (Sat) 16:20]
How many MCU films did I watch in the last 3 months?"""
Q49 = """[Question asked at 2023/05/30 (Tue) 21:17]
I'm planning a trip to Denver soon. Any suggestions on what to do there?"""
Q50 = """[Question asked at 2023/05/30 (Tue) 22:43]
Which social media platform did I gain the most followers on over the past month?"""
Q54 = """[Question asked at 2023/03/25 (Sat) 18:26]
What kitchen appliance did I buy 10 days ago?"""
Q71 = """[Question asked at 2023/05/30 (Tue) 20:26]
I was thinking about our previous conversation about data privacy and security. You mentioned that companies use two-factor authentication to enhance security. Can you remind me what kind of two-factor authentication methods you were referring to?"""
Q77 = """[Question asked at 2023/03/25 (Sat) 17:18]
How many months have passed since I last visited a museum with a friend?"""
Q79 = """[Question asked at 2023/05/30 (Tue) 23:20]
How much did I spend on a designer handbag?"""
Q75 = """[Question asked at 2023/05/25 (Thu) 15:52]
How much more did I spend on accommodations per night in Hawaii compared to Tokyo?"""
Q97 = """[Question asked at 2023/05/30 (Tue) 16:15]
Did I receive a higher percentage discount on my first order from HelloFresh, compared to my first UberEats order?"""


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _binding(index: int, group: int | None = None) -> EvidenceHandleBinding:
    group_index = index if group is None else group
    return EvidenceHandleBinding(
        f"H{index:03d}",
        EvidenceOrigin.MAP,
        ProvenanceGrade.EXACT_CITATION,
        f"G{group_index:03d}",
        _sha("artifact"),
        _sha("parent"),
        _sha(f"item-{index}"),
        _sha(f"payload-{index}"),
        _sha(f"citation-{index}"),
        30,
        _sha(f"local-source-{group_index}"),
    )


def _packet(question: str, raw_items: list[dict[str, object]], *, mode: FrontierMode):
    spec = compile_typed_operator_spec(question)
    handle_count = max(
        int(handle[1:])
        for row in raw_items
        for handle in row["handle_ids"]  # type: ignore[index]
    )
    bindings = tuple(_binding(index) for index in range(1, handle_count + 1))
    parsed = parse_typed_items(raw_items, operator_spec=spec, bindings=bindings)
    assert not parsed.rejected_items
    packet = build_typed_evidence_packet(
        spec,
        bindings,
        parsed,
        sealed_input_artifact_sha256s=(_sha("artifact"),),
        frontier_mode=mode,
    )
    return spec, packet


def test_q28_rescues_two_distinct_service_events_including_plan() -> None:
    spec, packet = _packet(
        Q28,
        [
            {
                "handle_ids": ["H001"],
                "kind": "operand",
                "summary": "One road bike serviced in March",
                "entity_key": "road bike service",
                "numeric_value": 1,
                "numeric_role": "operand",
                "status": "completed",
            },
            {
                "handle_ids": ["H002"],
                "kind": "operand",
                "summary": "One mountain bike planned for service in March",
                "entity_key": "mountain bike service",
                "numeric_value": 1,
                "numeric_role": "operand",
                "status": "proposed",
            },
        ],
        mode=FrontierMode.EXHAUSTIVE,
    )
    assert spec.include_proposed is True
    result = execute_typed_operator(spec, packet)
    assert result.status is ExecutionStatus.SUPPORTED
    assert result.prediction == "2"
    assert result.numeric_result == 2
    assert result.provider_prompt_count == 0
    assert result.retained_transformer_token_state_bytes == 0


def test_q47_count_abstains_without_exhaustive_frontier() -> None:
    spec, packet = _packet(
        Q47,
        [
            {
                "handle_ids": [f"H{index:03d}"],
                "kind": "operand",
                "summary": f"MCU film event {index} watched in the last three months",
                "entity_key": f"MCU film {index}",
                "numeric_value": 1,
                "numeric_role": "operand",
                "status": "completed",
            }
            for index in range(1, 5)
        ],
        mode=FrontierMode.BOUNDED,
    )
    result = execute_numeric(spec, packet)
    assert result.status is ExecutionStatus.INSUFFICIENT
    assert result.prediction == ""
    assert result.reason == "frontier_not_closed"


def test_q50_computes_mixed_baseline_end_and_delta_groups() -> None:
    spec, packet = _packet(
        Q50,
        [
            {
                "handle_ids": ["H001"],
                "kind": "operand",
                "summary": "TikTok started the month at 100 followers",
                "group_key": "TikTok",
                "numeric_value": 100,
                "numeric_role": "baseline",
            },
            {
                "handle_ids": ["H002"],
                "kind": "operand",
                "summary": "TikTok ended the month at 300 followers",
                "group_key": "TikTok",
                "numeric_value": 300,
                "numeric_role": "end",
            },
            {
                "handle_ids": ["H003"],
                "kind": "operand",
                "summary": "Twitter gained 120 followers during the month",
                "group_key": "Twitter",
                "numeric_value": 120,
                "numeric_role": "delta",
            },
        ],
        mode=FrontierMode.EXHAUSTIVE,
    )
    result = execute_typed_operator(spec, packet)
    assert result.status is ExecutionStatus.SUPPORTED
    assert result.prediction == "TikTok"
    assert result.numeric_result == 200


def test_q75_and_q97_comparisons_follow_spec_side_order_not_item_order() -> None:
    q75_items = [
        {
            "handle_ids": ["H001"],
            "kind": "operand",
            "summary": "Tokyo accommodations cost $30 per night.",
            "entity_key": "Tokyo",
            "numeric_value": 30,
            "numeric_role": "operand",
            "unit": "$",
        },
        {
            "handle_ids": ["H002"],
            "kind": "operand",
            "summary": "Hawaii accommodations cost $300 per night.",
            "entity_key": "Hawaii",
            "numeric_value": 300,
            "numeric_role": "operand",
            "unit": "$",
        },
    ]
    for ordered in (q75_items, list(reversed(q75_items))):
        spec, packet = _packet(Q75, ordered, mode=FrontierMode.EXHAUSTIVE)
        result = execute_numeric(spec, packet)
        assert result.status is ExecutionStatus.SUPPORTED
        assert result.prediction == "$270"
        assert result.numeric_result == 270

    q97_items = [
        {
            "handle_ids": ["H001"],
            "kind": "operand",
            "summary": (
                "I received a percentage discount of 20 on my first "
                "UberEats order."
            ),
            "entity_key": "UberEats",
            "numeric_value": 20,
            "numeric_role": "operand",
            "unit": "%",
        },
        {
            "handle_ids": ["H002"],
            "kind": "operand",
            "summary": (
                "I received a percentage discount of 40 on my first "
                "HelloFresh order."
            ),
            "entity_key": "HelloFresh",
            "numeric_value": 40,
            "numeric_role": "operand",
            "unit": "%",
        },
    ]
    for ordered in (q97_items, list(reversed(q97_items))):
        spec, packet = _packet(Q97, ordered, mode=FrontierMode.EXHAUSTIVE)
        result = execute_numeric(spec, packet)
        assert result.status is ExecutionStatus.SUPPORTED
        assert result.prediction == "Yes"
        assert result.numeric_result == 20


def test_qualified_q75_operands_never_emit_an_exact_deterministic_scalar() -> None:
    spec, packet = _packet(
        Q75,
        [
            {
                "handle_ids": ["H001"],
                "kind": "operand",
                "summary": "I spent over $300 per night in Hawaii.",
                "entity_key": "Hawaii",
                "numeric_value": 300,
                "numeric_role": "operand",
                "numeric_qualifier": "lower_bound",
                "unit": "$",
            },
            {
                "handle_ids": ["H002"],
                "kind": "operand",
                "summary": "Tokyo accommodations cost around $30 per night.",
                "entity_key": "Tokyo",
                "numeric_value": 30,
                "numeric_role": "operand",
                "numeric_qualifier": "approximate",
                "unit": "$",
            },
        ],
        mode=FrontierMode.EXHAUSTIVE,
    )
    result = execute_numeric(spec, packet)
    assert result.status is ExecutionStatus.INSUFFICIENT
    assert result.prediction == ""
    assert result.reason == "qualified_numeric_operands_require_model"


def test_q79_latest_explicit_transaction_state_rescues_800() -> None:
    spec, packet = _packet(
        Q79,
        [
            {
                "handle_ids": ["H001"],
                "kind": "state",
                "summary": "Designer handbag purchased in January 2023 for $2,000",
                "entity_key": "designer handbag purchase",
                "numeric_value": 2000,
                "numeric_role": "end",
                "unit": "$",
                "date": "January 2023",
                "status": "completed",
                "value_authority": "explicit",
            },
            {
                "handle_ids": ["H002"],
                "kind": "state",
                "summary": "Designer handbag purchased in May 2023 for $800",
                "entity_key": "designer handbag purchase",
                "numeric_value": 800,
                "numeric_role": "end",
                "unit": "$",
                "date": "May 2023",
                "status": "completed",
                "value_authority": "explicit",
            },
        ],
        mode=FrontierMode.EXHAUSTIVE,
    )
    result = execute_typed_operator(spec, packet)
    assert result.status is ExecutionStatus.SUPPORTED
    assert result.prediction == "$800"
    assert result.numeric_result == 800


def test_q16_and_q77_use_query_timestamp_as_implicit_interval_end() -> None:
    q16_spec, q16_packet = _packet(
        Q16,
        [
            {
                "handle_ids": ["H001"],
                "kind": "state",
                "summary": "I began living in my current apartment in Harajuku on 2023-07-15",
                "entity_key": "current apartment in Harajuku",
                "date": "2023-07-15",
                "status": "current",
            }
        ],
        mode=FrontierMode.EXHAUSTIVE,
    )
    q16_result = execute_time(q16_spec, q16_packet)
    assert q16_result.status is ExecutionStatus.SUPPORTED
    assert q16_result.prediction == "3 months"

    q77_spec, q77_packet = _packet(
        Q77,
        [
            {
                "handle_ids": ["H001"],
                "kind": "event",
                "summary": "I visited a museum with a friend on 2022-10-25",
                "entity_key": "museum visit with a friend",
                "date": "2022-10-25",
                "participant_count": 1,
                "status": "completed",
            }
        ],
        mode=FrontierMode.EXHAUSTIVE,
    )
    q77_result = execute_time(q77_spec, q77_packet)
    assert q77_result.status is ExecutionStatus.SUPPORTED
    assert q77_result.prediction == "5 months"


def test_time_interval_accepts_mixed_naive_and_offset_aware_boundaries() -> None:
    spec, packet = _packet(
        Q16,
        [
            {
                "handle_ids": ["H001"],
                "kind": "state",
                "summary": (
                    "I began living in my current apartment in Harajuku "
                    "on 2023-07-15"
                ),
                "entity_key": "current apartment in Harajuku",
                "date": "2023-07-15T01:00:00+02:00",
                "status": "completed",
            },
            {
                "handle_ids": ["H002"],
                "kind": "state",
                "summary": (
                    "I was living in my current apartment in Harajuku "
                    "on 2023-10-15"
                ),
                "entity_key": "current apartment in Harajuku",
                "date": "2023-10-15",
                "status": "current",
            },
        ],
        mode=FrontierMode.EXHAUSTIVE,
    )
    result = execute_time(spec, packet)
    assert result.status is ExecutionStatus.SUPPORTED
    assert result.prediction == "3 months"


def test_time_order_normalizes_non_utc_offset_before_dropping_tzinfo() -> None:
    question = """[Question asked at 2023/08/03 (Thu) 10:00]
In what chronological order did I visit Alpha and Beta?"""
    spec, packet = _packet(
        question,
        [
            {
                "handle_ids": ["H001"],
                "kind": "event",
                "summary": "I visited Alpha just before midnight UTC",
                "entity_key": "Alpha",
                "date": "2023-08-02T01:00:00+02:00",
                "status": "completed",
            },
            {
                "handle_ids": ["H002"],
                "kind": "event",
                "summary": "I visited Beta at midnight",
                "entity_key": "Beta",
                "date": "2023-08-02",
                "status": "completed",
            },
        ],
        mode=FrontierMode.EXHAUSTIVE,
    )
    result = execute_time(spec, packet)
    assert result.status is ExecutionStatus.SUPPORTED
    # 01:00 at +02:00 is 23:00 UTC on the prior day. Merely stripping the
    # offset would reverse this order, so this proves normalization precedes
    # removal of tzinfo.
    assert result.prediction == "Alpha → Beta"


def test_latest_state_orders_mixed_naive_and_non_utc_offset_instants() -> None:
    spec, packet = _packet(
        Q79,
        [
            {
                "handle_ids": ["H001"],
                "kind": "state",
                "summary": "Designer handbag purchase was $100 before midnight UTC",
                "entity_key": "designer handbag purchase",
                "numeric_value": 100,
                "numeric_role": "end",
                "unit": "$",
                "date": "2023-08-02T01:00:00+02:00",
                "status": "completed",
                "value_authority": "explicit",
            },
            {
                "handle_ids": ["H002"],
                "kind": "state",
                "summary": "Designer handbag purchase was $200 at midnight",
                "entity_key": "designer handbag purchase",
                "numeric_value": 200,
                "numeric_role": "end",
                "unit": "$",
                "date": "2023-08-02",
                "status": "completed",
                "value_authority": "explicit",
            },
        ],
        mode=FrontierMode.EXHAUSTIVE,
    )
    result = execute_typed_operator(spec, packet)
    assert result.status is ExecutionStatus.SUPPORTED
    assert result.prediction == "$200"
    assert result.numeric_result == 200


def test_set_executor_deduplicates_members_and_requires_closed_frontier() -> None:
    question = """[Question asked at 2023/05/30 (Tue) 10:00]
What are all the cities I visited?"""
    spec, packet = _packet(
        question,
        [
            {
                "handle_ids": ["H001"],
                "kind": "member",
                "summary": "Visited Denver",
                "entity_key": "Denver",
            },
            {
                "handle_ids": ["H002"],
                "kind": "member",
                "summary": "Denver was visited",
                "entity_key": "Denver",
            },
            {
                "handle_ids": ["H003"],
                "kind": "member",
                "summary": "Visited Seattle",
                "entity_key": "Seattle",
            },
        ],
        mode=FrontierMode.EXHAUSTIVE,
    )
    result = execute_set(spec, packet)
    assert result.status is ExecutionStatus.SUPPORTED
    assert result.prediction == "Denver, Seattle"


def test_question_compiled_set_cardinality_blocks_overfilled_closed_set() -> None:
    question = """[Question asked at 2026/08/28 10:00]
What are the two workshop tools I stored?"""
    spec, packet = _packet(
        question,
        [
            {
                "handle_ids": [f"H{index:03d}"],
                "kind": "member",
                "summary": f"Stored the {name} workshop tool",
                "entity_key": name,
            }
            for index, name in enumerate(("wrench", "mallet", "drill"), start=1)
        ],
        mode=FrontierMode.EXHAUSTIVE,
    )

    legacy_receipt = spec.receipt_sha256
    assert spec.cardinality is None
    overlay = compile_downstream_operator_overlay(question, spec)
    assert overlay.effective_set_cardinality == 2
    assert spec.receipt_sha256 == legacy_receipt

    legacy_result = execute_set(spec, packet)
    assert legacy_result.status is ExecutionStatus.SUPPORTED
    result = execute_downstream_typed_operator(spec, packet, overlay)
    assert result.status is ExecutionStatus.INSUFFICIENT
    assert result.prediction == ""
    assert result.reason.startswith("downstream_set_cardinality_not_closed:")


def test_downstream_set_cardinality_accepts_number_words_and_digits() -> None:
    for count_text, expected in (("ten", 10), ("14", 14)):
        question = (
            "[Question asked at 2026/08/28 10:00]\n"
            f"What are the {count_text} workshop tools I stored?"
        )
        spec = compile_typed_operator_spec(question)
        overlay = compile_downstream_operator_overlay(question, spec)

        assert spec.cardinality is None
        assert overlay.effective_set_cardinality == expected


def test_q8_q49_q71_shadow_rewrites_cannot_erase_selected_detail() -> None:
    q8_spec, q8_packet = _packet(
        Q8,
        [
            {
                "handle_ids": ["H001"],
                "summary": "The specific recommended beers were pilsner or lager",
                "specificity_terms": ["pilsner", "lager"],
            }
        ],
        mode=FrontierMode.EXHAUSTIVE,
    )
    q8_receipt = assess_candidate_preservation(
        q8_spec, q8_packet, "A light or medium-bodied beer."
    )
    assert q8_receipt.preserves_required_content is False
    assert set(q8_receipt.missing_specificity_terms) == {"pilsner", "lager"}

    q49_spec, q49_packet = _packet(
        Q49,
        [
            {
                "handle_ids": ["H001"],
                "kind": "claim",
                "summary": "Denver suggestions grounded in prior interests",
                "personalization_anchors": ["Red Rocks", "Union Station"],
            }
        ],
        mode=FrontierMode.EXHAUSTIVE,
    )
    q49_receipt = assess_candidate_preservation(
        q49_spec, q49_packet, "Visit museums and try local restaurants."
    )
    assert q49_receipt.preserves_required_content is False
    assert set(q49_receipt.missing_personalization_anchors) == {
        "red", "rock", "union", "station"
    }

    q71_spec, q71_packet = _packet(
        Q71,
        [
            {
                "handle_ids": ["H001"],
                "summary": "The methods were biometric verification and OTP",
                "specificity_terms": ["biometric", "OTP"],
            }
        ],
        mode=FrontierMode.EXHAUSTIVE,
    )
    q71_receipt = assess_candidate_preservation(
        q71_spec, q71_packet, "A one-time password (OTP)."
    )
    assert q71_receipt.preserves_required_content is False
    assert q71_receipt.missing_specificity_terms == ("biometric",)


def test_consensus_reports_cross_group_support_without_source_identifiers() -> None:
    spec = compile_typed_operator_spec(Q28)
    bindings = (_binding(1, group=1), _binding(2, group=2))
    parsed = parse_typed_items(
        [
            {
                "handle_ids": ["H001"],
                "kind": "operand",
                "summary": "One road bike service in March",
                "entity_key": "road bike service",
                "numeric_value": 1,
                "numeric_role": "operand",
            },
            {
                "handle_ids": ["H002"],
                "kind": "operand",
                "summary": "One road bike service in March",
                "entity_key": "road bike service",
                "numeric_value": 1,
                "numeric_role": "operand",
            },
        ],
        operator_spec=spec,
        bindings=bindings,
    )
    packet = build_typed_evidence_packet(
        spec,
        bindings,
        parsed,
        sealed_input_artifact_sha256s=(_sha("artifact"),),
        frontier_mode=FrontierMode.EXHAUSTIVE,
    )
    consensus = build_evidence_consensus(packet)
    assert len(consensus.groups) == 1
    assert consensus.groups[0].support_count == 2
    assert consensus.groups[0].cross_group_support_count == 2
    assert "local-source" not in repr(consensus.projection())


def test_relative_select_without_an_executable_target_never_becomes_a_timeline() -> None:
    spec, packet = _packet(
        Q54,
        [
            {
                "handle_ids": ["H001"],
                "kind": "event",
                "summary": "I bought a smoker today.",
                "date": "2023-03-15T04:56:00-07:00",
                "status": "completed",
            },
            {
                "handle_ids": ["H002"],
                "kind": "event",
                "summary": "I bought a toaster last Saturday.",
                "date": "2023-03-18T10:00:00-07:00",
                "status": "completed",
            },
        ],
        mode=FrontierMode.EXHAUSTIVE,
    )
    assert spec.temporal_window_days is None
    result = execute_time(spec, packet)
    assert result.status is ExecutionStatus.INSUFFICIENT
    assert result.prediction == ""
    assert result.used_handle_ids == ()
    assert result.reason == "relative_selection_target_unresolved"
