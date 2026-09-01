from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from memory_condense.domain._tokenizer import count_tokens
from tools.matched_eval.contracts import MatchedEvalContractError
from tools.matched_eval.contracts import StageDisposition, identity_sha256
from tools.matched_eval.query_evidence_map_solver_v2_live import (
    MAP_ITEM_FORMAT,
    EvidenceMapPlanRow,
    ValidatedMapItem,
    VerifiedEvidenceMapRow,
)
from tools.matched_eval.query_payload_live import PayloadEvidenceAlias
from tools.matched_eval.typed_operator_adapter import (
    ContentCoherence,
    EvidenceHandleBinding,
    EvidenceOrigin,
    EvidenceStatus,
    FrontierMode,
    HARD_PROMPT_TOKEN_CAP,
    NumericQualifier,
    ProviderPayloadMode,
    ProvenanceGrade,
    TypedEvidenceContribution,
    adapt_verified_evidence,
    build_typed_evidence_packet,
    compact_typed_evidence_projection,
    conservative_numeric_value,
    merge_typed_evidence_contributions,
    parse_typed_items,
)
from tools.matched_eval.typed_operator_spec import compile_typed_operator_spec


Q28 = """[Question asked at 2023/03/20 (Mon) 23:57]
How many bikes did I service or plan to service in March?"""
Q97 = """[Question asked at 2023/05/30 (Tue) 16:15]
Did I receive a higher percentage discount on my first order from HelloFresh, compared to my first UberEats order?"""


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("I had 6 plants on 2023-03-01.", 6.0),
        ("On 2023-03-01 I had 6 plants.", 6.0),
        ("The note was recorded on 2023-03-01.", None),
        ("I had 6 plants and then 8 plants.", None),
        ("I serviced my road bike on March 2.", None),
        ("The aquarium plants need a 31-day treatment window.", None),
        ("Top 5 Hawaii experiences.", None),
    ),
)
def test_conservative_numeric_value_never_promotes_iso_date_fragments(
    text: str, expected: float | None
) -> None:
    assert conservative_numeric_value(text) == expected


def test_parser_strips_incompatible_raw_numeric_and_infers_qualifier() -> None:
    q28_spec = compile_typed_operator_spec(Q28)
    q28 = parse_typed_items(
        [
            {
                "handle_ids": ["H001"],
                "summary": "I serviced my road bike on March 2.",
                "numeric_value": 2,
                "numeric_role": "operand",
            }
        ],
        operator_spec=q28_spec,
        bindings=(_binding(1),),
    ).accepted_items[0]
    assert q28.numeric_value is None
    assert q28.numeric_role.value == "none"
    assert q28.numeric_qualifier is NumericQualifier.EXACT

    q75 = """[Question asked at 2023/05/30 (Tue) 22:16]
How much more did I spend on accommodations per night in Hawaii compared to Tokyo?"""
    q75_spec = compile_typed_operator_spec(q75)
    bounded = parse_typed_items(
        [
            {
                "handle_ids": ["H001"],
                "summary": "I spent over $300 per night in Hawaii.",
                "numeric_value": 300,
                "numeric_role": "operand",
                "unit": "$",
            }
        ],
        operator_spec=q75_spec,
        bindings=(_binding(1),),
    ).accepted_items[0]
    assert bounded.numeric_qualifier is NumericQualifier.LOWER_BOUND


def test_factual_negative_polarity_is_evidence_not_content_conflict() -> None:
    spec = compile_typed_operator_spec(Q97)
    bindings = (_binding(1), _binding(2))
    parsed = parse_typed_items(
        [
            {
                "handle_ids": ["H001"],
                "summary": "No: the HelloFresh first-order discount was 20 percent.",
                "numeric_value": 20,
                "numeric_role": "operand",
                "unit": "%",
            },
            {
                "handle_ids": ["H002"],
                "summary": "The UberEats first-order discount was 40 percent.",
                "numeric_value": 40,
                "numeric_role": "operand",
                "unit": "%",
            },
        ],
        operator_spec=spec,
        bindings=bindings,
    )
    assert len(parsed.accepted_items) == 2
    assert all(not row.content_conflict for row in parsed.accepted_items)
    assert all(row.content_coherence is ContentCoherence.MATCH for row in parsed.accepted_items)


def test_meta_unknown_is_unresolved_while_pending_obligation_is_current() -> None:
    spec = compile_typed_operator_spec(Q28)
    bindings = (_binding(1), _binding(2))
    parsed = parse_typed_items(
        [
            {
                "handle_ids": ["H001"],
                "summary": "The March bike-service count is unknown.",
                "numeric_value": 1,
                "numeric_role": "operand",
            },
            {
                "handle_ids": ["H002"],
                "summary": "One mountain bike has not yet been serviced in March.",
                "numeric_value": 1,
                "numeric_role": "operand",
            },
        ],
        operator_spec=spec,
        bindings=bindings,
    )
    unknown, pending = parsed.accepted_items
    assert unknown.content_coherence is ContentCoherence.UNRESOLVED
    assert not unknown.content_conflict
    assert pending.status is EvidenceStatus.CURRENT
    assert pending.content_coherence is ContentCoherence.MATCH


def _binding(
    index: int,
    *,
    citation_chars: int = 20,
    artifact: str = "map-artifact",
    group: int | None = None,
) -> EvidenceHandleBinding:
    citation = "c" * citation_chars
    group_index = index if group is None else group
    return EvidenceHandleBinding(
        handle_id=f"H{index:03d}",
        origin=EvidenceOrigin.MAP,
        provenance_grade=ProvenanceGrade.EXACT_CITATION,
        source_group_handle=f"G{group_index:03d}",
        sealed_artifact_sha256=_sha(artifact),
        parent_receipt_sha256=_sha("map-parent"),
        evidence_receipt_sha256=_sha(f"map-item-{index}"),
        payload_sha256=_sha(f"payload-{index}"),
        citation_sha256=_sha(citation),
        citation_char_count=len(citation),
        local_source_locator_sha256=_sha(f"raw-source::{index}"),
    )


def _map_item(
    item_id: str,
    alias: str,
    candidate: str,
    citation: str,
    source_index: int,
) -> ValidatedMapItem:
    body = {
        "alias": alias,
        "candidate": candidate,
        "citation": citation,
        "citation_match": "exact",
        "format": MAP_ITEM_FORMAT,
        "item_id": item_id,
        "kind": "operand",
        "source_index": source_index,
    }
    return ValidatedMapItem(
        item_id,
        source_index,
        "operand",
        alias,
        citation,
        candidate,
        "exact",
        identity_sha256(body),
    )


def test_541_character_citation_is_local_and_one_bad_item_is_salvaged() -> None:
    spec = compile_typed_operator_spec(Q28)
    bindings = (_binding(1, citation_chars=541), _binding(2))
    parsed = parse_typed_items(
        [
            {
                "handle_ids": ["H001"],
                "kind": "operand",
                "summary": "One road bike serviced in March",
                "entity_key": "road bike service",
                "numeric_value": 1,
                "numeric_role": "operand",
            },
            {
                "handle_ids": ["H001"],
                "summary": "malformed sibling",
                "numeric_value": "one",
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
        operator_spec=spec,
        bindings=bindings,
    )
    assert len(parsed.accepted_items) == 2
    assert len(parsed.rejected_items) == 1
    assert parsed.rejected_items[0].reason == "numeric_schema"

    packet = build_typed_evidence_packet(
        spec,
        bindings,
        parsed,
        sealed_input_artifact_sha256s=(_sha("map-artifact"),),
        frontier_mode=FrontierMode.EXHAUSTIVE,
    )
    assert packet.frontier.closed is True
    assert packet.provider_prompt_count == 0
    assert packet.retained_transformer_token_state_bytes == 0
    assert packet.hard_prompt_token_cap == HARD_PROMPT_TOKEN_CAP
    assert packet.provider_payload_token_proxy + packet.output_token_reserve <= 8_000
    provider_payload = packet.render_provider_payload()
    assert "c" * 541 not in provider_payload
    assert "raw-source" not in provider_payload
    assert packet.local_bindings[0].citation_char_count == 541


def test_overlong_item_is_rejected_without_invalidating_small_sibling() -> None:
    spec = compile_typed_operator_spec(Q28)
    bindings = (_binding(1),)
    parsed = parse_typed_items(
        [
            {
                "handle_ids": ["H001"],
                "summary": ("overflow " * 10_000).strip(),
                "numeric_value": 99,
                "numeric_role": "operand",
            },
            {
                "handle_ids": ["H001"],
                "summary": "One bike service in March",
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
        sealed_input_artifact_sha256s=(_sha("map-artifact"),),
        frontier_mode=FrontierMode.EXHAUSTIVE,
    )
    assert len(packet.items) == 1
    assert packet.items[0].summary == "One bike service in March"
    assert any(row.reason == "hard_8k_item_overflow" for row in packet.rejected_items)
    assert packet.frontier.closed is True


def test_budget_salvage_accounts_for_late_frontier_growth_and_keeps_fair_prefix() -> None:
    spec = compile_typed_operator_spec(
        "[Question asked at 2023/03/20 (Mon) 23:57] What did I do?"
    )
    first = tuple(_binding(index) for index in range(1, 9))
    second = tuple(
        _binding(index, group=index) for index in range(100_001, 100_009)
    )
    bindings = (*first, *second)
    # A fair allocator places one protected item from each mechanism first.
    # Later fill items can overflow and add rejection receipts to the frontier,
    # but that growth must neither invalidate final construction nor evict the
    # protected prefix.
    fair_order = (first[0], second[0], *first[1:], *second[1:])
    raw_items = [
        {
            "handle_ids": [binding.handle_id],
            "summary": (
                f"fair item {index} " + ("detail " * 180)
            ).strip(),
        }
        for index, binding in enumerate(fair_order)
    ]
    parsed = parse_typed_items(
        raw_items,
        operator_spec=spec,
        bindings=bindings,
    )
    packet = build_typed_evidence_packet(
        spec,
        bindings,
        parsed,
        sealed_input_artifact_sha256s=(_sha("map-artifact"),),
        frontier_mode=FrontierMode.BOUNDED,
    )

    represented = {
        handle for item in packet.items for handle in item.handle_ids
    }
    assert {first[0].handle_id, second[0].handle_id} <= represented
    assert packet.rejected_items
    assert packet.frontier.rejected_item_receipt_sha256s == tuple(
        row.rejection_sha256 for row in packet.rejected_items
    )
    assert (
        packet.provider_payload_token_proxy + packet.output_token_reserve
        <= HARD_PROMPT_TOKEN_CAP
    )


def test_compact_final_mode_prevents_canonical_projection_tax() -> None:
    question = (
        "[Question asked at 2026/08/27 12:00]\n"
        "What color was my bicycle?"
    )
    spec = compile_typed_operator_spec(question)
    bindings = tuple(_binding(index) for index in range(1, 19))
    summaries = tuple(
        (f"candidate {index} " + ("detail " * 170)).strip()
        for index in range(1, 19)
    )
    parsed = parse_typed_items(
        [
            {"handle_ids": [binding.handle_id], "summary": summary}
            for binding, summary in zip(bindings, summaries, strict=True)
        ],
        operator_spec=spec,
        bindings=bindings,
    )
    canonical = build_typed_evidence_packet(
        spec,
        bindings,
        parsed,
        sealed_input_artifact_sha256s=(_sha("map-artifact"),),
        output_token_reserve=1,
    )
    compact = build_typed_evidence_packet(
        spec,
        bindings,
        parsed,
        sealed_input_artifact_sha256s=(_sha("map-artifact"),),
        output_token_reserve=1,
        provider_payload_mode=ProviderPayloadMode.COMPACT_FINAL,
    )

    assert len(canonical.items) < len(summaries)
    assert len(compact.items) == len(summaries)
    assert compact.local_bindings == bindings
    assert tuple(item.summary for item in compact.items) == summaries
    assert compact.provider_projection() == compact_typed_evidence_projection(
        compact
    )
    rendered = json.dumps(
        compact.provider_projection(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert compact.provider_payload_token_proxy == count_tokens(rendered)
    assert compact.provider_payload_token_proxy + compact.output_token_reserve <= 8_000
    assert compact.receipt_sha256 != canonical.receipt_sha256
    assert compact.projection()["provider_payload_mode"] == "compact_final"
    assert "local_source_locator" not in rendered
    assert "sealed_artifact_sha256" not in rendered


def test_unknown_gold_bearing_schema_key_is_rejected_per_item() -> None:
    spec = compile_typed_operator_spec(Q28)
    bindings = (_binding(1),)
    parsed = parse_typed_items(
        [
            {
                "handle_ids": ["H001"],
                "summary": "One bike service in March",
                "reference": "forbidden posthoc value",
            },
            {
                "handle_ids": ["H001"],
                "summary": "One bike service in March",
                "numeric_value": 1,
                "numeric_role": "operand",
            },
        ],
        operator_spec=spec,
        bindings=bindings,
    )
    assert len(parsed.accepted_items) == 1
    assert len(parsed.rejected_items) == 1
    assert parsed.rejected_items[0].reason == "item_schema"


def test_mem0_origin_cannot_overstate_inferred_text_as_exact_citation() -> None:
    with pytest.raises(MatchedEvalContractError, match="overstates"):
        EvidenceHandleBinding(
            "H001",
            EvidenceOrigin.MEM0,
            ProvenanceGrade.EXACT_CITATION,
            "G001",
            _sha("mem0-artifact"),
            _sha("mem0-request"),
            _sha("mem0-record"),
            _sha("inferred-memory-text"),
            _sha("inferred-memory-text"),
            20,
            _sha("mem0-memory-id"),
        )

    binding = EvidenceHandleBinding(
        "H001",
        EvidenceOrigin.MEM0,
        ProvenanceGrade.INFERRED_MEMORY,
        "G001",
        _sha("mem0-artifact"),
        _sha("mem0-request"),
        _sha("mem0-record"),
        _sha("inferred-memory-text"),
        _sha("inferred-memory-text"),
        20,
        _sha("mem0-memory-id"),
    )
    assert binding.opaque().projection()["provenance_grade"] == "inferred_memory"


def test_exact_v2_map_seam_uses_only_opaque_provider_handles() -> None:
    spec = compile_typed_operator_spec(Q28)
    long_citation = "z" * 541
    items = (
        _map_item("M001", "S001", "1 road bike serviced in March", long_citation, 0),
        _map_item("M002", "S002", "1 mountain bike planned for service in March", "short citation", 1),
    )
    raw_source = "a9f6b44c::answer_private_source"
    aliases = tuple(
        PayloadEvidenceAlias(
            alias=item.alias,
            tier="protected_s0",
            rank=index,
            evidence_id=_sha(f"evidence-{index}"),
            source_id=raw_source,
            text_sha256=_sha(item.citation),
            token_count=1,
        )
        for index, item in enumerate(items, start=1)
    )
    source = SimpleNamespace(
        ordinal=28,
        packet=SimpleNamespace(dated_question_sha256=spec.question_sha256),
    )
    plan_row = EvidenceMapPlanRow(
        direct_plan_row=SimpleNamespace(adapter=SimpleNamespace(source=source)),
        direct_answer_row=SimpleNamespace(),
        aliases=aliases,
        retained_query_delta=(),
        dropped_query_delta_ids=(),
        messages=None,
        messages_sha256=None,
        prompt_id=None,
        prompt_token_proxy=None,
        alias_receipt_sha256=_sha("aliases"),
        packet_id=_sha("packet"),
        receipt_sha256=_sha("map-plan-row"),
        disposition=StageDisposition.ADDED,
        reason="test exact seam",
    )
    terminal = VerifiedEvidenceMapRow(
        ordinal=28,
        question_id="opaque-test-question",
        question_sha256=_sha("question"),
        dated_question_sha256=spec.question_sha256,
        route_id=spec.style.value,
        answer_kind="operand",
        accepted_items=items,
        rejected_items=(),
        map_status="validated_items",
        map_parse_receipt_sha256=_sha("map-parse"),
        map_plan_row_receipt_sha256=plan_row.receipt_sha256,
        direct_parent_prediction_sha256=_sha("parent-prediction"),
        source_row_sha256=_sha("source-row"),
        runtime_row_id=_sha("runtime-row"),
        call_key_sha256=None,
        request_journal_sha256=None,
        response_journal_sha256=None,
    )
    packet = adapt_verified_evidence(
        spec,
        plan_row,
        terminal,
        map_artifact_sha256=_sha("map-run"),
        frontier_mode=FrontierMode.EXHAUSTIVE,
    )
    assert len(packet.items) == 2
    assert packet.local_bindings[0].citation_char_count == 541
    assert {row.source_group_handle for row in packet.handles} == {"G001"}
    provider_payload = packet.render_provider_payload()
    assert raw_source not in provider_payload
    assert "a9f6b44c" not in provider_payload
    assert all(row.handle_id.startswith("H") for row in packet.handles)

    rebased = adapt_verified_evidence(
        spec,
        plan_row,
        terminal,
        map_artifact_sha256=_sha("map-run"),
        frontier_mode=FrontierMode.EXHAUSTIVE,
        handle_start=101,
        group_start=201,
    )
    assert tuple(row.handle_id for row in rebased.handles) == ("H101", "H102")
    assert {row.source_group_handle for row in rebased.handles} == {"G201"}


def test_contribution_merge_never_upgrades_a_bounded_tail_to_exhaustive() -> None:
    spec = compile_typed_operator_spec(Q28)
    first_binding = (_binding(1),)
    second_binding = (_binding(2, artifact="tail-artifact"),)
    first = parse_typed_items(
        [
            {
                "handle_ids": ["H001"],
                "summary": "One road bike serviced in March",
                "entity_key": "road bike service",
                "numeric_value": 1,
                "numeric_role": "operand",
            }
        ],
        operator_spec=spec,
        bindings=first_binding,
    )
    second = parse_typed_items(
        [
            {
                "handle_ids": ["H002"],
                "summary": "One mountain bike planned for service in March",
                "entity_key": "mountain bike service",
                "numeric_value": 1,
                "numeric_role": "operand",
            }
        ],
        operator_spec=spec,
        bindings=second_binding,
    )
    packet = merge_typed_evidence_contributions(
        spec,
        (
            TypedEvidenceContribution(
                "adaptive_v3",
                first_binding,
                first,
                _sha("map-artifact"),
                FrontierMode.EXHAUSTIVE,
                False,
            ),
            TypedEvidenceContribution(
                "full_store_tail",
                second_binding,
                second,
                _sha("tail-artifact"),
                FrontierMode.BOUNDED,
                False,
            ),
        ),
    )
    assert len(packet.items) == 2
    assert packet.frontier.mode is FrontierMode.BOUNDED
    assert packet.frontier.closed is False
    assert packet.sealed_input_artifact_sha256s == (
        _sha("map-artifact"),
        _sha("tail-artifact"),
    )


def test_independent_g001_groups_cannot_be_silently_merged() -> None:
    spec = compile_typed_operator_spec(Q28)
    first_binding = (_binding(1, group=1),)
    second_binding = (_binding(2, artifact="tail-artifact", group=1),)
    first = parse_typed_items(
        [{"handle_ids": ["H001"], "summary": "One road bike", "numeric_value": 1}],
        operator_spec=spec,
        bindings=first_binding,
    )
    second = parse_typed_items(
        [{"handle_ids": ["H002"], "summary": "One mountain bike", "numeric_value": 1}],
        operator_spec=spec,
        bindings=second_binding,
    )
    with pytest.raises(MatchedEvalContractError, match="source groups collide"):
        merge_typed_evidence_contributions(
            spec,
            (
                TypedEvidenceContribution(
                    "adaptive_v3",
                    first_binding,
                    first,
                    _sha("map-artifact"),
                    FrontierMode.EXHAUSTIVE,
                    False,
                ),
                TypedEvidenceContribution(
                    "full_store_tail",
                    second_binding,
                    second,
                    _sha("tail-artifact"),
                    FrontierMode.EXHAUSTIVE,
                    False,
                ),
            ),
        )
