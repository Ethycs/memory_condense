from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, Sequence

import pytest

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval._retrieval_qa_prompt import (
    RESPONDER_OUTPUT_TOKEN_RESERVE,
    build_qa_prompt,
)
from tools.matched_eval.contracts import MatchedEvalContractError, identity_sha256
from tools.matched_eval.hot_v3_typed_operator_adapter import (
    PROVIDER_PACKET_FORMAT,
    adapt_hot_v3_arm,
)
from tools.matched_eval.operator_first_numeric_policy import (
    compile_operator_first_numeric_candidates,
    execute_operator_first_numeric_policy,
)
from tools.matched_eval.typed_operator_adapter import FrontierMode
from tools.matched_eval.typed_operator_executor import (
    ExecutionStatus,
    build_slot_closure,
    execute_typed_operator,
)


SELECTION_SHA256 = "a" * 64


def _evidence(
    chunk_id: str,
    text: str,
    *,
    source_id: str | None = None,
    role: str = "user",
    created_at: str = "2023-05-30T20:00:00-07:00",
    projection: bool = False,
) -> dict[str, Any]:
    source = source_id or f"source-{chunk_id}"
    route = "activated_assertion_projection" if projection else "bm25"
    rendered = f"[{created_at} | {role}] {text}"
    row: dict[str, Any] = {
        "evidence_id": chunk_id,
        "chunk_id": chunk_id,
        "turn_id": f"turn-{chunk_id}",
        "source_id": source,
        "role": role,
        "created_at": created_at,
        "route": route,
        "score": 1.0 if projection else "1",
        "raw_text": text,
        "raw_text_sha256": quote_sha256(text),
        "rendered_text": rendered,
        "rendered_text_sha256": quote_sha256(rendered),
    }
    if projection:
        row.pop("turn_id")
        row.update(
            {
                "source_handle": "G000001",
                "created_at_semantics": "source_metadata_only_not_event_time",
                "assertion_fact_receipt_sha256": "c" * 64,
            }
        )
    return row


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _arm(
    question: str,
    selected: Sequence[Mapping[str, Any]],
    *,
    packed_count: int | None = None,
) -> dict[str, Any]:
    exact_selected = [copy.deepcopy(dict(row)) for row in selected]
    count = len(exact_selected) if packed_count is None else packed_count
    packed = copy.deepcopy(exact_selected[:count])
    rendered = [str(row["rendered_text"]) for row in packed]
    messages = build_qa_prompt(question, rendered)
    payload = _canonical_json_bytes({"messages": messages})
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    context_tokens = (
        0
        if not rendered
        else count_tokens(
            "\n".join(
                f"[{index}] {text}" for index, text in enumerate(rendered, 1)
            )
        )
    )
    selected_ids = [str(row["chunk_id"]) for row in exact_selected]
    return {
        "selected_evidence": exact_selected,
        "packed_evidence": packed,
        "selected_chunk_ids": selected_ids,
        "packed_chunk_ids": selected_ids[:count],
        "dropped_chunk_ids": selected_ids[count:],
        "context_token_proxy": context_tokens,
        "prompt_token_proxy": prompt_tokens,
        "prompt_workspace_token_proxy": (
            prompt_tokens + RESPONDER_OUTPUT_TOKEN_RESERVE
        ),
        "provider_messages": messages,
        "provider_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "provider_payload_utf8_bytes": len(payload),
        "raw_evidence_only": True,
    }


def _provider_packet(arm: Mapping[str, Any]) -> dict[str, Any]:
    packet_body = {
        "format": PROVIDER_PACKET_FORMAT,
        "route": "hybrid",
        "selected_count": len(arm["selected_chunk_ids"]),
        "selected_chunk_ids_sha256": identity_sha256(arm["selected_chunk_ids"]),
        "packed_count": len(arm["packed_chunk_ids"]),
        "packed_chunk_ids_sha256": identity_sha256(arm["packed_chunk_ids"]),
        "dropped_count": len(arm["dropped_chunk_ids"]),
        "dropped_chunk_ids_sha256": identity_sha256(arm["dropped_chunk_ids"]),
        "context_token_proxy": arm["context_token_proxy"],
        "prompt_token_proxy": arm["prompt_token_proxy"],
        "prompt_workspace_token_proxy": arm["prompt_workspace_token_proxy"],
        "provider_payload_sha256": arm["provider_payload_sha256"],
        "provider_payload_utf8_bytes": arm["provider_payload_utf8_bytes"],
        "raw_evidence_only": True,
        "parent_receipts": {
            "v2_selection_sha256": "1" * 64,
            "v2_question_sha256": "2" * 64,
            "v2_projection_packet_receipt_sha256": "3" * 64,
            "v7_fallback_reference_receipt_sha256": "4" * 64,
            "projection_prefix_receipt_sha256": "5" * 64,
            "hybrid_receipt_sha256": "6" * 64,
        },
    }
    return {
        **packet_body,
        "receipt_sha256": identity_sha256(packet_body),
    }


def _adapt(
    question: str,
    arm: Mapping[str, Any],
    *,
    output_token_reserve: int = 768,
    provider_packet: Mapping[str, Any] | None = None,
):
    return adapt_hot_v3_arm(
        question,
        arm,
        selection_sha256=SELECTION_SHA256,
        provider_packet=(
            _provider_packet(arm) if provider_packet is None else provider_packet
        ),
        output_token_reserve=output_token_reserve,
    )


def test_adapts_fixed_comparison_to_existing_operator_without_provider() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:33]\n"
        "Did I receive a higher percentage discount on my first order from "
        "HelloFresh, compared to my first UberEats order?"
    )
    bundle = _adapt(
        question,
        _arm(
            question,
            [
                _evidence(
                    "hello",
                    "I got a 40% discount on my first HelloFresh order.",
                ),
                _evidence(
                    "uber",
                    "Last week I got 20% off my first UberEats order.",
                ),
            ],
        ),
    )

    decision = execute_operator_first_numeric_policy(bundle.provider_input)
    local_decision = execute_operator_first_numeric_policy(
        bundle.local_inventory.operator_input()
    )

    assert decision.status is ExecutionStatus.SUPPORTED
    assert decision.prediction == "Yes"
    assert decision.used_handle_ids == ("H001", "H002")
    assert local_decision.status is ExecutionStatus.SUPPORTED
    assert local_decision.prediction == "Yes"
    assert bundle.evidence_packet.frontier.mode is FrontierMode.BOUNDED
    assert bundle.evidence_packet.frontier.closed is False
    assert bundle.audit.scalar_item_count == 2
    assert bundle.audit.provider_prompt_count == 0
    assert bundle.audit.gold_loaded is False


def test_only_packed_prefix_crosses_adapter_and_dropped_text_cannot_leak() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:33]\n"
        "What meal delivery discounts did I receive?"
    )
    dropped_text = "DROPPED-SENTINEL must not cross the typed boundary."
    arm = _arm(
        question,
        [
            _evidence("one", "I received 40% off HelloFresh.", source_id="shared"),
            _evidence(
                "two",
                "I received 20% off UberEats.",
                source_id="shared",
                projection=True,
            ),
            _evidence("three", dropped_text),
        ],
        packed_count=2,
    )

    bundle = _adapt(question, arm)
    packet_projection = json.dumps(bundle.evidence_packet.projection())
    provider_projection = json.dumps(dict(bundle.provider_input))

    assert dropped_text not in packet_projection
    assert dropped_text not in provider_projection
    assert tuple(row.handle_id for row in bundle.evidence_packet.handles) == (
        "H001",
        "H002",
    )
    assert tuple(
        row.source_group_handle for row in bundle.evidence_packet.local_bindings
    ) == ("G001", "G001")
    assert bundle.audit.selected_evidence_count == 3
    assert bundle.audit.packed_evidence_count == 2
    assert bundle.audit.dropped_evidence_count == 1
    assert bundle.audit.packed_raw_evidence_count == 1
    assert bundle.audit.packed_projection_evidence_count == 1
    assert bundle.audit.source_group_count == 1
    assert bundle.audit.citation_binding_count == 2
    assert bundle.audit.frontier_truncated is True
    assert bundle.evidence_packet.items[0].date is None
    assert bundle.evidence_packet.items[0].value_authority.value == "explicit"
    assert bundle.evidence_packet.items[1].date is None
    assert bundle.evidence_packet.items[1].value_authority.value == "explicit"
    local_items = bundle.local_inventory.operator_input()["typed_evidence"]["items"]
    assert [row["temporal_anchor"] for row in local_items] == [
        "2023-05-30T20:00:00-07:00",
        "2023-05-30T20:00:00-07:00",
    ]
    assert "temporal_anchor" not in json.dumps(dict(bundle.provider_input))
    assert (
        "date_basis=source_metadata_only_not_event_time"
        in str(bundle.evidence_packet.items[1].relation)
    )


def test_local_temporal_anchor_resolves_only_explicit_relative_language() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:51]\n"
        "How many plants did I acquire in the last month?"
    )
    source_clock = "2023-05-20T10:00:00-07:00"
    bundle = _adapt(
        question,
        _arm(
            question,
            [
                _evidence(
                    "peace",
                    "I bought a peace lily two weeks ago.",
                    created_at=source_clock,
                    projection=True,
                )
            ],
        ),
    )

    local = compile_operator_first_numeric_candidates(
        bundle.local_inventory.operator_input()
    )
    capped = compile_operator_first_numeric_candidates(bundle.provider_input)

    assert bundle.local_inventory.items[0].date is None
    assert local.candidate_atoms[0].event_date == "2023-05-06"
    assert capped.candidate_atoms[0].event_date == "2023-05-16"
    assert local.candidate_atoms[0].temporal_basis.value == "relative_event_time"


def test_bare_source_metadata_cannot_become_executable_event_time() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:51]\n"
        "How many plants did I acquire in April?"
    )
    bundle = _adapt(
        question,
        _arm(
            question,
            [
                _evidence(
                    "peace",
                    "I bought a peace lily.",
                    created_at="2023-04-12T10:00:00-07:00",
                    projection=True,
                )
            ],
        ),
    )

    compilation = compile_operator_first_numeric_candidates(
        bundle.local_inventory.operator_input()
    )

    assert bundle.local_inventory.items[0].date is None
    assert compilation.candidate_atoms == ()
    assert {row.reason for row in compilation.exclusions} == {
        "event_time_not_in_scope"
    }


def test_open_ended_count_cannot_invent_global_closure_from_hot_packet() -> None:
    question = (
        "[Question asked at 2023/03/20 (Mon) 23:57]\n"
        "How many bikes did I service in March?"
    )
    bundle = _adapt(
        question,
        _arm(
            question,
            [
                _evidence(
                    "road",
                    "I got my road bike serviced on March 10th.",
                    created_at="2023-03-10T11:00:00-07:00",
                ),
                _evidence(
                    "commuter",
                    "I got my commuter bike serviced on March 15th.",
                    created_at="2023-03-15T11:00:00-07:00",
                ),
            ],
        ),
    )

    decision = execute_operator_first_numeric_policy(bundle.provider_input)

    assert decision.status is ExecutionStatus.INSUFFICIENT
    assert decision.prediction == ""
    assert bundle.audit.frontier_mode is FrontierMode.BOUNDED
    assert bundle.audit.frontier_closed is False


def test_system_row_is_audited_locally_but_absent_from_both_execution_planes() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:33]\n"
        "What system notice was retained?"
    )
    bundle = _adapt(
        question,
        _arm(
            question,
            [
                _evidence(
                    "system-row",
                    "The memory import boundary was initialized.",
                    role="system",
                )
            ],
        ),
    )

    assert bundle.evidence_packet.items == ()
    assert bundle.evidence_packet.handles == ()
    assert bundle.evidence_packet.frontier.available_handle_ids == ()
    assert bundle.local_inventory.items == ()
    assert tuple(
        row.handle_id for row in bundle.local_inventory.handles
    ) == ("H001",)
    assert bundle.local_inventory.system_excluded_handle_ids == ("H001",)
    assert bundle.local_inventory.frontier.omitted_handle_ids == ("H001",)
    assert "H001" not in json.dumps(bundle.local_inventory.operator_input())
    assert "memory import boundary" not in json.dumps(
        bundle.local_inventory.operator_input()
    )
    assert "memory import boundary" not in json.dumps(dict(bundle.provider_input))
    closure = build_slot_closure(bundle.operator_spec, bundle.evidence_packet)
    execution = execute_typed_operator(bundle.operator_spec, bundle.evidence_packet)
    assert closure.usable_item_receipt_sha256s == ()
    assert closure.sufficient is False
    assert execution.used_item_receipt_sha256s == ()
    assert execution.used_handle_ids == ()
    assert bundle.audit.citation_binding_count == 1
    assert bundle.audit.system_excluded_handle_count == 1
    assert bundle.audit.local_item_count == 0


def test_hard_cap_salvage_is_visible_as_omitted_bounded_handles() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:33]\n"
        "What details did I mention?"
    )
    rows = [
        _evidence(
            f"chunk-{index}",
            (f"detail-{index} " + ("word " * 90)).strip(),
        )
        for index in range(8)
    ]

    bundle = _adapt(question, _arm(question, rows), output_token_reserve=7_000)

    assert bundle.audit.citation_binding_count == 8
    assert 0 < bundle.audit.omitted_handle_count < 8
    assert bundle.audit.represented_handle_count + bundle.audit.omitted_handle_count == 8
    assert bundle.audit.frontier_truncated is True
    assert bundle.evidence_packet.provider_payload_token_proxy + 7_000 <= 8_000
    assert len(bundle.local_inventory.items) == 8
    assert len(bundle.local_inventory.frontier.represented_handle_ids) == 8
    assert bundle.local_inventory.frontier.omitted_handle_ids == ()
    assert bundle.local_inventory.frontier.closed is False
    assert bundle.audit.provider_capacity_omitted_handle_count == (
        bundle.audit.omitted_handle_count
    )
    assert bundle.audit.local_operator_token_proxy == count_tokens(
        json.dumps(
            bundle.local_inventory.operator_input(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    assert bundle.local_inventory.operator_projection()["provider_use_forbidden"] is True


def test_local_operator_sees_tail_operand_omitted_from_capped_packet() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:33]\n"
        "Did I receive a higher percentage discount on my first order from "
        "HelloFresh, compared to my first UberEats order?"
    )
    filler = ("unrelated delivery conversation " * 45).strip()
    rows = [
        _evidence("hello", "I got a 40% discount on my first HelloFresh order."),
        *[
            _evidence(f"filler-{index}", filler)
            for index in range(5)
        ],
        _evidence(
            "uber",
            "I got 20% off my first UberEats order. " + filler,
        ),
    ]
    bundle = _adapt(
        question,
        _arm(question, rows),
        output_token_reserve=7_000,
    )

    capped = execute_operator_first_numeric_policy(bundle.provider_input)
    local = execute_operator_first_numeric_policy(
        bundle.local_inventory.operator_input()
    )

    assert "H007" in bundle.evidence_packet.frontier.omitted_handle_ids
    assert len(bundle.local_inventory.items) == len(rows)
    assert capped.status is not ExecutionStatus.SUPPORTED
    assert local.status is ExecutionStatus.SUPPORTED
    assert local.prediction == "Yes"
    assert local.used_handle_ids == ("H001", "H007")


def test_required_role_rows_remain_visible_but_are_not_operator_eligible() -> None:
    question = (
        "[Question asked at 2026-08-28]\n"
        "What did I buy when I visited the shop?"
    )
    bundle = _adapt(
        question,
        _arm(
            question,
            [
                _evidence(
                    "assistant",
                    "You bought a red bicycle when you visited the shop.",
                    role="assistant",
                ),
                _evidence(
                    "user",
                    "I bought a blue bicycle when I visited the shop.",
                ),
            ],
        ),
    )

    assert bundle.operator_spec.required_evidence_role == "user"
    assert bundle.local_inventory.role_filtered_handle_ids == ("H001",)
    assert bundle.local_inventory.items[0].included is False
    assert bundle.local_inventory.items[1].included is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda arm: arm.update({"reference_answer": "forbidden"}),
        lambda arm: arm.__setitem__("provider_payload_sha256", "0" * 64),
        lambda arm: arm["packed_evidence"][0].__setitem__(
            "raw_text_sha256", "0" * 64
        ),
        lambda arm: arm["packed_chunk_ids"].reverse(),
    ],
)
def test_rejects_schema_digest_and_prefix_tampering(mutation) -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:33]\n"
        "What did I choose?"
    )
    arm = _arm(
        question,
        [_evidence("one", "I chose tea."), _evidence("two", "I chose coffee.")],
    )
    mutation(arm)

    with pytest.raises(MatchedEvalContractError):
        _adapt(question, arm)


@pytest.mark.parametrize("reseal", [False, True])
def test_provider_packet_receipt_and_arm_binding_are_both_required(reseal: bool) -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:33]\n"
        "What did I choose?"
    )
    arm = _arm(question, [_evidence("one", "I chose tea.")])
    provider_packet = _provider_packet(arm)
    provider_packet["packed_count"] = 2
    if reseal:
        body = dict(provider_packet)
        body.pop("receipt_sha256")
        provider_packet["receipt_sha256"] = identity_sha256(body)

    with pytest.raises(MatchedEvalContractError):
        _adapt(question, arm, provider_packet=provider_packet)


def test_provider_packet_rejects_boolean_count_even_when_resealed() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:33]\n"
        "What did I choose?"
    )
    arm = _arm(question, [_evidence("one", "I chose tea.")])
    provider_packet = _provider_packet(arm)
    provider_packet["packed_count"] = True
    body = dict(provider_packet)
    body.pop("receipt_sha256")
    provider_packet["receipt_sha256"] = identity_sha256(body)

    with pytest.raises(MatchedEvalContractError):
        _adapt(question, arm, provider_packet=provider_packet)


def test_projection_and_receipts_are_deterministic() -> None:
    question = (
        "[Question asked at 2023/05/30 (Tue) 21:33]\n"
        "What did I choose?"
    )
    arm = _arm(question, [_evidence("one", "I chose tea.")])

    left = _adapt(question, copy.deepcopy(arm))
    right = _adapt(question, copy.deepcopy(arm))

    assert left.receipt_sha256 == right.receipt_sha256
    assert left.projection() == right.projection()
    assert left.audit.provider_input_sha256 == right.audit.provider_input_sha256
    projection = left.projection()
    assert projection["citation_binding_count"] == 1
    assert projection["source_group_count"] == 1
    assert projection["frontier_mode"] == "bounded"
    assert projection["frontier_closed"] is False
