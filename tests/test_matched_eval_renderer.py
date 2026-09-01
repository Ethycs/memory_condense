from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from tools.matched_eval.contracts import (
    EvidenceItem,
    FactItem,
    LinkItem,
    MatchedEvalContractError,
    MemoryPacket,
    ObservationDelta,
    StageDisposition,
    StageTrace,
    identity_sha256,
)
from tools.matched_eval.renderer import (
    RENDERER_ID,
    SLOT_HEADERS,
    SLOT_ORDER,
    SYSTEM_POLICY,
    MatchedRendererError,
    render_memory_packet,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _packet(**updates: object) -> MemoryPacket:
    values: dict[str, object] = {
        "question_id": "q-1",
        "question_sha256": SHA_A,
        "dated_question": "[2026-08-26]\nWhat color did I choose?",
        "dated_question_sha256": SHA_B,
        "stage_id": "S0",
        "protected_evidence": (
            EvidenceItem("e-root", "turn-1", "I chose blue.", 4),
        ),
    }
    values.update(updates)
    return MemoryPacket(**values)  # type: ignore[arg-type]


def test_present_slots_are_always_a_canonical_subsequence() -> None:
    prompt = render_memory_packet(
        _packet(
            facts=(FactItem("f-1", "The choice was blue.", ("e-root",), 5),),
            answer_operators=(("direct", "Return the selected color."),),
        )
    )

    assert tuple(slot.slot_id for slot in prompt.slots) == (
        "dated_question",
        "protected_raw_evidence",
        "cited_fact_representation",
        "answer_operator",
    )
    user = prompt.messages[-1]["content"]
    positions = [user.index(SLOT_HEADERS[name]) for name in SLOT_ORDER if name in {
        slot.slot_id for slot in prompt.slots
    }]
    assert positions == sorted(positions)
    assert SLOT_HEADERS["admitted_raw_additions"] not in user
    assert SLOT_HEADERS["link_guide"] not in user


def test_renderer_identity_and_nested_messages_are_stable_and_immutable() -> None:
    first = render_memory_packet(_packet())
    second = render_memory_packet(_packet())

    assert first == second
    assert first.renderer_id == RENDERER_ID
    assert first.prompt_id == second.prompt_id
    assert first.messages_sha256 == identity_sha256(
        [dict(message) for message in first.messages]
    )
    with pytest.raises(TypeError):
        first.messages[0]["content"] = "changed"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        first.messages_sha256 = SHA_A  # type: ignore[misc]


def test_s0_only_control_has_one_final_user_message_and_no_empty_slots() -> None:
    prompt = render_memory_packet(_packet())

    assert [message["role"] for message in prompt.messages] == ["system", "user"]
    assert prompt.messages[0]["content"] == SYSTEM_POLICY
    assert tuple(slot.slot_id for slot in prompt.slots) == (
        "dated_question",
        "protected_raw_evidence",
    )
    assert prompt.slot_item_counts == {
        "dated_question": 1,
        "protected_raw_evidence": 1,
    }
    assert all(value > 0 for value in prompt.slot_token_proxies.values())


def test_all_typed_slots_render_once_with_complete_token_accounting() -> None:
    packet = _packet(
        admitted_evidence=(
            EvidenceItem("e-added", "turn-2", "Later I confirmed blue.", 5),
        ),
        facts=(
            FactItem("f-1", "The selected color is blue.", ("e-root",), 6),
        ),
        links=(
            LinkItem(
                "l-1",
                "The later confirmation supports the original choice.",
                ("e-root", "e-added"),
                8,
            ),
        ),
        answer_operators=(("latest", "Prefer the latest explicit choice."),),
        applied_stage_ids=("S1", "EM", "CAV", "LATEST"),
        stage_id="LATEST",
    )
    prompt = render_memory_packet(packet)

    assert tuple(slot.slot_id for slot in prompt.slots) == SLOT_ORDER
    assert prompt.slot_item_counts == {name: 1 for name in SLOT_ORDER}
    assert prompt.total_prompt_token_proxy == count_chat_prompt_token_proxy(
        prompt.messages
    )
    assert prompt.user_message_token_proxy > sum(
        item.token_count
        for item in packet.protected_evidence + packet.admitted_evidence
    )
    assert prompt.projection()["messages_sha256"] == prompt.messages_sha256


def test_renderer_fails_closed_on_forbidden_gold_fields() -> None:
    with pytest.raises(MatchedEvalContractError, match="gold_answer"):
        render_memory_packet({"gold_answer": "blue"})


def test_observation_delta_is_never_rendered() -> None:
    observation = ObservationDelta(
        stage_id="OBSERVE",
        parent_stage_id="S0",
        trace=StageTrace(
            candidate_ids=("event-1",),
            selected_before_dedup_ids=("event-1",),
            admitted_ids=("event-1",),
            token_cap=0,
            tokens_used=0,
            disposition=StageDisposition.ADDED,
        ),
        receipt_sha256=SHA_A,
    )

    with pytest.raises(MatchedRendererError, match="never renderable"):
        render_memory_packet(observation)
