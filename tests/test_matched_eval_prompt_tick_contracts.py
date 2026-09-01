from __future__ import annotations

import hashlib
from dataclasses import fields, replace

import pytest

from memory_condense.domain._tokenizer import count_tokens
from tools.matched_eval.contracts import (
    ArtifactRef,
    EvidenceItem,
    FactItem,
    MatchedEvalContractError,
    MemoryPacket,
    StageDisposition,
    StageTrace,
    assert_gold_blind,
    identity_sha256,
)
from tools.matched_eval.prompt_tick_contracts import (
    AnswerOperatorSpec,
    AnswerReceipt,
    CallBudget,
    CitationMatch,
    CitationRef,
    EvidenceRecordRef,
    ExposureReceipt,
    FactUnionDelta,
    GroundedFact,
    LaneBudget,
    LanePreparationReceipt,
    LinkOverlay,
    ModelCallReceipt,
    ObservationReceipt,
    PacketNodeRef,
    PromptTickPacket,
    PromptTickPlan,
    PromptTickReceipt,
    RelationItem,
    TickMode,
    TickRenderReceipt,
    base_tick_packet_id,
    represented_tick_packet_id,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def _evidence(evidence_id: str, source_id: str, text: str) -> EvidenceItem:
    return EvidenceItem(evidence_id, source_id, text, count_tokens(text))


def _base() -> MemoryPacket:
    return MemoryPacket(
        question_id="q-1",
        question_sha256=SHA_A,
        dated_question="[2026-08-27] What color did I choose?",
        dated_question_sha256=SHA_B,
        stage_id="DIRECT",
        protected_evidence=(
            _evidence("e-direct", "turn-1", "I was choosing a color."),
        ),
        admitted_evidence=(
            _evidence("e-query", "turn-2", "Later I chose blue."),
        ),
        applied_stage_ids=("DIRECT",),
    )


def _call(*, prompt_id: str, tokens: int = 20, retained: int = 0) -> ModelCallReceipt:
    return ModelCallReceipt(
        model_id="local/terra",
        prompt_id=prompt_id,
        messages_sha256=SHA_C,
        prompt_token_proxy=tokens,
        output_token_reserve=16,
        context_token_cap=128,
        request_journal_sha256=SHA_D,
        response_journal_sha256=SHA_E,
        retained_transformer_token_state_bytes=retained,  # type: ignore[arg-type]
    )


def _lane(
    lane_id: str,
    evidence_id: str,
    fact_id: str,
    *,
    parent_id: str,
    fact_text: str = "The selected color was blue.",
    quote: str = "I chose blue in the end.",
    budget: LaneBudget | None = None,
) -> LanePreparationReceipt:
    lane_budget = budget or LaneBudget(lane_id, 32, CallBudget(128, 16, 1))
    source_text = "The discussion wandered. I chose blue in the end."
    source = EvidenceRecordRef(
        _evidence(evidence_id, f"history-{lane_id}", source_text),
        ArtifactRef(f"{lane_id}_history", SHA_F),
        SHA_C,
    )
    fact = GroundedFact(
        FactItem(fact_id, fact_text, (evidence_id,), count_tokens(fact_text)),
        lane_id,
        ("color",),
        (
            CitationRef(
                evidence_id,
                quote,
                CitationMatch.EXACT_CONTIGUOUS_SUBSTRING,
            ),
        ),
        SHA_D,
    )
    return LanePreparationReceipt(
        lane_id=lane_id,
        mechanism_id=f"{lane_id}_source_fact_map",
        snapshot_id=SHA_A,
        as_of_turn=12,
        parent_tick_packet_id=parent_id,
        obligation_ids=("color",),
        budget=lane_budget,
        source_trace=StageTrace(
            candidate_ids=(evidence_id,),
            selected_before_dedup_ids=(evidence_id,),
            admitted_ids=(evidence_id,),
            token_cap=0,
            disposition=StageDisposition.ADDED,
        ),
        fact_trace=StageTrace(
            candidate_ids=(fact_id,),
            selected_before_dedup_ids=(fact_id,),
            admitted_ids=(fact_id,),
            token_cap=lane_budget.final_content_token_cap,
            tokens_used=fact.fact.token_count,
            provider_prompt_count=1,
            disposition=StageDisposition.ADDED,
        ),
        evidence_catalog=(source,),
        fact_candidates=(fact,),
        model_calls=(_call(prompt_id=identity_sha256(lane_id)),),
    )


def _union(
    base: MemoryPacket,
) -> tuple[FactUnionDelta, LanePreparationReceipt, LanePreparationReceipt]:
    parent = base_tick_packet_id(base)
    partition = _lane("partition", "e-partition", "f-partition", parent_id=parent)
    guided = _lane("guided", "e-guided", "f-guided", parent_id=parent)
    kept = partition.fact_candidates[0]
    union = FactUnionDelta(
        stage_id="SOURCE_FACT_UNION",
        parent_tick_packet_id=parent,
        parent_fact_ids=(),
        lanes=(partition, guided),
        trace=StageTrace(
            candidate_ids=("f-partition", "f-guided"),
            selected_before_dedup_ids=("f-partition", "f-guided"),
            dedup_excluded_ids=("f-guided",),
            admitted_ids=("f-partition",),
            token_cap=64,
            tokens_used=kept.fact.token_count,
            disposition=StageDisposition.ADDED,
        ),
        dedup_alias_bindings=(("f-guided", "f-partition"),),
        facts=(kept,),
    )
    return union, partition, guided


def _packet() -> tuple[PromptTickPacket, LanePreparationReceipt, LanePreparationReceipt]:
    base = _base()
    union, partition, guided = _union(base)
    represented = represented_tick_packet_id(base_tick_packet_id(base), union)
    evidence_nodes = tuple(
        PacketNodeRef("evidence", row.evidence_id)
        for row in base.protected_evidence + base.admitted_evidence
    )
    fact_node = PacketNodeRef("fact", "f-partition")
    inputs = evidence_nodes + (fact_node,)
    relation_text = "The mapped fact resolves the direct color discussion."
    links = LinkOverlay(
        parent_tick_packet_id=represented,
        input_nodes=inputs,
        steered_node_order=(fact_node,) + evidence_nodes,
        relations=(
            RelationItem(
                "rel-color",
                relation_text,
                (evidence_nodes[1], fact_node),
                count_tokens(relation_text),
            ),
        ),
        token_cap=32,
        cav_state_receipt_sha256=SHA_D,
        steered_readout_receipt_sha256=SHA_E,
    )
    packet = PromptTickPacket(
        base,
        (union,),
        links,
        AnswerOperatorSpec(
            "direct_extract", "Return only the selected color.", 16
        ),
    )
    return packet, partition, guided


def _render(packet: PromptTickPacket, *, tokens: int = 80, cap: int = 96) -> TickRenderReceipt:
    prompt_id = identity_sha256({"packet": packet.packet_id, "tokens": tokens})
    return TickRenderReceipt(
        packet_id=packet.packet_id,
        renderer_id="prompt_tick_renderer_v1",
        prompt_id=prompt_id,
        messages_sha256=SHA_C,
        prompt_token_proxy=tokens,
        prompt_token_cap=cap,
        exposure=ExposureReceipt(
            prompt_id,
            tuple(row.evidence_id for row in packet.evidence),
            tuple(row.fact_id for row in packet.facts),
            tuple(row.relation_id for row in packet.link_overlay.relations),
            packet.link_overlay.steered_node_order,
        ),
    )


def _answer(packet: PromptTickPacket, render: TickRenderReceipt) -> AnswerReceipt:
    call = ModelCallReceipt(
        model_id="local/terra",
        prompt_id=render.prompt_id,
        messages_sha256=render.messages_sha256,
        prompt_token_proxy=render.prompt_token_proxy,
        output_token_reserve=16,
        context_token_cap=128,
        request_journal_sha256=SHA_D,
        response_journal_sha256=SHA_E,
    )
    return AnswerReceipt(
        snapshot_id=SHA_A,
        tick_packet_id=packet.packet_id,
        render=render,
        call=call,
        decision="answer",
        prediction="blue",
        prediction_sha256=hashlib.sha256(b"blue").hexdigest(),
        used_nodes=(PacketNodeRef("fact", "f-partition"),),
    )


def _plan(
    packet: PromptTickPacket,
    partition: LanePreparationReceipt,
    guided: LanePreparationReceipt,
    *,
    lane_order: tuple[str, ...] = ("partition", "guided"),
) -> PromptTickPlan:
    budget_by_id = {
        partition.lane_id: partition.budget,
        guided.lane_id: guided.budget,
    }
    return PromptTickPlan(
        plan_id="prompt-tick-test",
        mode=TickMode.EVALUATION_READ_ONLY,
        snapshot_id=SHA_A,
        question_id=packet.base_packet.question_id,
        question_sha256=packet.base_packet.question_sha256,
        dated_question_sha256=packet.base_packet.dated_question_sha256,
        sealed_parent_packet_id=packet.base_packet.packet_id,
        sealed_input_artifacts=(ArtifactRef("sealed_retrieval", SHA_F),),
        as_of_turn=12,
        question_plan_receipt_sha256=SHA_B,
        lane_budgets=tuple(budget_by_id[row] for row in lane_order),
        global_internal_call_cap=2,
        link_token_cap=32,
        answer_operator_token_cap=16,
        final_answer_budget=CallBudget(128, 16, 1),
        final_prompt_token_cap=96,
    )


def _complete_tick() -> PromptTickReceipt:
    packet, partition, guided = _packet()
    render = _render(packet)
    answer = _answer(packet, render)
    observation = ObservationReceipt(
        TickMode.EVALUATION_READ_ONLY,
        SHA_A,
        answer,
    )
    return PromptTickReceipt(
        _plan(packet, partition, guided), packet, answer, observation
    )


def test_v3_overlay_does_not_change_the_sealed_v2_packet_shape_or_identity() -> None:
    packet = _base()

    assert tuple(row.name for row in fields(MemoryPacket)) == (
        "question_id",
        "question_sha256",
        "dated_question",
        "dated_question_sha256",
        "stage_id",
        "protected_evidence",
        "admitted_evidence",
        "facts",
        "links",
        "answer_operators",
        "applied_stage_ids",
    )
    assert packet.packet_id == "099a61e9c86894f13b88637d4cf0d31a89c481157f717143fcb2cc6554c19284"
    assert base_tick_packet_id(packet) != packet.packet_id


def test_two_lanes_fan_out_then_post_map_dedup_into_one_linkable_fact() -> None:
    tick = _complete_tick()
    union = tick.packet.fact_unions[0]

    assert tuple(row.parent_tick_packet_id for row in union.lanes) == (
        union.parent_tick_packet_id,
        union.parent_tick_packet_id,
    )
    assert union.trace.selected_before_dedup_ids == (
        "f-partition",
        "f-guided",
    )
    assert union.trace.dedup_excluded_ids == ("f-guided",)
    assert union.dedup_alias_bindings == (("f-guided", "f-partition"),)
    assert tuple(row.fact_id for row in tick.packet.facts) == ("f-partition",)
    relation = tick.packet.link_overlay.relations[0]
    assert {row.kind for row in relation.nodes} == {"evidence", "fact"}
    assert tick.observation.disposition == "evaluation_no_op"
    assert tick.observation.child_snapshot_id is None
    assert_gold_blind(tick.projection())


def test_lane_cannot_borrow_another_lanes_unused_calls_or_token_envelope() -> None:
    parent = base_tick_packet_id(_base())
    no_calls = LaneBudget("partition", 32, CallBudget(128, 16, 0))
    with pytest.raises(MatchedEvalContractError, match="provider-call cap"):
        _lane(
            "partition",
            "e-partition",
            "f-partition",
            parent_id=parent,
            budget=no_calls,
        )

    wrong_envelope = LaneBudget("partition", 32, CallBudget(256, 16, 1))
    with pytest.raises(MatchedEvalContractError, match="different token envelope"):
        _lane(
            "partition",
            "e-partition",
            "f-partition",
            parent_id=parent,
            budget=wrong_envelope,
        )

    lane = _lane(
        "partition",
        "e-partition-owned",
        "f-partition-owned",
        parent_id=parent,
    )
    with pytest.raises(MatchedEvalContractError, match="provider-free"):
        replace(
            lane,
            source_trace=replace(lane.source_trace, provider_prompt_count=1),
        )
    with pytest.raises(MatchedEvalContractError, match="fact-map provider-call"):
        replace(
            lane,
            fact_trace=replace(lane.fact_trace, provider_prompt_count=0),
        )


def test_fact_union_rejects_dedup_that_did_not_follow_selection() -> None:
    base = _base()
    union, partition, guided = _union(base)
    invalid_trace = StageTrace(
        candidate_ids=("f-partition", "f-guided"),
        selected_before_dedup_ids=("f-partition", "f-guided"),
        dedup_excluded_ids=("f-guided",),
        admitted_ids=("f-partition",),
        token_cap=64,
        tokens_used=partition.fact_candidates[0].fact.token_count,
        disposition=StageDisposition.ADDED,
    )
    with pytest.raises(MatchedEvalContractError, match="exact aliases"):
        FactUnionDelta(
            union.stage_id,
            union.parent_tick_packet_id,
            (),
            (partition, guided),
            invalid_trace,
            (("f-partition", "f-partition"),),
            union.facts,
        )

    unequal_guided = _lane(
        "guided",
        "e-guided-unequal",
        "f-guided-unequal",
        parent_id=union.parent_tick_packet_id,
        fact_text="The selected color was green.",
    )
    with pytest.raises(MatchedEvalContractError, match="canonical fact key"):
        FactUnionDelta(
            union.stage_id,
            union.parent_tick_packet_id,
            (),
            (partition, unequal_guided),
            StageTrace(
                candidate_ids=("f-partition", "f-guided-unequal"),
                selected_before_dedup_ids=("f-partition", "f-guided-unequal"),
                dedup_excluded_ids=("f-guided-unequal",),
                admitted_ids=("f-partition",),
                token_cap=64,
                tokens_used=partition.fact_candidates[0].fact.token_count,
                disposition=StageDisposition.ADDED,
            ),
            (("f-guided-unequal", "f-partition"),),
            (partition.fact_candidates[0],),
        )


def test_fact_mapping_and_cav_fail_closed_on_unverified_or_dangling_sources() -> None:
    parent = base_tick_packet_id(_base())
    with pytest.raises(MatchedEvalContractError, match="unverified lane citation"):
        _lane(
            "partition",
            "e-partition",
            "f-partition",
            parent_id=parent,
            quote="green was selected",
        )

    valid_lane = _lane(
        "partition",
        "e-partition-hash",
        "f-partition-hash",
        parent_id=parent,
    )
    record = valid_lane.evidence_catalog[0]
    assert record.projection()["text_sha256"] == hashlib.sha256(
        record.evidence.text.encode("utf-8")
    ).hexdigest()

    packet, _partition, _guided = _packet()
    link = packet.link_overlay
    dangling = PacketNodeRef("fact", "missing-fact")
    text = "A dangling relation."
    with pytest.raises(MatchedEvalContractError, match="outside the frontier"):
        LinkOverlay(
            link.parent_tick_packet_id,
            link.input_nodes,
            link.steered_node_order,
            (RelationItem("dangling", text, (dangling,), count_tokens(text)),),
            link.token_cap,
            link.cav_state_receipt_sha256,
            link.steered_readout_receipt_sha256,
        )


def test_cav_must_consume_final_representation_and_steered_output_affects_packet() -> None:
    packet, _partition, _guided = _packet()
    link = packet.link_overlay

    with pytest.raises(MatchedEvalContractError, match="final represented packet"):
        PromptTickPacket(
            packet.base_packet,
            packet.fact_unions,
            LinkOverlay(
                SHA_F,
                link.input_nodes,
                link.steered_node_order,
                link.relations,
                link.token_cap,
                link.cav_state_receipt_sha256,
                link.steered_readout_receipt_sha256,
            ),
            packet.answer_operator,
        )

    reordered_link = LinkOverlay(
        link.parent_tick_packet_id,
        link.input_nodes,
        tuple(reversed(link.steered_node_order)),
        link.relations,
        link.token_cap,
        link.cav_state_receipt_sha256,
        SHA_F,
    )
    reordered = PromptTickPacket(
        packet.base_packet,
        packet.fact_unions,
        reordered_link,
        packet.answer_operator,
    )
    assert reordered.packet_id != packet.packet_id

    stale_prompt_id = identity_sha256("stale-cav-order")
    stale_render = TickRenderReceipt(
        packet.packet_id,
        "prompt_tick_renderer_v1",
        stale_prompt_id,
        SHA_C,
        80,
        96,
        ExposureReceipt(
            stale_prompt_id,
            tuple(row.evidence_id for row in packet.evidence),
            tuple(row.fact_id for row in packet.facts),
            tuple(row.relation_id for row in packet.link_overlay.relations),
            packet.link_overlay.input_nodes,
        ),
    )
    stale_answer = _answer(packet, stale_render)
    with pytest.raises(MatchedEvalContractError, match="CAV-steered node order"):
        PromptTickReceipt(
            _plan(packet, _partition, _guided),
            packet,
            stale_answer,
            ObservationReceipt(TickMode.EVALUATION_READ_ONLY, SHA_A, stale_answer),
        )


def test_hard_final_cap_and_zero_retained_model_or_cav_state() -> None:
    packet, _partition, _guided = _packet()
    with pytest.raises(MatchedEvalContractError, match="hard cap"):
        _render(packet, tokens=97, cap=96)
    with pytest.raises(MatchedEvalContractError, match="transformer token state"):
        _call(prompt_id=SHA_A, retained=1)
    with pytest.raises(MatchedEvalContractError, match="transformer token state"):
        _call(prompt_id=SHA_A, retained=False)  # type: ignore[arg-type]

    link = packet.link_overlay
    with pytest.raises(MatchedEvalContractError, match="latent tensor state"):
        LinkOverlay(
            link.parent_tick_packet_id,
            link.input_nodes,
            link.steered_node_order,
            link.relations,
            link.token_cap,
            link.cav_state_receipt_sha256,
            link.steered_readout_receipt_sha256,
            retained_latent_state_bytes=1,  # type: ignore[arg-type]
        )
    with pytest.raises(MatchedEvalContractError, match="owned token cap"):
        LinkOverlay(
            link.parent_tick_packet_id,
            link.input_nodes,
            link.steered_node_order,
            link.relations,
            0,
            link.cav_state_receipt_sha256,
            link.steered_readout_receipt_sha256,
        )
    with pytest.raises(MatchedEvalContractError, match="owned token cap"):
        AnswerOperatorSpec("too-large", "Return the answer.", 0)


def test_observation_requires_answer_and_evaluation_is_an_exact_idempotent_noop() -> None:
    packet, _partition, _guided = _packet()
    answer = _answer(packet, _render(packet))
    assert answer.prediction_sha256 == hashlib.sha256(b"blue").hexdigest()
    with pytest.raises(MatchedEvalContractError, match="prediction SHA-256"):
        replace(answer, prediction_sha256=identity_sha256("blue"))

    with pytest.raises(MatchedEvalContractError, match="completed answer"):
        ObservationReceipt(
            TickMode.EVALUATION_READ_ONLY,
            SHA_A,
            None,  # type: ignore[arg-type]
        )
    with pytest.raises(MatchedEvalContractError, match="exact no-op"):
        ObservationReceipt(
            TickMode.EVALUATION_READ_ONLY,
            SHA_A,
            answer,
            (ArtifactRef("transcript_delta", SHA_F),),
        )

    first = ObservationReceipt(TickMode.EVALUATION_READ_ONLY, SHA_A, answer)
    replay = ObservationReceipt(TickMode.EVALUATION_READ_ONLY, SHA_A, answer)
    assert first.receipt_sha256 == replay.receipt_sha256
    assert first.idempotency_key_sha256 == replay.idempotency_key_sha256
    assert first.persistent_delta_refs == ()


def test_tick_rejects_noncanonical_merge_order_and_runtime_surface_is_gold_blind() -> None:
    packet, partition, guided = _packet()
    render = _render(packet)
    answer = _answer(packet, render)
    observation = ObservationReceipt(TickMode.EVALUATION_READ_ONLY, SHA_A, answer)

    with pytest.raises(MatchedEvalContractError, match="canonical plan order"):
        PromptTickReceipt(
            _plan(packet, partition, guided, lane_order=("guided", "partition")),
            packet,
            answer,
            observation,
        )

    with pytest.raises(MatchedEvalContractError, match="snapshot and turn"):
        PromptTickReceipt(
            replace(_plan(packet, partition, guided), as_of_turn=13),
            packet,
            answer,
            observation,
        )

    foreign_answer = replace(answer, snapshot_id=SHA_B)
    foreign_observation = ObservationReceipt(
        TickMode.EVALUATION_READ_ONLY, SHA_B, foreign_answer
    )
    with pytest.raises(MatchedEvalContractError, match="snapshot and turn"):
        PromptTickReceipt(
            replace(_plan(packet, partition, guided), snapshot_id=SHA_B),
            packet,
            foreign_answer,
            foreign_observation,
        )

    tick = PromptTickReceipt(
        _plan(packet, partition, guided), packet, answer, observation
    )
    projection = tick.projection()
    assert set(projection) == {
        "answer_receipt_sha256",
        "final_packet_id",
        "format",
        "observation_receipt_sha256",
        "plan_receipt_sha256",
        "question_id",
        "snapshot_id",
    }
    assert_gold_blind(projection)
