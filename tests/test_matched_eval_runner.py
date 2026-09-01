from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from memory_condense.domain._tokenizer import count_tokens

from tools.matched_eval.contracts import (
    AnswerOperatorDelta,
    ArmPlan,
    ArtifactRef,
    EvaluationMemorySnapshot,
    EvidenceItem,
    FactItem,
    LinkingDelta,
    LinkItem,
    MatchedEvalContractError,
    MembershipDelta,
    MemoryPacket,
    ObservationDelta,
    PlanMode,
    RepresentationDelta,
    StageBudget,
    StageDisposition,
    StagePlan,
    StageTrace,
    assert_gold_blind,
)
from tools.matched_eval.runner import MatchedEvalRunner
from tools.matched_eval.renderer import (
    V4_RENDERER_ID,
    render_memory_packet,
    render_memory_packet_for_id,
)
from tools.matched_eval.ledger import runtime_entry_from_stage_run


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _snapshot(*, renderer_id: str = "matched_typed_slots_v2") -> EvaluationMemorySnapshot:
    return EvaluationMemorySnapshot(
        population_identity_sha256=SHA_A,
        question_order_sha256=SHA_B,
        source_artifacts=(ArtifactRef("retrieval", SHA_C),),
        renderer_id=renderer_id,
    )


def _root() -> MemoryPacket:
    return MemoryPacket(
        question_id="q001",
        question_sha256=SHA_A,
        dated_question="2026-08-26: Which evidence matters?",
        dated_question_sha256=SHA_B,
        stage_id="S0",
        protected_evidence=(EvidenceItem("e0", "turn-0", "protected root", 2),),
    )


def _stage(
    stage_id: str,
    parent_stage_id: str,
    mechanism_id: str,
    delta_kind: str,
    *,
    token_cap: int = 16,
    provider_prompt_cap: int = 0,
) -> StagePlan:
    return StagePlan(
        stage_id=stage_id,
        parent_stage_id=parent_stage_id,
        mechanism_id=mechanism_id,
        delta_kind=delta_kind,  # type: ignore[arg-type]
        budget=StageBudget(
            token_cap=token_cap,
            provider_prompt_cap=provider_prompt_cap,
        ),
    )


def _added_trace(
    *ids: str,
    token_cap: int = 16,
    tokens_used: int = 1,
    provider_prompt_count: int = 0,
    candidates: tuple[str, ...] | None = None,
    selected: tuple[str, ...] | None = None,
    excluded: tuple[str, ...] = (),
    not_admitted: tuple[str, ...] = (),
) -> StageTrace:
    return StageTrace(
        candidate_ids=candidates if candidates is not None else ids,
        selected_before_dedup_ids=selected if selected is not None else ids,
        dedup_excluded_ids=excluded,
        not_admitted_ids=not_admitted,
        admitted_ids=ids,
        token_cap=token_cap,
        tokens_used=tokens_used,
        provider_prompt_count=provider_prompt_count,
        disposition=StageDisposition.ADDED,
    )


class _Adapter:
    def __init__(
        self,
        mechanism_id: str,
        delta_kind: str,
        factory: Callable[..., Any],
    ) -> None:
        self.mechanism_id = mechanism_id
        self.delta_kind = delta_kind
        self.factory = factory
        self.packets: list[MemoryPacket] = []

    def propose(self, *, snapshot: Any, packet: MemoryPacket, stage: StagePlan) -> Any:
        self.packets.append(packet)
        return self.factory(snapshot=snapshot, packet=packet, stage=stage)


def _membership_factory(evidence_id: str) -> Callable[..., MembershipDelta]:
    def factory(*, stage: StagePlan, **_: Any) -> MembershipDelta:
        text = f"raw {evidence_id}"
        token_count = count_tokens(text)
        return MembershipDelta(
            stage_id=stage.stage_id,
            parent_stage_id=stage.parent_stage_id,
            trace=_added_trace(
                evidence_id,
                token_cap=stage.budget.token_cap,
                tokens_used=token_count,
            ),
            additions=(
                EvidenceItem(
                    evidence_id,
                    f"turn-{evidence_id}",
                    text,
                    token_count,
                ),
            ),
        )

    return factory


def test_isolated_mode_is_an_exact_star_over_the_root_packet() -> None:
    root = _root()
    first = _Adapter("first", "membership", _membership_factory("e1"))
    second = _Adapter("second", "membership", _membership_factory("e2"))
    plan = ArmPlan(
        plan_id="isolated-star",
        mode=PlanMode.ISOLATED,
        root_stage_id="S0",
        stages=(
            _stage("S1", "S0", "first", "membership"),
            _stage("S2", "S0", "second", "membership"),
        ),
        global_provider_prompt_cap=0,
    )

    result = MatchedEvalRunner({"first": first, "second": second}).run(
        snapshot=_snapshot(), root_packet=root, plan=plan
    )

    assert first.packets == [root]
    assert second.packets == [root]
    assert [row.evidence_id for row in result.packet_for("S1").admitted_evidence] == [
        "e1"
    ]
    assert [row.evidence_id for row in result.packet_for("S2").admitted_evidence] == [
        "e2"
    ]
    assert result.packet_for("S1").protected_evidence == root.protected_evidence
    assert result.packet_for("S2").protected_evidence == root.protected_evidence


def test_common_runner_stage_flattens_losslessly_into_runtime_ledger() -> None:
    adapter = _Adapter("membership", "membership", _membership_factory("e1"))
    plan = ArmPlan(
        plan_id="ledger-adapter",
        mode=PlanMode.ISOLATED,
        root_stage_id="S0",
        stages=(_stage("S1", "S0", "membership", "membership"),),
        global_provider_prompt_cap=0,
    )
    result = MatchedEvalRunner({"membership": adapter}).run(
        snapshot=_snapshot(), root_packet=_root(), plan=plan
    )

    entry = runtime_entry_from_stage_run(
        ordinal=0,
        arm_label="S0_PLUS_S1_V2",
        parent_arm_label="S0_V2",
        run=result,
        stage_id="S1",
    )

    stage = result.stage("S1")
    assert entry.stage_receipt_sha256 == stage.receipt.receipt_sha256
    assert entry.delta_sha256 == stage.receipt.delta_sha256
    assert entry.prompt_id == stage.rendered_prompt.prompt_id
    assert entry.prompt_token_proxy == stage.rendered_prompt.total_prompt_token_proxy
    assert entry.provider_prompt_reserved == stage.receipt.provider_prompt_reserved
    assert entry.packet_sha256 == stage.packet.packet_id


def test_cumulative_mode_nests_and_preserves_post_selection_dedup() -> None:
    root = _root()

    def membership(*, stage: StagePlan, **_: Any) -> MembershipDelta:
        return MembershipDelta(
            stage_id=stage.stage_id,
            parent_stage_id=stage.parent_stage_id,
            trace=_added_trace(
                "e1",
                token_cap=stage.budget.token_cap,
                tokens_used=3,
                provider_prompt_count=1,
                candidates=("e0", "e1"),
                selected=("e0", "e1"),
                excluded=("e0",),
            ),
            additions=(EvidenceItem("e1", "turn-1", "new raw evidence", 3),),
        )

    def representation(
        *, packet: MemoryPacket, stage: StagePlan, **_: Any
    ) -> RepresentationDelta:
        assert [row.evidence_id for row in packet.admitted_evidence] == ["e1"]
        return RepresentationDelta(
            stage_id=stage.stage_id,
            parent_stage_id=stage.parent_stage_id,
            trace=_added_trace(
                "e1",
                token_cap=stage.budget.token_cap,
                tokens_used=4,
                provider_prompt_count=1,
                candidates=("e0", "e1"),
                selected=("e0", "e1"),
                excluded=("e0",),
            ),
            dedup_against_evidence_ids=("e0",),
            bound_evidence_ids=("e1",),
            facts=(FactItem("f1", "fact from e1", ("e1",), 4),),
        )

    membership_adapter = _Adapter("membership", "membership", membership)
    representation_adapter = _Adapter(
        "representation", "representation", representation
    )
    plan = ArmPlan(
        plan_id="cumulative-chain",
        mode=PlanMode.CUMULATIVE,
        root_stage_id="S0",
        stages=(
            _stage(
                "S1",
                "S0",
                "membership",
                "membership",
                provider_prompt_cap=1,
            ),
            _stage(
                "EM",
                "S1",
                "representation",
                "representation",
                provider_prompt_cap=1,
            ),
        ),
        global_provider_prompt_cap=2,
    )

    result = MatchedEvalRunner(
        {"membership": membership_adapter, "representation": representation_adapter}
    ).run(snapshot=_snapshot(), root_packet=root, plan=plan)

    s1 = result.packet_for("S1")
    em = result.packet_for("EM")
    assert representation_adapter.packets == [s1]
    assert s1.protected_evidence == root.protected_evidence
    assert [row.evidence_id for row in em.admitted_evidence] == ["e1"]
    assert [row.fact_id for row in em.facts] == ["f1"]
    assert result.stage("S1").trace.selected_before_dedup_ids == ("e0", "e1")
    assert result.stage("S1").trace.dedup_excluded_ids == ("e0",)
    assert result.provider_prompt_reserved == 2
    assert result.provider_prompt_count == 2


def test_membership_alias_dedup_and_partial_admission_preserve_full_trace() -> None:
    root = _root()

    def membership(*, stage: StagePlan, **_: Any) -> MembershipDelta:
        text = "retained bridge evidence"
        return MembershipDelta(
            stage_id=stage.stage_id,
            parent_stage_id=stage.parent_stage_id,
            trace=_added_trace(
                "atom-keep",
                token_cap=stage.budget.token_cap,
                tokens_used=count_tokens(text),
                candidates=(
                    "e0",
                    "atom-duplicate-a",
                    "atom-duplicate-b",
                    "atom-drop",
                    "atom-keep",
                ),
                selected=(
                    "e0",
                    "atom-duplicate-a",
                    "atom-duplicate-b",
                    "atom-drop",
                    "atom-keep",
                ),
                excluded=("e0", "atom-duplicate-a", "atom-duplicate-b"),
                not_admitted=("atom-drop",),
            ),
            dedup_alias_bindings=(
                ("atom-duplicate-a", "e0"),
                ("atom-duplicate-b", "e0"),
            ),
            additions=(
                EvidenceItem("atom-keep", "turn-keep", text, count_tokens(text)),
            ),
        )

    plan = ArmPlan(
        "alias-partition",
        PlanMode.ISOLATED,
        "S0",
        (_stage("BRIDGE", "S0", "bridge", "membership"),),
        0,
    )
    result = MatchedEvalRunner(
        {"bridge": _Adapter("bridge", "membership", membership)}
    ).run(snapshot=_snapshot(), root_packet=root, plan=plan)

    stage = result.stage("BRIDGE")
    assert stage.trace.disposition is StageDisposition.ADDED
    assert stage.trace.dedup_excluded_ids == (
        "e0",
        "atom-duplicate-a",
        "atom-duplicate-b",
    )
    assert stage.trace.not_admitted_ids == ("atom-drop",)
    assert stage.trace.admitted_ids == ("atom-keep",)
    assert [row.evidence_id for row in stage.packet.admitted_evidence] == [
        "atom-keep"
    ]


def test_membership_alias_must_resolve_to_protected_parent_evidence() -> None:
    def invalid(*, stage: StagePlan, **_: Any) -> MembershipDelta:
        return MembershipDelta(
            stage_id=stage.stage_id,
            parent_stage_id=stage.parent_stage_id,
            trace=StageTrace(
                candidate_ids=("atom-duplicate",),
                selected_before_dedup_ids=("atom-duplicate",),
                dedup_excluded_ids=("atom-duplicate",),
                token_cap=stage.budget.token_cap,
                disposition=StageDisposition.NO_OP,
            ),
            dedup_alias_bindings=(("atom-duplicate", "missing-parent"),),
        )

    plan = ArmPlan(
        "bad-alias",
        PlanMode.ISOLATED,
        "S0",
        (_stage("BRIDGE", "S0", "bridge", "membership"),),
        0,
    )
    result = MatchedEvalRunner(
        {"bridge": _Adapter("bridge", "membership", invalid)}
    ).run(snapshot=_snapshot(), root_packet=_root(), plan=plan)

    assert result.stage("BRIDGE").trace.disposition is StageDisposition.INVALID
    assert result.stage("BRIDGE").trace.reason == "packet_invariant"


def test_runner_dispatches_the_snapshot_renderer_for_root_and_stages() -> None:
    snapshot = _snapshot(renderer_id=V4_RENDERER_ID)
    root = _root()
    plan = ArmPlan(
        "renderer-v4",
        PlanMode.ISOLATED,
        "S0",
        (_stage("S1", "S0", "membership", "membership"),),
        0,
    )
    result = MatchedEvalRunner(
        {
            "membership": _Adapter(
                "membership", "membership", _membership_factory("e1")
            )
        }
    ).run(snapshot=snapshot, root_packet=root, plan=plan)

    assert result.root_prompt == render_memory_packet_for_id(
        root,
        renderer_id=V4_RENDERER_ID,
    )
    assert result.root_prompt.renderer_id == V4_RENDERER_ID
    assert result.stage("S1").rendered_prompt.renderer_id == V4_RENDERER_ID


def test_all_delta_kinds_update_only_their_typed_packet_slot() -> None:
    root = _root()

    def produce(*, packet: MemoryPacket, stage: StagePlan, **_: Any) -> Any:
        if stage.delta_kind == "membership":
            return MembershipDelta(
                stage.stage_id,
                stage.parent_stage_id,
                _added_trace("e1"),
                additions=(EvidenceItem("e1", "turn-1", "raw", 1),),
            )
        if stage.delta_kind == "representation":
            return RepresentationDelta(
                stage.stage_id,
                stage.parent_stage_id,
                _added_trace("em-raw-1"),
                # EM represents a selected raw row without rendering that row
                # into packet membership.
                bound_evidence_ids=("em-raw-1",),
                facts=(FactItem("f1", "fact", ("em-raw-1",), 1),),
            )
        if stage.delta_kind == "linking":
            return LinkingDelta(
                stage.stage_id,
                stage.parent_stage_id,
                _added_trace("l1", tokens_used=5),
                bound_evidence_ids=("e0", "e1"),
                links=(LinkItem("l1", "e0 links e1", ("e0", "e1"), 5),),
            )
        if stage.delta_kind == "answer_operator":
            return AnswerOperatorDelta(
                stage.stage_id,
                stage.parent_stage_id,
                _added_trace("op1", tokens_used=5),
                operator_id="op1",
                instructions="Use the typed evidence.",
            )
        assert stage.delta_kind == "observation"
        return ObservationDelta(
            stage.stage_id,
            stage.parent_stage_id,
            _added_trace("observation-1", tokens_used=0),
            receipt_sha256=SHA_C,
        )

    kinds = (
        ("S1", "S0", "membership", "membership"),
        ("EM", "S1", "representation", "representation"),
        ("CAV", "EM", "linking", "linking"),
        ("OP", "CAV", "operator", "answer_operator"),
        ("OBS", "OP", "observer", "observation"),
    )
    adapters = {
        mechanism_id: _Adapter(mechanism_id, kind, produce)
        for _, _, mechanism_id, kind in kinds
    }
    plan = ArmPlan(
        plan_id="all-deltas",
        mode=PlanMode.CUMULATIVE,
        root_stage_id="S0",
        stages=tuple(_stage(*row) for row in kinds),
        global_provider_prompt_cap=0,
    )

    result = MatchedEvalRunner(adapters).run(
        snapshot=_snapshot(), root_packet=root, plan=plan
    )

    before_observation = result.packet_for("OP")
    final = result.packet_for("OBS")
    assert [row.evidence_id for row in final.admitted_evidence] == ["e1"]
    assert [row.fact_id for row in final.facts] == ["f1"]
    assert [row.link_id for row in final.links] == ["l1"]
    assert final.answer_operators == (("op1", "Use the typed evidence."),)
    assert final.protected_evidence == root.protected_evidence
    assert (
        final.protected_evidence,
        final.admitted_evidence,
        final.facts,
        final.links,
        final.answer_operators,
    ) == (
        before_observation.protected_evidence,
        before_observation.admitted_evidence,
        before_observation.facts,
        before_observation.links,
        before_observation.answer_operators,
    )
    assert final.applied_stage_ids == ("S1", "EM", "CAV", "OP", "OBS")


@pytest.mark.parametrize(
    ("returned_stage", "returned_parent", "reason"),
    (("WRONG", "S0", "delta_stage_mismatch"), ("S1", "WRONG", "delta_parent_mismatch")),
)
def test_wrong_delta_stage_or_parent_is_an_exact_parent_no_op(
    returned_stage: str, returned_parent: str, reason: str
) -> None:
    root = _root()

    def invalid(**_: Any) -> MembershipDelta:
        return MembershipDelta(
            returned_stage,
            returned_parent,
            _added_trace("e1"),
            additions=(EvidenceItem("e1", "turn-1", "raw", 1),),
        )

    adapter = _Adapter("membership", "membership", invalid)
    plan = ArmPlan(
        "bad-binding",
        PlanMode.ISOLATED,
        "S0",
        (_stage("S1", "S0", "membership", "membership"),),
        0,
    )

    result = MatchedEvalRunner({"membership": adapter}).run(
        snapshot=_snapshot(), root_packet=root, plan=plan
    )

    failed = result.stage("S1")
    assert failed.packet is root
    assert failed.trace.disposition is StageDisposition.INVALID
    assert failed.trace.reason == reason


def test_declared_or_returned_delta_kind_mismatch_fails_closed() -> None:
    root = _root()
    uncalled = _Adapter("declared", "membership", _membership_factory("e1"))
    wrong_return = _Adapter("returned", "representation", _membership_factory("e2"))
    plan = ArmPlan(
        "kind-mismatch",
        PlanMode.ISOLATED,
        "S0",
        (
            _stage("A", "S0", "declared", "representation"),
            _stage("B", "S0", "returned", "representation"),
        ),
        0,
    )

    result = MatchedEvalRunner({"declared": uncalled, "returned": wrong_return}).run(
        snapshot=_snapshot(), root_packet=root, plan=plan
    )

    assert uncalled.packets == []
    assert wrong_return.packets == [root]
    assert result.stage("A").trace.reason == "adapter_delta_kind_mismatch"
    assert result.stage("B").trace.reason == "delta_kind_mismatch"
    assert all(row.packet is root for row in result.stages)


def test_exception_duplicate_and_stage_overflow_preserve_exact_parent() -> None:
    root = _root()

    def explode(**_: Any) -> Any:
        raise RuntimeError("reference_answer=must-not-enter-receipt")

    def duplicate(*, stage: StagePlan, **_: Any) -> MembershipDelta:
        return MembershipDelta(
            stage.stage_id,
            stage.parent_stage_id,
            _added_trace("e0", token_cap=stage.budget.token_cap),
            additions=(EvidenceItem("e0", "turn-other", "duplicate", 1),),
        )

    def token_overflow(*, stage: StagePlan, **_: Any) -> MembershipDelta:
        return MembershipDelta(
            stage.stage_id,
            stage.parent_stage_id,
            _added_trace("e2", token_cap=9, tokens_used=5),
            additions=(EvidenceItem("e2", "turn-2", "too large", 2),),
        )

    def provider_overflow(*, stage: StagePlan, **_: Any) -> MembershipDelta:
        return MembershipDelta(
            stage.stage_id,
            stage.parent_stage_id,
            _added_trace(
                "e3",
                token_cap=stage.budget.token_cap,
                tokens_used=3,
                provider_prompt_count=2,
            ),
            additions=(EvidenceItem("e3", "turn-3", "too many calls", 3),),
        )

    adapters = {
        "throws": _Adapter("throws", "membership", explode),
        "duplicate": _Adapter("duplicate", "membership", duplicate),
        "token": _Adapter("token", "membership", token_overflow),
        "provider": _Adapter("provider", "membership", provider_overflow),
    }
    plan = ArmPlan(
        "fail-closed",
        PlanMode.ISOLATED,
        "S0",
        (
            _stage("THROW", "S0", "throws", "membership", token_cap=4),
            _stage("DUP", "S0", "duplicate", "membership", token_cap=4),
            _stage("TOK", "S0", "token", "membership", token_cap=4),
            _stage(
                "CALL",
                "S0",
                "provider",
                "membership",
                token_cap=4,
                provider_prompt_cap=1,
            ),
        ),
        global_provider_prompt_cap=1,
    )

    result = MatchedEvalRunner(adapters).run(
        snapshot=_snapshot(), root_packet=root, plan=plan
    )

    assert all(row.packet is root for row in result.stages)
    assert [row.trace.disposition for row in result.stages] == [
        StageDisposition.FAILED,
        StageDisposition.INVALID,
        StageDisposition.OVERFLOW,
        StageDisposition.OVERFLOW,
    ]
    assert [row.trace.reason for row in result.stages] == [
        "adapter_exception",
        "packet_invariant",
        "stage_token_cap",
        "stage_provider_prompt_cap",
    ]
    assert result.stage("TOK").receipt.reported_tokens_used == 5
    assert result.stage("TOK").trace.tokens_used == 4
    assert result.stage("TOK").trace.selected_before_dedup_ids == ("e2",)
    assert result.stage("TOK").trace.not_admitted_ids == ("e2",)
    assert result.provider_prompt_reserved == 1
    assert result.provider_prompt_count == 2
    assert result.provider_prompt_cap_compliant is False
    assert "must-not-enter-receipt" not in str(result.projection())
    for row in result.stages:
        assert_gold_blind(row.receipt.projection())


def test_global_ceiling_reserves_full_stage_caps_before_adapter_mutation() -> None:
    root = _root()
    first = _Adapter("first", "membership", _membership_factory("e1"))
    second = _Adapter("second", "membership", _membership_factory("e2"))
    plan = ArmPlan(
        "global-cap",
        PlanMode.ISOLATED,
        "S0",
        (
            _stage(
                "S1",
                "S0",
                "first",
                "membership",
                provider_prompt_cap=1,
            ),
            _stage(
                "S2",
                "S0",
                "second",
                "membership",
                provider_prompt_cap=1,
            ),
        ),
        global_provider_prompt_cap=1,
    )

    result = MatchedEvalRunner({"first": first, "second": second}).run(
        snapshot=_snapshot(), root_packet=root, plan=plan
    )

    assert first.packets == [root]
    assert second.packets == []
    assert result.stage("S1").trace.disposition is StageDisposition.ADDED
    assert result.stage("S2").trace.disposition is StageDisposition.OVERFLOW
    assert result.stage("S2").trace.reason == "global_provider_prompt_cap"
    assert result.stage("S2").packet is root
    assert result.provider_prompt_reserved == 1
    assert result.provider_prompt_count == 0


def test_provider_prompt_compliance_is_per_stage_and_cannot_borrow() -> None:
    root = _root()

    def produce(*, stage: StagePlan, **_: Any) -> MembershipDelta:
        evidence_id = stage.stage_id.casefold()
        return MembershipDelta(
            stage.stage_id,
            stage.parent_stage_id,
            _added_trace(
                evidence_id,
                token_cap=stage.budget.token_cap,
                provider_prompt_count=2 if stage.stage_id == "A" else 0,
            ),
            additions=(EvidenceItem(evidence_id, evidence_id, "raw", 1),),
        )

    adapter = _Adapter("membership", "membership", produce)
    plan = ArmPlan(
        "no-borrowing",
        PlanMode.ISOLATED,
        "S0",
        (
            _stage("A", "S0", "membership", "membership", provider_prompt_cap=1),
            _stage("B", "S0", "membership", "membership", provider_prompt_cap=1),
        ),
        2,
    )

    result = MatchedEvalRunner({"membership": adapter}).run(
        snapshot=_snapshot(), root_packet=root, plan=plan
    )

    assert result.provider_prompt_count == result.provider_prompt_reserved == 2
    assert result.stage("A").trace.disposition is StageDisposition.OVERFLOW
    assert result.stage("B").trace.disposition is StageDisposition.ADDED
    assert result.stage("A").receipt.provider_prompt_cap_compliant is False
    assert result.provider_prompt_cap_compliant is False


def test_measured_delta_tokens_and_final_renderer_cap_fail_closed() -> None:
    root = _root()
    root_tokens = render_memory_packet(root).total_prompt_token_proxy

    def underreported(*, stage: StagePlan, **_: Any) -> MembershipDelta:
        return MembershipDelta(
            stage.stage_id,
            stage.parent_stage_id,
            _added_trace("large", token_cap=stage.budget.token_cap, tokens_used=1),
            additions=(
                EvidenceItem("large", "turn-large", "three actual tokens", 3),
            ),
        )

    measured_plan = ArmPlan(
        "measured-token-cap",
        PlanMode.ISOLATED,
        "S0",
        (_stage("S1", "S0", "membership", "membership", token_cap=2),),
        0,
    )
    measured = MatchedEvalRunner(
        {"membership": _Adapter("membership", "membership", underreported)}
    ).run(snapshot=_snapshot(), root_packet=root, plan=measured_plan)
    assert measured.stage("S1").packet is root
    assert measured.stage("S1").trace.reason == "stage_token_cap"

    final_plan = ArmPlan(
        "final-render-cap",
        PlanMode.ISOLATED,
        "S0",
        (_stage("S1", "S0", "membership", "membership", token_cap=3),),
        0,
        max_final_prompt_tokens=root_tokens + 1,
    )
    rendered = MatchedEvalRunner(
        {"membership": _Adapter("membership", "membership", underreported)}
    ).run(snapshot=_snapshot(), root_packet=root, plan=final_plan)
    assert rendered.stage("S1").packet is root
    assert rendered.stage("S1").trace.reason == "stage_token_accounting_mismatch"

    def exact(*, stage: StagePlan, **_: Any) -> MembershipDelta:
        return MembershipDelta(
            stage.stage_id,
            stage.parent_stage_id,
            _added_trace("large", token_cap=stage.budget.token_cap, tokens_used=3),
            additions=(
                EvidenceItem("large", "turn-large", "three actual tokens", 3),
            ),
        )

    rendered = MatchedEvalRunner(
        {"membership": _Adapter("membership", "membership", exact)}
    ).run(snapshot=_snapshot(), root_packet=root, plan=final_plan)
    assert rendered.stage("S1").packet is root
    assert rendered.stage("S1").trace.reason == "final_prompt_token_cap"
    assert rendered.stage("S1").rendered_prompt == rendered.root_prompt

    with pytest.raises(MatchedEvalContractError, match="root packet exceeds"):
        MatchedEvalRunner({}).run(
            snapshot=_snapshot(),
            root_packet=root,
            plan=ArmPlan("root-overflow", PlanMode.ISOLATED, "S0", (), 0, 1),
        )


def test_cumulative_chain_continues_from_the_exact_no_op_parent_packet() -> None:
    root = _root()

    def explode(**_: Any) -> Any:
        raise RuntimeError("boom")

    failed = _Adapter("failed", "membership", explode)
    recovery = _Adapter("recovery", "membership", _membership_factory("e2"))
    plan = ArmPlan(
        "cumulative-no-op",
        PlanMode.CUMULATIVE,
        "S0",
        (
            _stage("S1", "S0", "failed", "membership"),
            _stage("S2", "S1", "recovery", "membership"),
        ),
        0,
    )

    result = MatchedEvalRunner({"failed": failed, "recovery": recovery}).run(
        snapshot=_snapshot(), root_packet=root, plan=plan
    )

    assert result.stage("S1").packet is root
    assert recovery.packets == [root]
    assert [
        row.evidence_id for row in result.stage("S2").packet.admitted_evidence
    ] == ["e2"]
    assert result.stage("S2").receipt.parent_stage_id == "S1"
    assert result.stage("S2").receipt.parent_packet_id == root.packet_id
    assert result.stage("S1").receipt.stage_id == "S1"
    assert result.stage("S1").packet.stage_id == "S0"


def test_arm_result_rejects_cross_snapshot_or_broken_logical_lineage() -> None:
    plan = ArmPlan(
        "bound-result",
        PlanMode.ISOLATED,
        "S0",
        (_stage("S1", "S0", "membership", "membership"),),
        0,
    )
    result = MatchedEvalRunner(
        {"membership": _Adapter("membership", "membership", _membership_factory("e1"))}
    ).run(snapshot=_snapshot(), root_packet=_root(), plan=plan)
    stage = result.stage("S1")

    wrong_snapshot = replace(stage.receipt, snapshot_id=SHA_B)
    with pytest.raises(MatchedEvalContractError, match="snapshot, plan, or question"):
        replace(result, stages=(replace(stage, receipt=wrong_snapshot),))

    wrong_parent = replace(stage.receipt, parent_stage_id="OTHER")
    with pytest.raises(MatchedEvalContractError, match="lineage"):
        replace(result, stages=(replace(stage, receipt=wrong_parent),))


def test_empty_exact_evidence_can_be_admitted_without_token_charge() -> None:
    def empty(*, stage: StagePlan, **_: Any) -> MembershipDelta:
        return MembershipDelta(
            stage.stage_id,
            stage.parent_stage_id,
            _added_trace("empty", token_cap=0, tokens_used=0),
            additions=(EvidenceItem("empty", "turn-empty", "", 0),),
        )

    plan = ArmPlan(
        "empty-evidence",
        PlanMode.ISOLATED,
        "S0",
        (_stage("S1", "S0", "empty", "membership", token_cap=0),),
        0,
    )
    result = MatchedEvalRunner(
        {"empty": _Adapter("empty", "membership", empty)}
    ).run(snapshot=_snapshot(), root_packet=_root(), plan=plan)
    assert result.stage("S1").trace.disposition is StageDisposition.ADDED
    assert result.packet_for("S1").admitted_evidence[0].text == ""


def test_registry_and_results_are_strict_and_immutable() -> None:
    adapter = _Adapter("actual", "membership", _membership_factory("e1"))
    with pytest.raises(MatchedEvalContractError, match="registry key"):
        MatchedEvalRunner({"alias": adapter})

    plan = ArmPlan("empty", PlanMode.ISOLATED, "S0", (), 0)
    result = MatchedEvalRunner({}).run(
        snapshot=_snapshot(), root_packet=_root(), plan=plan
    )
    with pytest.raises(FrozenInstanceError):
        result.provider_prompt_count = 1  # type: ignore[misc]
    assert result.result_sha256 == result.result_sha256
