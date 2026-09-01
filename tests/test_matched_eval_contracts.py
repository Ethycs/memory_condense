from __future__ import annotations

from dataclasses import asdict

import pytest

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
    PlanMode,
    RepresentationDelta,
    StageBudget,
    StageDisposition,
    StagePlan,
    StageTrace,
    assert_gold_blind,
    delta_projection,
    identity_sha256,
)
from memory_condense.domain.discourse import identity_sha256 as legacy_identity_sha256


SHA_A = "a" * 64
SHA_B = "b" * 64


def _trace(*ids: str, token_cap: int = 32, tokens_used: int = 4) -> StageTrace:
    return StageTrace(
        candidate_ids=ids,
        selected_before_dedup_ids=ids,
        admitted_ids=ids,
        token_cap=token_cap,
        tokens_used=tokens_used,
        disposition=StageDisposition.ADDED,
    )


def test_snapshot_is_a_stable_read_only_vector() -> None:
    snapshot = EvaluationMemorySnapshot(
        population_identity_sha256=SHA_A,
        question_order_sha256=SHA_B,
        source_artifacts=(ArtifactRef("retrieval", SHA_A, "sealed/retrieval.json"),),
        overlay_revisions=(ArtifactRef("causal_overlay", SHA_B),),
        model_ids=("codex_sdk/gpt-5.6-terra",),
    )

    assert snapshot.snapshot_id == snapshot.snapshot_id
    assert snapshot.projection()["reheat_memories"] is False
    assert snapshot.projection()["learn_consolidation"] is False

    with pytest.raises(MatchedEvalContractError, match="disable"):
        EvaluationMemorySnapshot(
            population_identity_sha256=SHA_A,
            question_order_sha256=SHA_B,
            source_artifacts=(ArtifactRef("retrieval", SHA_A),),
            reheat_memories=True,
        )


def test_behavior_identity_matches_the_existing_runtime_convention() -> None:
    value = {"messages": [{"role": "user", "content": "hello"}]}
    assert identity_sha256(value) == legacy_identity_sha256(value)


def test_runtime_gold_firewall_is_recursive() -> None:
    assert_gold_blind({"questions": [{"prediction_sha256": SHA_A}]})
    with pytest.raises(MatchedEvalContractError, match="reference_answer"):
        assert_gold_blind({"questions": [{"reference_answer": "hidden"}]})
    for key in ("ground_truth", "desired_answer", "gold_answer_sha256"):
        with pytest.raises(MatchedEvalContractError, match=key):
            assert_gold_blind({"questions": [{key: "hidden"}]})
    with pytest.raises(MatchedEvalContractError, match="must be false"):
        assert_gold_blind({"gold_loaded": True})


def test_delta_kinds_preserve_distinct_semantics() -> None:
    evidence = EvidenceItem("e1", "turn-1", "raw evidence", 2)
    membership = MembershipDelta(
        stage_id="S1",
        parent_stage_id="S0",
        trace=_trace("e1"),
        additions=(evidence,),
    )
    fact = FactItem("f1", "compressed fact", ("e1",), 2)
    representation = RepresentationDelta(
        stage_id="EM",
        parent_stage_id="S0",
        trace=_trace("e1"),
        bound_evidence_ids=("e1",),
        facts=(fact,),
    )
    link = LinkItem("l1", "e1 precedes another event", ("e1",), 6)
    linking = LinkingDelta(
        stage_id="CAV",
        parent_stage_id="S0",
        trace=_trace("l1"),
        bound_evidence_ids=("e1",),
        links=(link,),
    )
    operator = AnswerOperatorDelta(
        stage_id="NUMERIC",
        parent_stage_id="S0",
        trace=_trace("sum", tokens_used=1),
        operator_id="sum",
        instructions="Add only explicitly supported values.",
    )

    assert membership.kind == "membership"
    assert representation.kind == "representation"
    assert linking.kind == "linking"
    assert linking.evidence_additions == ()
    assert operator.kind == "answer_operator"
    assert "kind" in asdict(representation)


def test_post_selection_dedup_contract_excludes_only_after_selection() -> None:
    trace = StageTrace(
        candidate_ids=("e1", "e2", "e3"),
        selected_before_dedup_ids=("e1", "e2"),
        dedup_excluded_ids=("e1",),
        admitted_ids=("e2",),
        token_cap=10,
        tokens_used=3,
        disposition=StageDisposition.ADDED,
    )
    assert trace.selected_before_dedup_ids == ("e1", "e2")
    assert trace.dedup_excluded_ids == ("e1",)

    with pytest.raises(MatchedEvalContractError, match="selection"):
        StageTrace(
            candidate_ids=("e1",),
            selected_before_dedup_ids=("e1",),
            dedup_excluded_ids=("e2",),
            token_cap=10,
        )

    with pytest.raises(MatchedEvalContractError, match="partition exactly"):
        StageTrace(
            candidate_ids=("e1", "e2"),
            selected_before_dedup_ids=("e1", "e2"),
            admitted_ids=("e2",),
            token_cap=10,
            disposition=StageDisposition.ADDED,
        )

    terminal = StageTrace(
        candidate_ids=("e1", "e2"),
        selected_before_dedup_ids=("e1", "e2"),
        dedup_excluded_ids=("e1",),
        not_admitted_ids=("e2",),
        token_cap=10,
        disposition=StageDisposition.OVERFLOW,
        reason="final_prompt_token_cap",
    )
    assert terminal.not_admitted_ids == ("e2",)


def test_added_membership_preserves_partial_admission_and_dedup_aliases() -> None:
    trace = StageTrace(
        candidate_ids=("atom-duplicate", "atom-drop", "atom-keep"),
        selected_before_dedup_ids=("atom-duplicate", "atom-drop", "atom-keep"),
        dedup_excluded_ids=("atom-duplicate",),
        not_admitted_ids=("atom-drop",),
        admitted_ids=("atom-keep",),
        token_cap=16,
        tokens_used=2,
        disposition=StageDisposition.ADDED,
    )
    delta = MembershipDelta(
        stage_id="BRIDGE",
        parent_stage_id="S0",
        trace=trace,
        dedup_alias_bindings=(("atom-duplicate", "s0-evidence"),),
        additions=(EvidenceItem("atom-keep", "turn-keep", "kept evidence", 2),),
    )

    assert delta.trace.not_admitted_ids == ("atom-drop",)
    assert delta.dedup_alias_bindings == (("atom-duplicate", "s0-evidence"),)
    assert delta_projection(delta)["dedup_alias_bindings"] == (
        ("atom-duplicate", "s0-evidence"),
    )

    legacy = MembershipDelta(
        stage_id="S1",
        parent_stage_id="S0",
        trace=_trace("e1"),
        additions=(EvidenceItem("e1", "turn-1", "raw evidence", 2),),
    )
    assert "dedup_alias_bindings" not in delta_projection(legacy)


def test_membership_dedup_aliases_are_immutable_unique_and_selection_ordered() -> None:
    trace = StageTrace(
        candidate_ids=("duplicate-a", "duplicate-b"),
        selected_before_dedup_ids=("duplicate-a", "duplicate-b"),
        dedup_excluded_ids=("duplicate-a", "duplicate-b"),
        token_cap=0,
        disposition=StageDisposition.NO_OP,
    )

    with pytest.raises(MatchedEvalContractError, match="immutable"):
        MembershipDelta(
            "BRIDGE",
            "S0",
            trace,
            dedup_alias_bindings=[("duplicate-a", "e0")],  # type: ignore[arg-type]
        )
    with pytest.raises(MatchedEvalContractError, match="unique"):
        MembershipDelta(
            "BRIDGE",
            "S0",
            trace,
            dedup_alias_bindings=(
                ("duplicate-a", "e0"),
                ("duplicate-a", "e0"),
            ),
        )
    with pytest.raises(MatchedEvalContractError, match="order"):
        MembershipDelta(
            "BRIDGE",
            "S0",
            trace,
            dedup_alias_bindings=(
                ("duplicate-b", "e0"),
                ("duplicate-a", "e0"),
            ),
        )


def test_item_token_counts_are_measured_and_exact_empty_evidence_is_preserved() -> None:
    assert EvidenceItem("empty", "turn-empty", "", 0).text == ""
    with pytest.raises(MatchedEvalContractError, match="tokenizer count"):
        EvidenceItem("e1", "turn-1", "raw evidence", 1)
    with pytest.raises(MatchedEvalContractError, match="tokenizer count"):
        FactItem("f1", "compressed fact", ("e1",), 1)
    with pytest.raises(MatchedEvalContractError, match="non-empty"):
        LinkItem("l1", "", ("e1",), 0)


def test_runtime_collections_are_immutable_by_construction() -> None:
    evidence = EvidenceItem("e1", "turn-1", "raw evidence", 2)
    with pytest.raises(MatchedEvalContractError, match="immutable typed tuple"):
        EvaluationMemorySnapshot(
            population_identity_sha256=SHA_A,
            question_order_sha256=SHA_B,
            source_artifacts=[ArtifactRef("retrieval", SHA_A)],  # type: ignore[arg-type]
        )
    with pytest.raises(MatchedEvalContractError, match="immutable typed tuple"):
        MemoryPacket(
            question_id="q1",
            question_sha256=SHA_A,
            dated_question="[Question asked at 2026/08/26]\nWhat happened?",
            dated_question_sha256=SHA_B,
            stage_id="S0",
            protected_evidence=[evidence],  # type: ignore[arg-type]
        )
    with pytest.raises(MatchedEvalContractError, match="immutable EvidenceItem"):
        MembershipDelta(
            stage_id="S1",
            parent_stage_id="S0",
            trace=_trace("e1", tokens_used=2),
            additions=[evidence],  # type: ignore[arg-type]
        )


def test_arm_plan_enforces_isolated_star_and_cumulative_chain() -> None:
    budget = StageBudget(token_cap=100, provider_prompt_cap=1)
    isolated = ArmPlan(
        plan_id="isolated-v2",
        mode=PlanMode.ISOLATED,
        root_stage_id="S0",
        stages=(
            StagePlan("EM", "S0", "em", "representation", budget),
            StagePlan("CAV", "S0", "cav", "linking", budget),
        ),
        global_provider_prompt_cap=2,
    )
    cumulative = ArmPlan(
        plan_id="cumulative-v2",
        mode=PlanMode.CUMULATIVE,
        root_stage_id="S0",
        stages=(
            StagePlan("S1", "S0", "membership", "membership", budget),
            StagePlan("EM", "S1", "em", "representation", budget),
        ),
        global_provider_prompt_cap=2,
    )

    assert isolated.mode is PlanMode.ISOLATED
    assert cumulative.stages[-1].parent_stage_id == "S1"

    with pytest.raises(MatchedEvalContractError, match="root"):
        ArmPlan(
            plan_id="bad-isolated",
            mode=PlanMode.ISOLATED,
            root_stage_id="S0",
            stages=(
                StagePlan("S1", "S0", "membership", "membership", budget),
                StagePlan("EM", "S1", "em", "representation", budget),
            ),
            global_provider_prompt_cap=2,
        )

    with pytest.raises(MatchedEvalContractError, match="mode"):
        ArmPlan(
            plan_id="string-mode",
            mode="isolated",  # type: ignore[arg-type]
            root_stage_id="S0",
            stages=(),
            global_provider_prompt_cap=0,
        )
