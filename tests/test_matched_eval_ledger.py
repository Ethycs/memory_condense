from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import (
    ArmPlan,
    ArtifactRef,
    EvaluationMemorySnapshot,
    MatchedEvalContractError,
    MembershipDelta,
    MemoryPacket,
    PlanMode,
    StageBudget,
    StageDisposition,
    StagePlan,
    StageTrace,
    identity_sha256,
)
from tools.matched_eval.ledger import (
    RuntimeLedgerEntry,
    RuntimeStageRunBinding,
    ScoreLedgerEntry,
    build_runtime_ledger,
    build_score_ledger,
    load_verified_runtime_answer_plane,
    runtime_entry_from_stage_run,
)
from tools.matched_eval.runner import MatchedEvalRunner


SHA_A = "a" * 64
SHA_B = "b" * 64
PREDICTION = "A concise answer."
PREDICTION_SHA = hashlib.sha256(PREDICTION.encode("utf-8")).hexdigest()


class _NoOpMembershipAdapter:
    mechanism_id = "runtime_noop"
    delta_kind = "membership"

    def propose(
        self,
        *,
        snapshot: EvaluationMemorySnapshot,
        packet: MemoryPacket,
        stage: StagePlan,
    ) -> MembershipDelta:
        del snapshot, packet
        return MembershipDelta(
            stage_id=stage.stage_id,
            parent_stage_id=stage.parent_stage_id,
            trace=StageTrace(
                token_cap=stage.budget.token_cap,
                disposition=StageDisposition.NO_OP,
                reason="no_candidates",
            ),
        )


def _stage_plane() -> tuple[
    EvaluationMemorySnapshot,
    ArmPlan,
    tuple[RuntimeStageRunBinding, ...],
]:
    snapshot = EvaluationMemorySnapshot(
        population_identity_sha256=SHA_A,
        question_order_sha256=SHA_B,
        source_artifacts=(ArtifactRef("sealed_retrieval", PREDICTION_SHA),),
    )
    plan = ArmPlan(
        plan_id="runtime-answer-plan",
        mode=PlanMode.ISOLATED,
        root_stage_id="S0",
        stages=(
            StagePlan(
                stage_id="S1",
                parent_stage_id="S0",
                mechanism_id="runtime_noop",
                delta_kind="membership",
                budget=StageBudget(token_cap=16, provider_prompt_cap=0),
            ),
        ),
        global_provider_prompt_cap=0,
    )
    runner = MatchedEvalRunner(
        {"runtime_noop": _NoOpMembershipAdapter()}
    )
    bindings: list[RuntimeStageRunBinding] = []
    for ordinal in range(2):
        question_id = f"q{ordinal}"
        question = f"What happened in row {ordinal}?"
        dated = f"[Question asked at 2026-08-27]\n{question}"
        root = MemoryPacket(
            question_id=question_id,
            question_sha256=hashlib.sha256(question.encode("utf-8")).hexdigest(),
            dated_question=dated,
            dated_question_sha256=hashlib.sha256(dated.encode("utf-8")).hexdigest(),
            stage_id="S0",
        )
        result = runner.run(snapshot=snapshot, root_packet=root, plan=plan)
        bindings.append(
            RuntimeStageRunBinding(
                ordinal=ordinal,
                arm_label="RUNTIME_TEST_ARM",
                parent_arm_label="S0_CONTROL",
                run=result,
                stage_id="S1",
            )
        )
    return snapshot, plan, tuple(bindings)


def _published_runtime_plane(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot, plan, bindings = _stage_plane()
    answer_value = {
        "arm_label": "RUNTIME_TEST_ARM",
        "format": "synthetic-runtime-bound-answer-v1",
        "gold_loaded": False,
        "question_count": len(bindings),
        "renderer_id": snapshot.renderer_id,
        "snapshot_id": snapshot.snapshot_id,
        "questions": [
            {
                "ordinal": binding.ordinal,
                "question_id": binding.run.root_packet.question_id,
                "question_sha256": binding.run.root_packet.question_sha256,
            }
            for binding in bindings
        ],
    }
    answer, _created = publish_sealed_json(tmp_path / "run.json", answer_value)
    publish_sealed_json(tmp_path / "run-replay.json", answer_value)
    entries = tuple(
        runtime_entry_from_stage_run(
            ordinal=binding.ordinal,
            arm_label=binding.arm_label,
            parent_arm_label=binding.parent_arm_label,
            run=binding.run,
            stage_id=binding.stage_id,
        )
        for binding in bindings
    )
    runtime_value = build_runtime_ledger(
        snapshot_id=snapshot.snapshot_id,
        plan_id=plan.plan_id,
        entries=entries,
        source_artifacts=(
            {"role": "RUNTIME_TEST_ARM:retrieval", "sha256": SHA_A},
            {"role": "RUNTIME_TEST_ARM:run", "sha256": answer.sha256},
        ),
    )
    runtime, _created = publish_sealed_json(
        tmp_path / "runtime-ledger.json", runtime_value
    )
    publish_sealed_json(tmp_path / "runtime-ledger-replay.json", runtime_value)
    arguments: dict[str, Any] = {
        "answer_run_path": answer.path,
        "answer_run_replay_path": tmp_path / "run-replay.json",
        "expected_answer_run_sha256": answer.sha256,
        "runtime_ledger_path": runtime.path,
        "runtime_ledger_replay_path": tmp_path / "runtime-ledger-replay.json",
        "expected_runtime_ledger_sha256": runtime.sha256,
        "snapshot_id": snapshot.snapshot_id,
        "plan_id": plan.plan_id,
        "renderer_id": snapshot.renderer_id,
        "stage_runs": bindings,
        "answer_run_artifact_role": "RUNTIME_TEST_ARM:run",
        "source_artifacts": (
            {"role": "RUNTIME_TEST_ARM:retrieval", "sha256": SHA_A},
        ),
    }
    return arguments, runtime_value


def _runtime(question: str, ordinal: int) -> RuntimeLedgerEntry:
    return RuntimeLedgerEntry(
        event_type="answer_observation",
        ordinal=ordinal,
        question_id=question,
        question_sha256=SHA_A,
        arm_label="S0_CONTROL_V2",
        parent_arm_label=None,
        stage_id="S0",
        parent_stage_id=None,
        mechanism_id="s0",
        delta_kind="membership",
        renderer_id="matched_typed_slots_v2",
        legacy_renderer=False,
        disposition=StageDisposition.NO_OP,
        token_cap=8_000,
        prediction=PREDICTION,
        prediction_sha256=PREDICTION_SHA,
        provider_calls=1,
    )


def test_runtime_and_score_planes_join_without_gold_leaking_runtime() -> None:
    first = _runtime("q1", 0)
    second = _runtime("q2", 1)
    runtime = build_runtime_ledger(
        snapshot_id=SHA_A,
        plan_id="matched-v2",
        entries=(first, second),
    )
    scores = build_score_ledger(
        runtime_ledger=runtime,
        entries=(
            ScoreLedgerEntry(first.row_id, True),
            ScoreLedgerEntry(second.row_id, False),
        ),
        source_artifacts=(
            {"role": "S0_CONTROL:judge", "sha256": SHA_B},
            {"role": "S0_CONTROL:judge_replay", "sha256": SHA_B},
        ),
    )

    assert runtime["gold_loaded"] is False
    assert "correct" not in str(runtime)
    assert runtime["total_provider_calls"] == 2
    assert scores["aggregate"]["candidate_correct"] == 1
    assert scores["source_artifacts"] == [
        {"role": "S0_CONTROL:judge", "sha256": SHA_B},
        {"role": "S0_CONTROL:judge_replay", "sha256": SHA_B},
    ]


def test_runtime_candidate_lifecycle_remains_select_then_dedup() -> None:
    row = RuntimeLedgerEntry(
        event_type="stage",
        ordinal=0,
        question_id="q1",
        question_sha256=SHA_A,
        arm_label="EM_V2",
        parent_arm_label="S0_V2",
        stage_id="EM",
        parent_stage_id="S0",
        mechanism_id="em",
        delta_kind="representation",
        renderer_id="matched_typed_slots_v2",
        legacy_renderer=False,
        disposition=StageDisposition.ADDED,
        candidate_ids=("x", "y"),
        selected_before_dedup_ids=("x", "y"),
        dedup_excluded_ids=("x",),
        admitted_ids=("y",),
        token_cap=10,
        tokens_used=3,
    )

    projection = row.projection()
    assert projection["selected_before_dedup_ids"] == ["x", "y"]
    assert projection["dedup_excluded_ids"] == ["x"]


def test_score_plane_rejects_impossible_target_counts() -> None:
    with pytest.raises(MatchedEvalContractError, match="exceeds"):
        ScoreLedgerEntry(
            runtime_row_id=SHA_A,
            correct=True,
            primary_target_count=1,
            primary_target_recalled=2,
        )


@pytest.mark.parametrize("kind", ("judge", "judge_replay", "score", "gold"))
def test_runtime_rejects_score_or_gold_source_artifact_roles(kind: str) -> None:
    with pytest.raises(MatchedEvalContractError, match="forbidden at runtime"):
        build_runtime_ledger(
            snapshot_id=SHA_A,
            plan_id="matched-v2",
            entries=(_runtime("q1", 0),),
            source_artifacts=({"role": f"S0_CONTROL:{kind}", "sha256": SHA_B},),
        )


def test_source_artifact_mappings_are_exact_and_roles_are_unique() -> None:
    row = _runtime("q1", 0)
    lookalike_roles = (
        {"role": "S0_CONTROL:golden_run", "sha256": SHA_A},
        {"role": "S0_CONTROL:scored_run", "sha256": SHA_B},
        {"role": "S0_CONTROL:prejudge_input", "sha256": PREDICTION_SHA},
    )
    runtime_with_lookalikes = build_runtime_ledger(
        snapshot_id=SHA_A,
        plan_id="matched-v2",
        entries=(row,),
        source_artifacts=lookalike_roles,
    )
    assert tuple(runtime_with_lookalikes["source_artifacts"]) == lookalike_roles
    with pytest.raises(MatchedEvalContractError, match="exactly role and sha256"):
        build_runtime_ledger(
            snapshot_id=SHA_A,
            plan_id="matched-v2",
            entries=(row,),
            source_artifacts=(
                {"role": "S0_CONTROL:run", "sha256": SHA_B, "path": "run.json"},
            ),
        )
    with pytest.raises(MatchedEvalContractError, match="roles must be unique"):
        build_runtime_ledger(
            snapshot_id=SHA_A,
            plan_id="matched-v2",
            entries=(row,),
            source_artifacts=(
                {"role": "S0_CONTROL:run", "sha256": SHA_A},
                {"role": "S0_CONTROL:run", "sha256": SHA_B},
            ),
        )
    with pytest.raises(MatchedEvalContractError, match="lowercase SHA-256"):
        build_runtime_ledger(
            snapshot_id=SHA_A,
            plan_id="matched-v2",
            entries=(row,),
            source_artifacts=(
                {"role": "S0_CONTROL:run", "sha256": "not-a-digest"},
            ),
        )
    runtime = build_runtime_ledger(
        snapshot_id=SHA_A,
        plan_id="matched-v2",
        entries=(row,),
    )
    with pytest.raises(MatchedEvalContractError, match="roles must be unique"):
        build_score_ledger(
            runtime_ledger=runtime,
            entries=(ScoreLedgerEntry(row.row_id, True),),
            source_artifacts=(
                {"role": "S0_CONTROL:judge", "sha256": SHA_A},
                {"role": "S0_CONTROL:judge", "sha256": SHA_B},
            ),
        )


def test_score_builder_verifies_runtime_seal_and_exact_answer_row_order() -> None:
    first = _runtime("q1", 0)
    stage = RuntimeLedgerEntry(
        event_type="stage",
        ordinal=0,
        question_id="q1",
        question_sha256=SHA_A,
        arm_label="S0_CONTROL_V2",
        parent_arm_label=None,
        stage_id="S0_PREP",
        parent_stage_id=None,
        mechanism_id="s0_prep",
        delta_kind="membership",
        renderer_id="matched_typed_slots_v2",
        legacy_renderer=False,
        disposition=StageDisposition.NO_OP,
    )
    second = _runtime("q2", 1)
    runtime = build_runtime_ledger(
        snapshot_id=SHA_A,
        plan_id="matched-v2",
        entries=(first, stage, second),
    )
    score_rows = (
        ScoreLedgerEntry(first.row_id, True),
        ScoreLedgerEntry(second.row_id, False),
    )
    scores = build_score_ledger(runtime_ledger=runtime, entries=score_rows)
    assert tuple(row["runtime_row_id"] for row in scores["rows"]) == (
        first.row_id,
        second.row_id,
    )

    with pytest.raises(MatchedEvalContractError, match="exact order"):
        build_score_ledger(
            runtime_ledger=runtime,
            entries=tuple(reversed(score_rows)),
        )
    with pytest.raises(MatchedEvalContractError, match="exact order"):
        build_score_ledger(runtime_ledger=runtime, entries=score_rows[:1])

    bad_seal = deepcopy(runtime)
    bad_seal["ledger_identity_sha256"] = SHA_B
    with pytest.raises(MatchedEvalContractError, match="identity seal"):
        build_score_ledger(runtime_ledger=bad_seal, entries=score_rows)

    bad_row = deepcopy(runtime)
    bad_row["rows"][0]["row_id"] = SHA_B
    unsigned = dict(bad_row)
    unsigned.pop("ledger_identity_sha256")
    bad_row["ledger_identity_sha256"] = identity_sha256(unsigned)
    with pytest.raises(MatchedEvalContractError, match="row 0 ID is invalid"):
        build_score_ledger(runtime_ledger=bad_row, entries=score_rows)


def test_runtime_entry_requires_exact_types_and_prediction_digest() -> None:
    row = _runtime("q1", 0)
    with pytest.raises(MatchedEvalContractError, match="prediction SHA-256"):
        replace(row, prediction_sha256=SHA_B)
    with pytest.raises(MatchedEvalContractError, match="exact integer"):
        replace(row, ordinal=True)
    with pytest.raises(MatchedEvalContractError, match="exact tuple"):
        replace(row, candidate_ids=[])  # type: ignore[arg-type]
    with pytest.raises(MatchedEvalContractError, match="exact StageDisposition"):
        replace(row, disposition="no_op")  # type: ignore[arg-type]
    with pytest.raises(MatchedEvalContractError, match="exact bool"):
        replace(row, legacy_renderer=0)  # type: ignore[arg-type]
    with pytest.raises(MatchedEvalContractError, match="exact integer"):
        replace(row, provider_calls=True)


def test_runtime_ledger_rejects_duplicate_logical_answers() -> None:
    first = _runtime("q1", 0)
    second_prediction = "A different answer."
    second = replace(
        first,
        prediction=second_prediction,
        prediction_sha256=hashlib.sha256(second_prediction.encode("utf-8")).hexdigest(),
    )
    with pytest.raises(MatchedEvalContractError, match="unique per"):
        build_runtime_ledger(
            snapshot_id=SHA_A,
            plan_id="matched-v2",
            entries=(first, second),
        )


def test_score_entry_requires_exact_types_and_consistent_comparison() -> None:
    with pytest.raises(MatchedEvalContractError, match="correctness must be"):
        ScoreLedgerEntry(SHA_A, 1)  # type: ignore[arg-type]
    with pytest.raises(MatchedEvalContractError, match="all present or all null"):
        ScoreLedgerEntry(SHA_A, True, baseline_correct=False)
    with pytest.raises(MatchedEvalContractError, match="rescue flag"):
        ScoreLedgerEntry(
            SHA_A,
            True,
            baseline_correct=False,
            changed_from_baseline=True,
            rescued=False,
            regressed=False,
        )
    with pytest.raises(MatchedEvalContractError, match="present together"):
        ScoreLedgerEntry(SHA_A, True, primary_target_count=1)
    with pytest.raises(MatchedEvalContractError, match="exact integer"):
        ScoreLedgerEntry(SHA_A, True, historical_provider_calls=True)


def test_verified_runtime_answer_plane_reconstructs_exact_stage_rows(
    tmp_path: Path,
) -> None:
    arguments, runtime_value = _published_runtime_plane(tmp_path)

    verified = load_verified_runtime_answer_plane(**arguments)

    assert verified.answer_run_sha256 == arguments["expected_answer_run_sha256"]
    assert verified.runtime_ledger_sha256 == arguments[
        "expected_runtime_ledger_sha256"
    ]
    assert verified.runtime_ledger_identity_sha256 == runtime_value[
        "ledger_identity_sha256"
    ]
    assert verified.snapshot_id == arguments["snapshot_id"]
    assert verified.plan_id == arguments["plan_id"]
    assert verified.renderer_id == arguments["renderer_id"]
    assert [row.ordinal for row in verified.entries] == [0, 1]
    for row, raw in zip(
        verified.entries, runtime_value["rows"], strict=True
    ):
        assert row.stage_receipt_sha256 == raw["stage_receipt_sha256"]
        assert row.parent_packet_sha256 == raw["parent_packet_sha256"]
        assert row.packet_sha256 == raw["packet_sha256"]
        assert row.renderer_id == arguments["renderer_id"]


@pytest.mark.parametrize(
    ("scope", "field", "replacement"),
    (
        ("ledger", "snapshot_id", SHA_B),
        ("ledger", "plan_id", "changed-plan"),
        ("row", "renderer_id", "changed-renderer"),
        ("row", "stage_receipt_sha256", SHA_B),
        ("row", "packet_sha256", SHA_B),
    ),
)
def test_verified_runtime_answer_plane_rejects_resealed_runtime_tampering(
    tmp_path: Path,
    scope: str,
    field: str,
    replacement: str,
) -> None:
    arguments, runtime_value = _published_runtime_plane(tmp_path)
    tampered = deepcopy(runtime_value)
    if scope == "ledger":
        tampered[field] = replacement
    else:
        tampered["rows"][0][field] = replacement
        row_body = dict(tampered["rows"][0])
        row_body.pop("row_id")
        tampered["rows"][0]["row_id"] = identity_sha256(row_body)
    ledger_body = dict(tampered)
    ledger_body.pop("ledger_identity_sha256")
    tampered["ledger_identity_sha256"] = identity_sha256(ledger_body)
    artifact, _created = publish_sealed_json(
        tmp_path / f"tampered-{field}.json", tampered
    )
    replay, _created = publish_sealed_json(
        tmp_path / f"tampered-{field}-replay.json", tampered
    )
    arguments.update(
        runtime_ledger_path=artifact.path,
        runtime_ledger_replay_path=replay.path,
        expected_runtime_ledger_sha256=artifact.sha256,
    )

    with pytest.raises(
        MatchedEvalContractError,
        match="differs from reconstructed stage executions",
    ):
        load_verified_runtime_answer_plane(**arguments)


def test_verified_runtime_answer_plane_rejects_answer_replay_and_gold_tampering(
    tmp_path: Path,
) -> None:
    arguments, _runtime_value = _published_runtime_plane(tmp_path)
    different = {
        "format": "different-answer-run",
        "gold_loaded": False,
    }
    bad_replay, _created = publish_sealed_json(
        tmp_path / "different-run-replay.json", different
    )
    mismatched = dict(arguments)
    mismatched["answer_run_replay_path"] = bad_replay.path
    with pytest.raises(MatchedEvalContractError, match="run/replay seals differ"):
        load_verified_runtime_answer_plane(**mismatched)

    gold = {
        "format": "gold-bearing-answer-run",
        "gold_loaded": False,
        "reference_answer": "forbidden",
    }
    gold_run, _created = publish_sealed_json(tmp_path / "gold-run.json", gold)
    gold_replay, _created = publish_sealed_json(
        tmp_path / "gold-run-replay.json", gold
    )
    gold_arguments = dict(arguments)
    gold_arguments.update(
        answer_run_path=gold_run.path,
        answer_run_replay_path=gold_replay.path,
        expected_answer_run_sha256=gold_run.sha256,
    )
    with pytest.raises(MatchedEvalContractError, match="gold-bearing field"):
        load_verified_runtime_answer_plane(**gold_arguments)


@pytest.mark.parametrize(
    ("scope", "field", "replacement"),
    (
        ("run", "snapshot_id", SHA_B),
        ("run", "renderer_id", "changed-renderer"),
        ("run", "arm_label", "CHANGED_ARM"),
        ("row", "question_id", "changed-question"),
        ("row", "question_sha256", SHA_B),
    ),
)
def test_verified_runtime_answer_plane_rejects_resealed_answer_binding_tampering(
    tmp_path: Path,
    scope: str,
    field: str,
    replacement: str,
) -> None:
    arguments, _runtime_value = _published_runtime_plane(tmp_path)
    source = read_sealed_json(arguments["answer_run_path"])
    tampered = deepcopy(source.payload)
    if scope == "run":
        tampered[field] = replacement
    else:
        tampered["questions"][0][field] = replacement
    run, _created = publish_sealed_json(
        tmp_path / f"answer-tampered-{field}.json", tampered
    )
    replay, _created = publish_sealed_json(
        tmp_path / f"answer-tampered-{field}-replay.json", tampered
    )
    arguments.update(
        answer_run_path=run.path,
        answer_run_replay_path=replay.path,
        expected_answer_run_sha256=run.sha256,
    )

    with pytest.raises(
        MatchedEvalContractError,
        match="runtime population envelope|question binding",
    ):
        load_verified_runtime_answer_plane(**arguments)


def test_verified_runtime_answer_plane_rejects_binding_order_and_renderer_drift(
    tmp_path: Path,
) -> None:
    arguments, _runtime_value = _published_runtime_plane(tmp_path)
    reversed_arguments = dict(arguments)
    reversed_arguments["stage_runs"] = tuple(
        reversed(arguments["stage_runs"])
    )
    with pytest.raises(MatchedEvalContractError, match="ordinal order"):
        load_verified_runtime_answer_plane(**reversed_arguments)

    renderer_arguments = dict(arguments)
    renderer_arguments["renderer_id"] = "another-renderer"
    with pytest.raises(MatchedEvalContractError, match="renderer binding"):
        load_verified_runtime_answer_plane(**renderer_arguments)
