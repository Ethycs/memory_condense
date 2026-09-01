from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tests.test_matched_eval_closure_judging import (
    _VerdictClient,
    _gold_loader,
    _parent_plane,
    _seal_parent_judge,
    _sha,
)
from tools import run_matched_eval_spine as spine_cli
from tools.matched_eval import fact_gate_judging, fact_gate_live, judging, live
from tools.matched_eval.artifacts import publish_sealed_json
from tools.matched_eval.contracts import MatchedEvalContractError, StageDisposition
from tools.matched_eval.fact_gate_live import (
    VerifiedFactGateAnswerPlane,
    VerifiedFactGateAnswerRow,
)
from tools.matched_eval.ledger import RuntimeLedgerEntry, build_runtime_ledger


def _fact_plane(
    parent: live.VerifiedS0V2AnswerPlane,
    *,
    predictions: tuple[str, ...],
    terra_ordinals: frozenset[int],
) -> VerifiedFactGateAnswerPlane:
    run_sha = _sha("fact-gate-answer-run")
    snapshot_id = _sha("fact-gate-snapshot")
    retrieval_sha = _sha("sealed-retrieval")
    em_run_sha = _sha("fact-gate-em-run")
    compression_sha = _sha("fact-gate-compression")
    route_policy_sha = _sha("fact-gate-route-policy")
    answer_preflight_sha = _sha("fact-gate-answer-preflight")
    rows = []
    entries = []
    for parent_row, prediction in zip(parent.rows, predictions, strict=True):
        ordinal = parent_row.ordinal
        terra = ordinal in terra_ordinals
        changed = quote_sha256(prediction) != parent_row.prediction_sha256
        packet_id = _sha(f"fact-gate-packet-{ordinal}")
        prompt_id = _sha(f"fact-gate-prompt-{ordinal}")
        prompt_sha = _sha(f"fact-gate-messages-{ordinal}")
        source_row_sha = _sha(f"fact-gate-source-row-{ordinal}")
        gate_receipt = _sha(f"fact-gate-receipt-{ordinal}")
        delta_id = _sha(f"fact-gate-delta-{ordinal}")
        stage = RuntimeLedgerEntry(
            event_type="stage",
            ordinal=ordinal,
            question_id=parent_row.question_id,
            question_sha256=parent_row.question_sha256,
            arm_label=fact_gate_live.ARM_LABEL,
            parent_arm_label=live.ARM_LABEL,
            stage_id=fact_gate_live.FACT_GATE_STAGE_ID,
            parent_stage_id=live.SOURCE_STAGE_ID,
            mechanism_id="provider_free_routed_em_fact_gate",
            delta_kind="fact_memory_representation",
            renderer_id=fact_gate_live.RENDERER_ID,
            legacy_renderer=False,
            disposition=(
                StageDisposition.ADDED if terra else StageDisposition.NO_OP
            ),
            candidate_ids=(delta_id,) if terra else (),
            selected_before_dedup_ids=(delta_id,) if terra else (),
            admitted_ids=(delta_id,) if terra else (),
            global_provider_prompt_cap=len(terra_ordinals),
            max_final_prompt_tokens=8_000,
            prompt_token_proxy=100,
            parent_packet_sha256=packet_id,
            packet_sha256=packet_id,
            prompt_id=prompt_id,
            prompt_messages_sha256=prompt_sha,
            delta_sha256=gate_receipt,
            stage_receipt_sha256=gate_receipt,
            reason=(
                "positive_cell_exact_cited_fact_delta"
                if terra
                else "question_route_not_admitted"
            ),
        )
        answer = RuntimeLedgerEntry(
            event_type="answer_observation",
            ordinal=ordinal,
            question_id=parent_row.question_id,
            question_sha256=parent_row.question_sha256,
            arm_label=fact_gate_live.ARM_LABEL,
            parent_arm_label=live.ARM_LABEL,
            stage_id=fact_gate_live.ANSWER_STAGE_ID,
            parent_stage_id=fact_gate_live.FACT_GATE_STAGE_ID,
            mechanism_id=(
                "routed_em_fact_gate_terra_responder"
                if terra
                else "sealed_parent_prediction_reuse"
            ),
            delta_kind="observation",
            renderer_id=fact_gate_live.RENDERER_ID,
            legacy_renderer=False,
            disposition=StageDisposition.NO_OP,
            provider_calls=int(terra),
            provider_prompt_cap=int(terra),
            provider_prompt_reserved=int(terra),
            global_provider_prompt_cap=len(terra_ordinals),
            max_final_prompt_tokens=8_000,
            prompt_token_proxy=100,
            parent_packet_sha256=packet_id,
            packet_sha256=packet_id,
            prompt_id=prompt_id,
            prompt_messages_sha256=prompt_sha,
            prediction=prediction,
            prediction_sha256=quote_sha256(prediction),
            changed_from_parent=changed,
            source_row_sha256=source_row_sha,
            reason=(
                "sealed_terra_routed_em_fact_gate_prediction"
                if terra
                else "sealed_s0_v2_parent_prediction_reuse"
            ),
        )
        entries.extend((stage, answer))
        rows.append(
            VerifiedFactGateAnswerRow(
                ordinal=ordinal,
                question_id=parent_row.question_id,
                question_sha256=parent_row.question_sha256,
                dated_question_sha256=parent_row.dated_question_sha256,
                prediction=prediction,
                prediction_sha256=quote_sha256(prediction),
                prediction_source=(
                    "terra_fact_gate" if terra else "sealed_parent_fallback"
                ),
                parent_prediction_sha256=parent_row.prediction_sha256,
                changed_from_parent=changed,
                route_id="numeric_reduce" if terra else "point_lookup",
                gate_disposition="compiled" if terra else "parent_fallback",
                gate_reason=(
                    "positive_cell_exact_cited_fact_delta"
                    if terra
                    else "question_route_not_admitted"
                ),
                fact_gate_receipt_sha256=gate_receipt,
                final_packet_id=packet_id,
                final_prompt_id=prompt_id,
                final_prompt_messages_sha256=prompt_sha,
                source_row_sha256=source_row_sha,
                runtime_row_id=answer.row_id,
                call_key_sha256=_sha(f"fact-gate-call-{ordinal}") if terra else None,
                request_journal_sha256=(
                    _sha(f"fact-gate-request-{ordinal}") if terra else None
                ),
                response_journal_sha256=(
                    _sha(f"fact-gate-response-{ordinal}") if terra else None
                ),
            )
        )
    ledger = build_runtime_ledger(
        snapshot_id=snapshot_id,
        plan_id=fact_gate_live.ANSWER_PLAN_ID,
        entries=entries,
        source_artifacts=(
            {
                "role": f"{fact_gate_live.ARM_LABEL}:sealed_retrieval",
                "sha256": retrieval_sha,
            },
            {
                "role": f"{fact_gate_live.ARM_LABEL}:em_run",
                "sha256": em_run_sha,
            },
            {
                "role": f"{fact_gate_live.ARM_LABEL}:em_compression",
                "sha256": compression_sha,
            },
            {
                "role": f"{fact_gate_live.ARM_LABEL}:route_policy",
                "sha256": route_policy_sha,
            },
            {
                "role": f"{fact_gate_live.ARM_LABEL}:parent_answer_run",
                "sha256": parent.run_sha256,
            },
            {
                "role": f"{fact_gate_live.ARM_LABEL}:answer_preflight",
                "sha256": answer_preflight_sha,
            },
            {
                "role": f"{fact_gate_live.ARM_LABEL}:answer_run",
                "sha256": run_sha,
            },
        ),
    )
    return VerifiedFactGateAnswerPlane(
        run_sha256=run_sha,
        replay_sha256=run_sha,
        runtime_ledger_sha256=_sha("fact-gate-runtime-ledger-artifact"),
        runtime_ledger=live._freeze_json(ledger),
        arm_label=fact_gate_live.ARM_LABEL,
        parent_arm_label=live.ARM_LABEL,
        parent_answer_run_sha256=parent.run_sha256,
        matched_population_id=parent.matched_population_id,
        population_identity_sha256=parent.population_identity_sha256,
        retrieval_sha256=retrieval_sha,
        source_em_run_sha256=em_run_sha,
        source_em_compression_sha256=compression_sha,
        source_preflight_sha256=answer_preflight_sha,
        route_policy_sha256=route_policy_sha,
        snapshot_id=snapshot_id,
        arm_plan_id=fact_gate_live.ARM_PLAN_ID,
        answer_plan_id=fact_gate_live.ANSWER_PLAN_ID,
        renderer_id=fact_gate_live.RENDERER_ID,
        rows=tuple(rows),
        parent_plane=parent,
    )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    changed: bool = True,
):
    parent_predictions = ("wrong parent answer 0", "gold answer 1")
    parent = _parent_plane(predictions=parent_predictions)
    child = _fact_plane(
        parent,
        predictions=(
            "gold answer 0" if changed else parent_predictions[0],
            parent_predictions[1],
        ),
        terra_ordinals=frozenset({0}),
    )
    parent_root, parent_judge_sha, parent_score_sha = _seal_parent_judge(
        tmp_path,
        monkeypatch,
        parent=parent,
        verdicts=["INCORRECT - parent missed the answer.", "CORRECT - exact."],
    )
    monkeypatch.setattr(fact_gate_judging, "_load_gold", _gold_loader)
    return {
        "answer_plane": child,
        "dataset_path": tmp_path / "unused-dataset.json",
        "split_path": tmp_path / "unused-split.json",
        "parent_judge_root": parent_root,
        "expected_parent_judge_sha256": parent_judge_sha,
        "expected_parent_score_ledger_sha256": parent_score_sha,
        "output_root": tmp_path / "fact-gate-judge",
        "expected_question_count": 2,
    }


def test_fact_gate_changed_only_judge_reuses_parent_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _fixture(tmp_path, monkeypatch)
    client = _VerdictClient(["CORRECT - candidate recovered the answer."])
    monkeypatch.setattr(judging, "_make_provider_client", lambda *_args: client)

    result = fact_gate_judging.run_fact_gate_changed_only_judge(
        **request,
        enable_provider=True,
        authorized_provider_calls=1,
        api_key_env="MATCHED_TEST_KEY",
        max_concurrency=1,
    )

    assert result.physical_provider_calls == 1
    assert len(client.completions.requests) == 1
    payload = result.judge_artifact.payload
    assert payload["aggregate"]["changed_predictions"] == 1
    assert payload["aggregate"]["fresh_judgments"] == 1
    assert payload["aggregate"]["inherited_judgments"] == 1
    assert payload["aggregate"]["correct"] == 2
    assert payload["questions"][0]["verdict_source"] == "new_sol_judge"
    assert payload["questions"][0]["rescued"] is True
    assert payload["questions"][1]["verdict_source"] == (
        "sealed_parent_s0_v2_judge"
    )
    assert payload["questions"][1]["judge_output"] is None
    score = result.score_ledger_artifact.payload
    assert score["row_count"] == 2
    assert score["aggregate"]["candidate_correct"] == 2
    assert score["total_historical_provider_calls"] == 1

    monkeypatch.setattr(
        judging,
        "_make_provider_client",
        lambda *_args: pytest.fail("fact-gate judge replay built a client"),
    )
    replay = fact_gate_judging.replay_fact_gate_changed_only_judge(
        **request,
        expected_judge_sha256=result.judge_artifact.sha256,
        max_concurrency=1,
    )
    assert replay.physical_provider_calls == 0
    assert replay.checkpoint_hits == 1
    assert replay.judge_artifact.sha256 == result.judge_artifact.sha256
    assert replay.score_ledger_artifact.sha256 == (
        result.score_ledger_artifact.sha256
    )

    resumed = fact_gate_judging.run_fact_gate_changed_only_judge(
        **request,
        enable_provider=True,
        authorized_provider_calls=1,
        api_key_env="MATCHED_TEST_KEY",
        max_concurrency=1,
    )
    assert resumed.judge_artifact.sha256 == result.judge_artifact.sha256
    assert resumed.physical_provider_calls == 0
    assert resumed.checkpoint_hits == 1


def test_fact_gate_judge_authorization_precedes_output_and_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _fixture(tmp_path, monkeypatch)
    output = Path(request["output_root"])
    monkeypatch.setattr(
        judging,
        "_make_provider_client",
        lambda *_args: pytest.fail("authorization failure built a client"),
    )

    with pytest.raises(MatchedEvalContractError, match="exactly equal 1"):
        fact_gate_judging.run_fact_gate_changed_only_judge(
            **request,
            enable_provider=True,
            authorized_provider_calls=0,
            max_concurrency=1,
        )

    assert not output.exists()


def test_fact_gate_structural_tamper_fails_before_gold_or_parent_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _fixture(tmp_path, monkeypatch)
    plane = request["answer_plane"]
    bad_row = replace(plane.rows[0], changed_from_parent=False)
    request["answer_plane"] = replace(plane, rows=(bad_row, plane.rows[1]))
    monkeypatch.setattr(
        fact_gate_judging,
        "_load_gold",
        lambda **_kwargs: pytest.fail("structural failure loaded gold"),
    )
    monkeypatch.setattr(
        fact_gate_judging.closure_judging,
        "_load_parent_judge",
        lambda **_kwargs: pytest.fail("structural failure loaded parent outcomes"),
    )

    with pytest.raises(MatchedEvalContractError, match="change flag"):
        fact_gate_judging.preflight_fact_gate_changed_only_judge(**request)


def test_fact_gate_terra_same_prediction_inherits_without_sol_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _fixture(tmp_path, monkeypatch, changed=False)
    monkeypatch.setattr(
        judging,
        "_make_provider_client",
        lambda *_args: pytest.fail("unchanged fact-gate judge built a client"),
    )

    result = fact_gate_judging.run_fact_gate_changed_only_judge(
        **request,
        enable_provider=False,
        authorized_provider_calls=0,
        max_concurrency=1,
    )

    assert result.physical_provider_calls == result.checkpoint_hits == 0
    assert result.judge_artifact.payload["completion_batch"] is None
    assert result.judge_artifact.payload["aggregate"]["inherited_judgments"] == 2


def _judge_cli_args(command: str, output: Path, answer_sha: str) -> list[str]:
    args = [
        command,
        "--output-root",
        str(output),
        "--dataset",
        str(output / "dataset.json"),
        "--expected-em-run-sha256",
        "1" * 64,
        "--expected-parent-answer-run-sha256",
        "2" * 64,
        "--expected-answer-run-sha256",
        answer_sha,
        "--expected-parent-judge-sha256",
        "3" * 64,
        "--expected-parent-score-ledger-sha256",
        "4" * 64,
    ]
    if command == "fact-gate-judge":
        args.extend(("--enable-provider", "--authorized-provider-calls", "14"))
    elif command == "fact-gate-judge-replay":
        args.extend(("--expected-judge-sha256", "5" * 64))
    return args


def _seal_answer_replays(output: Path) -> str:
    answer, _created = publish_sealed_json(
        output / fact_gate_live.ANSWER_REPLAY_NAME,
        {"kind": "fact-gate-answer-replay"},
    )
    ledger, _created = publish_sealed_json(
        output / fact_gate_live.RUNTIME_LEDGER_NAME,
        {"kind": "fact-gate-runtime-ledger"},
    )
    replay, _created = publish_sealed_json(
        output / fact_gate_live.RUNTIME_LEDGER_REPLAY_NAME,
        ledger.payload,
    )
    assert replay.sha256 == ledger.sha256
    return answer.sha256


@pytest.mark.parametrize(
    ("command", "function_name"),
    (
        (
            "fact-gate-judge-preflight",
            "preflight_fact_gate_changed_only_judge",
        ),
        ("fact-gate-judge", "run_fact_gate_changed_only_judge"),
        (
            "fact-gate-judge-replay",
            "replay_fact_gate_changed_only_judge",
        ),
    ),
)
def test_fact_gate_judge_cli_replays_answers_then_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    function_name: str,
) -> None:
    output = tmp_path / command
    answer_sha = _seal_answer_replays(output)
    answer_plane = SimpleNamespace(arm_label=fact_gate_live.ARM_LABEL)
    events: list[tuple[str, dict[str, Any]]] = []

    def fake_answer_replay(**kwargs):
        events.append(("answer_replay", kwargs))
        return answer_plane

    def fake_judge(**kwargs):
        events.append(("judge", kwargs))
        if function_name == "preflight_fact_gate_changed_only_judge":
            return SimpleNamespace(
                sha256="6" * 64,
                payload={
                    "arm_label": fact_gate_live.ARM_LABEL,
                    "required_authorized_provider_calls": 14,
                },
            )
        return SimpleNamespace(
            judge_artifact=SimpleNamespace(
                sha256="7" * 64,
                payload={"arm_label": fact_gate_live.ARM_LABEL},
            ),
            score_ledger_artifact=SimpleNamespace(sha256="8" * 64),
            checkpoint_hits=(
                14 if function_name == "replay_fact_gate_changed_only_judge" else 0
            ),
            correct=60,
            physical_provider_calls=(
                14 if function_name == "run_fact_gate_changed_only_judge" else 0
            ),
        )

    monkeypatch.setattr(fact_gate_live, "replay_fact_gate_answers", fake_answer_replay)
    monkeypatch.setattr(fact_gate_judging, function_name, fake_judge)

    assert spine_cli.main(_judge_cli_args(command, output, answer_sha)) == 0
    assert [name for name, _kwargs in events] == ["answer_replay", "judge"]
    answer_request = events[0][1]
    judge_request = events[1][1]
    assert answer_request["expected_run_sha256"] == answer_sha
    assert answer_request["output_root"] == output
    assert judge_request["answer_plane"] is answer_plane
    assert judge_request["output_root"] == output
    assert judge_request["parent_judge_root"] == spine_cli.DEFAULT_S0_V2_ROOT
    if command == "fact-gate-judge":
        assert judge_request["enable_provider"] is True
        assert judge_request["authorized_provider_calls"] == 14
        assert judge_request["max_concurrency"] == 4
    elif command == "fact-gate-judge-replay":
        assert judge_request["expected_judge_sha256"] == "5" * 64
        assert judge_request["max_concurrency"] == 4
    assert "physical_provider_calls=" in capsys.readouterr().out
