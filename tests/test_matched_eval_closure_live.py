from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tests.test_fast_completion_runtime import _FakeClient
from tests.test_matched_eval_closure import _sealed_campaign
from tools import run_matched_eval_spine as spine_cli
from tools.matched_eval import closure, closure_judging, closure_live, live
from tools.matched_eval.artifacts import (
    SealedArtifactError,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    StageDisposition,
    identity_sha256,
)


PARENT_RUN_SHA256 = "1" * 64
PARENT_RUNTIME_SHA256 = "2" * 64


def _parent_plane(population) -> live.VerifiedS0V2AnswerPlane:
    rows = []
    for source in population.rows:
        prediction = f"sealed parent answer {source.ordinal}"
        rows.append(
            live.VerifiedS0V2AnswerRow(
                ordinal=source.ordinal,
                question_id=source.packet.question_id,
                question_sha256=source.packet.question_sha256,
                dated_question_sha256=source.packet.dated_question_sha256,
                messages_sha256=source.rendered_prompt.messages_sha256,
                prediction=prediction,
                prediction_sha256=quote_sha256(prediction),
                call_key_sha256=identity_sha256(
                    {"parent_call": source.ordinal}
                ),
                request_journal_sha256=identity_sha256(
                    {"parent_request": source.ordinal}
                ),
                response_journal_sha256=identity_sha256(
                    {"parent_response": source.ordinal}
                ),
                source_row_sha256=identity_sha256(
                    {"parent_row": source.ordinal}
                ),
                runtime_row_id=identity_sha256(
                    {"parent_runtime": source.ordinal}
                ),
            )
        )
    return live.VerifiedS0V2AnswerPlane(
        run_sha256=PARENT_RUN_SHA256,
        replay_sha256=PARENT_RUN_SHA256,
        matched_population_id=population.population_id,
        population_identity_sha256=(
            population.snapshot.population_identity_sha256
        ),
        snapshot_id=population.snapshot.snapshot_id,
        renderer_id=population.renderer_id,
        runtime_ledger=live._freeze_json(
            {"ledger_identity_sha256": identity_sha256({"parent": True})}
        ),
        runtime_ledger_sha256=PARENT_RUNTIME_SHA256,
        rows=tuple(rows),
    )


def _campaign_files(tmp_path: Path):
    population, eligibility, eligibility_sha, generation, generation_sha = (
        _sealed_campaign()
    )
    source = tmp_path / "closure-source"
    eligibility_artifact, _created = publish_sealed_json(
        source / "eligibility-manifest.json",
        eligibility,
    )
    generation_artifact, _created = publish_sealed_json(
        source / "retrieval-generation.json",
        generation,
    )
    assert eligibility_artifact.sha256 == eligibility_sha
    assert generation_artifact.sha256 == generation_sha
    return (
        population,
        _parent_plane(population),
        eligibility_artifact,
        generation_artifact,
    )


def _request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    arm_label: str = closure.REPRESENTATIVE_ARM,
) -> tuple[dict[str, Any], live.VerifiedS0V2AnswerPlane]:
    population, parent, eligibility, generation = _campaign_files(tmp_path)
    monkeypatch.setattr(
        closure_live,
        "_load_base_and_parent",
        lambda **_kwargs: (population, parent),
    )
    return {
        "arm_label": arm_label,
        "retrieval_path": tmp_path / "sealed-retrieval.json",
        "generation_path": generation.path,
        "expected_generation_sha256": generation.sha256,
        "eligibility_manifest_path": eligibility.path,
        "expected_eligibility_manifest_sha256": eligibility.sha256,
        "parent_root": tmp_path / "parent-s0-v2",
        "expected_parent_answer_run_sha256": PARENT_RUN_SHA256,
        "output_root": tmp_path / f"answers-{arm_label.lower()}",
        "max_concurrency": 2,
        "expected_retrieval_sha256": None,
        "expected_question_count": 100,
    }, parent


@pytest.mark.parametrize("arm_label", closure.ARM_LABELS)
def test_closure_answer_preflight_counts_only_added_descendants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm_label: str,
) -> None:
    request, _parent = _request(tmp_path, monkeypatch, arm_label=arm_label)

    artifact = closure_live.preflight_closure_answers(**request)

    payload = artifact.payload
    assert payload["arm_label"] == arm_label
    assert payload["question_count"] == 100
    assert payload["required_authorized_provider_calls"] == 1
    assert payload["logical_prompt_count"] == 1
    assert payload["unique_prompt_count"] == 1
    assert payload["provider_calls"] == 0
    assert payload["gold_loaded"] is False
    assert payload["hard_prompt_token_cap"] == 8_000
    assert payload["retained_request_token_state_bytes"] == 0
    rows = payload["ordered_rows"]
    assert rows[6]["provider_call_planned"] is True
    assert rows[6]["stage_disposition"] == "added"
    assert rows[5]["provider_call_planned"] is False
    assert rows[80]["provider_call_planned"] is False
    assert rows[90]["provider_call_planned"] is False


def test_closure_answer_authorization_fails_before_output_or_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _parent = _request(tmp_path, monkeypatch)
    output = Path(request["output_root"])
    monkeypatch.setattr(
        closure_live,
        "_make_provider_client",
        lambda *_args: pytest.fail("authorization failure built a client"),
    )

    with pytest.raises(MatchedEvalContractError, match="exactly equal 1"):
        closure_live.run_closure_answers(
            **request,
            enable_provider=True,
            authorized_provider_calls=0,
            api_key_env="MATCHED_TEST_KEY",
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("failure_kind", "disposition"),
    (("exception", "failed"), ("invalid_delta", "invalid")),
)
def test_closure_answer_refuses_failed_or_invalid_stage_before_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    disposition: str,
) -> None:
    request, _parent = _request(tmp_path, monkeypatch)
    output = Path(request["output_root"])

    if failure_kind == "exception":

        def raises_from_propose(_adapter, **_kwargs):
            raise RuntimeError("synthetic adapter failure")

        broken_propose = raises_from_propose
    else:

        def returns_invalid_delta(_adapter, **_kwargs):
            return object()

        broken_propose = returns_invalid_delta
    monkeypatch.setattr(
        closure.IndependentClosureMembershipAdapter,
        "propose",
        broken_propose,
    )
    monkeypatch.setattr(
        closure_live,
        "_make_provider_client",
        lambda *_args: pytest.fail("unsafe stage built a provider client"),
    )

    with pytest.raises(
        MatchedEvalContractError,
        match=f"refuses stage disposition {disposition}",
    ):
        closure_live.preflight_closure_answers(**request)
    assert not output.exists()

    with pytest.raises(
        MatchedEvalContractError,
        match=f"refuses stage disposition {disposition}",
    ):
        closure_live.run_closure_answers(
            **request,
            enable_provider=True,
            authorized_provider_calls=1,
            api_key_env="MATCHED_TEST_KEY",
        )
    assert not output.exists()


def test_closure_answer_run_reuses_parent_and_replays_without_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, parent = _request(tmp_path, monkeypatch)
    output = Path(request["output_root"])
    client = _FakeClient(output / closure_live.CHECKPOINT_DIR_NAME)
    monkeypatch.setenv("MATCHED_TEST_KEY", "test-key")
    monkeypatch.setattr(
        closure_live,
        "_make_provider_client",
        lambda *_args: client,
    )

    result = closure_live.run_closure_answers(
        **request,
        enable_provider=True,
        authorized_provider_calls=1,
        api_key_env="MATCHED_TEST_KEY",
    )

    assert result.physical_provider_calls == 1
    assert result.checkpoint_hits == 0
    assert len(client.chat.completions.requests) == 1
    payload = result.answer_artifact.payload
    assert payload["question_count"] == 100
    assert payload["added_descendant_count"] == 1
    assert payload["parent_fallback_count"] == 99
    assert payload["unique_provider_prompt_count"] == 1
    assert payload["retained_request_token_state_bytes"] == 0
    assert payload["questions"][5]["prediction"] == parent.rows[5].prediction
    assert payload["questions"][5]["provider_calls"] == 0
    assert payload["questions"][5]["call_key_sha256"] is None
    assert payload["questions"][6]["prediction_source"] == "terra_descendant"
    assert payload["questions"][6]["provider_calls"] == 1
    ledger = result.runtime_ledger_artifact.payload
    assert ledger["question_count"] == 100
    assert ledger["row_count"] == 200
    assert ledger["total_provider_calls"] == 1

    plan_request = dict(request)
    plan_request.pop("output_root")
    plan = closure_live._build_plan(**plan_request)
    source = plan.rows[5]
    overflow_trace = replace(
        source.stage.trace,
        disposition=StageDisposition.OVERFLOW,
        reason="synthetic_overflow_noop",
    )
    overflow_stage = replace(
        source.stage,
        receipt=replace(source.stage.receipt, trace=overflow_trace),
    )
    overflow_row = replace(
        source,
        run=replace(source.run, stages=(overflow_stage,)),
        stage=overflow_stage,
    )
    overflow_plan = replace(
        plan,
        rows=(
            *plan.rows[:5],
            overflow_row,
            *plan.rows[6:],
        ),
    )
    replay_batch = closure_live._runtime(
        overflow_plan,
        checkpoint_dir=output / closure_live.CHECKPOINT_DIR_NAME,
        client=None,
        max_concurrency=2,
        preflight_artifact_sha256=read_sealed_json(
            output / closure_live.ANSWER_PREFLIGHT_NAME
        ).sha256,
    ).run()
    overflow_payload = closure_live._answer_artifact(
        overflow_plan,
        replay_batch,
        preflight_artifact_sha256=read_sealed_json(
            output / closure_live.ANSWER_PREFLIGHT_NAME
        ).sha256,
    )
    assert overflow_payload["questions"][5]["stage_disposition"] == "overflow"
    assert overflow_payload["questions"][5]["prediction"] == parent.rows[5].prediction
    assert overflow_payload["questions"][5]["provider_calls"] == 0

    monkeypatch.setattr(
        closure_live,
        "_make_provider_client",
        lambda *_args: pytest.fail("replay built a provider client"),
    )
    replay = closure_live.replay_closure_answers(
        **request,
        expected_run_sha256=result.answer_artifact.sha256,
    )

    assert replay.run_sha256 == replay.replay_sha256
    assert replay.parent_plane is parent
    assert replay.matched_population_id == parent.matched_population_id
    assert replay.rows[5].prediction == parent.rows[5].prediction
    assert replay.rows[5].prediction_source == "sealed_parent_fallback"
    assert replay.rows[6].prediction_source == "terra_descendant"
    assert replay.rows[6].messages_sha256 == (
        replay.rows[6].final_prompt_messages_sha256
    )
    assert read_sealed_json(
        output / closure_live.RUNTIME_LEDGER_REPLAY_NAME
    ).sha256 == result.runtime_ledger_artifact.sha256

    resumed = closure_live.run_closure_answers(
        **request,
        enable_provider=True,
        authorized_provider_calls=1,
        api_key_env="MATCHED_TEST_KEY",
    )
    assert resumed.answer_artifact.sha256 == result.answer_artifact.sha256
    assert resumed.physical_provider_calls == 0
    assert resumed.checkpoint_hits == 1


def _closure_cli_args(command: str, arm_label: str) -> list[str]:
    args = [
        command,
        "--arm",
        arm_label,
        "--expected-generation-sha256",
        "3" * 64,
        "--expected-eligibility-manifest-sha256",
        "4" * 64,
        "--expected-parent-answer-run-sha256",
        "5" * 64,
    ]
    if command == "closure-answer-replay":
        args.extend(("--expected-run-sha256", "6" * 64))
    return args


def test_closure_cli_defaults_to_separate_arm_roots() -> None:
    parser = spine_cli._parser()
    representative = parser.parse_args(
        _closure_cli_args(
            "closure-answer-preflight",
            closure.REPRESENTATIVE_ARM,
        )
    )
    global_arm = parser.parse_args(
        _closure_cli_args("closure-answer-preflight", closure.GLOBAL_ARM)
    )

    representative_root, _request_values = spine_cli._closure_answer_request(
        representative
    )
    global_root, _request_values = spine_cli._closure_answer_request(global_arm)
    assert representative_root == (
        spine_cli.DEFAULT_CLOSURE_REPRESENTATIVE_ANSWER_ROOT
    )
    assert global_root == spine_cli.DEFAULT_CLOSURE_GLOBAL_ANSWER_ROOT
    assert representative_root != global_root


@pytest.mark.parametrize(
    ("command", "function_name"),
    (
        ("closure-answer-preflight", "preflight_closure_answers"),
        ("closure-answer", "run_closure_answers"),
        ("closure-answer-replay", "replay_closure_answers"),
    ),
)
def test_closure_cli_routes_all_answer_phases(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    function_name: str,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake(**kwargs):
        calls.append(kwargs)
        if function_name == "preflight_closure_answers":
            return SimpleNamespace(
                sha256="7" * 64,
                payload={
                    "arm_label": closure.REPRESENTATIVE_ARM,
                    "required_authorized_provider_calls": 1,
                },
            )
        if function_name == "run_closure_answers":
            return SimpleNamespace(
                answer_artifact=SimpleNamespace(
                    sha256="8" * 64,
                    payload={"arm_label": closure.REPRESENTATIVE_ARM},
                ),
                runtime_ledger_artifact=SimpleNamespace(sha256="9" * 64),
                checkpoint_hits=0,
                physical_provider_calls=1,
            )
        return SimpleNamespace(
            arm_label=closure.REPRESENTATIVE_ARM,
            run_sha256="8" * 64,
            replay_sha256="8" * 64,
            runtime_ledger_sha256="9" * 64,
        )

    monkeypatch.setattr(closure_live, function_name, fake)
    args = _closure_cli_args(command, closure.REPRESENTATIVE_ARM)
    if command == "closure-answer":
        args.extend(("--enable-provider", "--authorized-provider-calls", "1"))

    assert spine_cli.main(args) == 0
    assert len(calls) == 1
    assert calls[0]["arm_label"] == closure.REPRESENTATIVE_ARM
    assert calls[0]["output_root"] == (
        spine_cli.DEFAULT_CLOSURE_REPRESENTATIVE_ANSWER_ROOT
    )
    assert "physical_provider_calls=" in capsys.readouterr().out


def _closure_judge_cli_args(
    command: str,
    *,
    output_root: Path,
    answer_run_sha256: str,
    enable_provider: bool = True,
    authorized_provider_calls: int = 1,
) -> list[str]:
    args = _closure_cli_args(command, closure.REPRESENTATIVE_ARM)
    args.extend(
        (
            "--dataset",
            str(output_root / "dataset.json"),
            "--output-root",
            str(output_root),
            "--expected-answer-run-sha256",
            answer_run_sha256,
            "--expected-parent-judge-sha256",
            "a" * 64,
            "--expected-parent-score-ledger-sha256",
            "b" * 64,
        )
    )
    if command == "closure-judge":
        if enable_provider:
            args.append("--enable-provider")
        args.extend(
            (
                "--authorized-provider-calls",
                str(authorized_provider_calls),
            )
        )
    elif command == "closure-judge-replay":
        args.extend(("--expected-judge-sha256", "c" * 64))
    return args


def _seal_closure_answer_replays(output_root: Path) -> str:
    answer, _created = publish_sealed_json(
        output_root / closure_live.ANSWER_REPLAY_NAME,
        {"kind": "closure-answer-replay"},
    )
    ledger, _created = publish_sealed_json(
        output_root / closure_live.RUNTIME_LEDGER_NAME,
        {"kind": "closure-runtime-ledger"},
    )
    replay, _created = publish_sealed_json(
        output_root / closure_live.RUNTIME_LEDGER_REPLAY_NAME,
        ledger.payload,
    )
    assert replay.sha256 == ledger.sha256
    return answer.sha256


def test_closure_judge_cli_parser_uses_answer_and_parent_judge_defaults(
    tmp_path: Path,
) -> None:
    answer_sha256 = "d" * 64
    parsed = spine_cli._parser().parse_args(
        _closure_judge_cli_args(
            "closure-judge-preflight",
            output_root=tmp_path / "representative",
            answer_run_sha256=answer_sha256,
        )
    )

    assert parsed.arm == closure.REPRESENTATIVE_ARM
    assert parsed.expected_answer_run_sha256 == answer_sha256
    assert parsed.parent_root == spine_cli.DEFAULT_S0_V2_ROOT
    assert parsed.parent_judge_root == spine_cli.DEFAULT_S0_V2_ROOT
    assert parsed.retrieval == spine_cli.DEFAULT_RETRIEVAL
    assert parsed.generation == spine_cli.DEFAULT_CLOSURE_GENERATION
    assert parsed.eligibility_manifest == spine_cli.DEFAULT_CLOSURE_ELIGIBILITY
    assert parsed.split == spine_cli.DEFAULT_SPLIT
    assert parsed.max_concurrency == 4


@pytest.mark.parametrize(
    ("command", "function_name"),
    (
        (
            "closure-judge-preflight",
            "preflight_closure_changed_only_judge",
        ),
        ("closure-judge", "run_closure_changed_only_judge"),
        (
            "closure-judge-replay",
            "replay_closure_changed_only_judge",
        ),
    ),
)
def test_closure_judge_cli_replays_answer_then_dispatches_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    function_name: str,
) -> None:
    output = tmp_path / command
    answer_sha256 = _seal_closure_answer_replays(output)
    answer_plane = SimpleNamespace(arm_label=closure.REPRESENTATIVE_ARM)
    events: list[tuple[str, dict[str, Any]]] = []

    def fake_answer_replay(**kwargs):
        events.append(("answer_replay", kwargs))
        return answer_plane

    def fake_judge(**kwargs):
        events.append(("judge", kwargs))
        if function_name == "preflight_closure_changed_only_judge":
            return SimpleNamespace(
                sha256="e" * 64,
                payload={
                    "arm_label": closure.REPRESENTATIVE_ARM,
                    "required_authorized_provider_calls": 1,
                },
            )
        return SimpleNamespace(
            judge_artifact=SimpleNamespace(
                sha256="f" * 64,
                payload={"arm_label": closure.REPRESENTATIVE_ARM},
            ),
            score_ledger_artifact=SimpleNamespace(sha256="0" * 64),
            checkpoint_hits=1,
            correct=60,
            physical_provider_calls=(
                1 if function_name == "run_closure_changed_only_judge" else 0
            ),
        )

    monkeypatch.setattr(closure_live, "replay_closure_answers", fake_answer_replay)
    monkeypatch.setattr(closure_judging, function_name, fake_judge)

    assert spine_cli.main(
        _closure_judge_cli_args(
            command,
            output_root=output,
            answer_run_sha256=answer_sha256,
        )
    ) == 0
    assert [event for event, _kwargs in events] == ["answer_replay", "judge"]
    answer_request = events[0][1]
    judge_request = events[1][1]
    assert answer_request["expected_run_sha256"] == answer_sha256
    assert answer_request["output_root"] == output
    assert judge_request["answer_plane"] is answer_plane
    assert judge_request["output_root"] == output
    assert judge_request["parent_judge_root"] == spine_cli.DEFAULT_S0_V2_ROOT
    assert judge_request["expected_parent_judge_sha256"] == "a" * 64
    assert judge_request["expected_parent_score_ledger_sha256"] == "b" * 64
    if command == "closure-judge-preflight":
        assert "max_concurrency" not in judge_request
    elif command == "closure-judge":
        assert judge_request["enable_provider"] is True
        assert judge_request["authorized_provider_calls"] == 1
        assert judge_request["max_concurrency"] == 4
    elif command == "closure-judge-replay":
        assert judge_request["expected_judge_sha256"] == "c" * 64
        assert judge_request["max_concurrency"] == 4
    assert "physical_provider_calls=" in capsys.readouterr().out


@pytest.mark.parametrize(
    "command",
    (
        "closure-judge-preflight",
        "closure-judge",
        "closure-judge-replay",
    ),
)
def test_closure_judge_cli_matches_real_public_api_signatures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    output = tmp_path / f"real-api-{command}"
    answer_sha256 = _seal_closure_answer_replays(output)
    answer_plane = SimpleNamespace(arm_label=closure.REPRESENTATIVE_ARM)
    plan = SimpleNamespace(required_calls=0)

    monkeypatch.setattr(
        closure_live,
        "replay_closure_answers",
        lambda **_kwargs: answer_plane,
    )
    monkeypatch.setattr(
        closure_judging,
        "_build_plan",
        lambda **_kwargs: plan,
    )
    monkeypatch.setattr(
        closure_judging,
        "_preflight_artifact",
        lambda _plan: {
            "arm_label": closure.REPRESENTATIVE_ARM,
            "required_authorized_provider_calls": 0,
        },
    )
    monkeypatch.setattr(
        closure_judging,
        "_judge_artifact",
        lambda *_args, **_kwargs: {
            "aggregate": {"correct": 60},
            "arm_label": closure.REPRESENTATIVE_ARM,
        },
    )
    monkeypatch.setattr(
        closure_judging,
        "_score_ledger",
        lambda *_args, **_kwargs: {"format": "synthetic-score-ledger"},
    )
    replay_result = SimpleNamespace(
        judge_artifact=SimpleNamespace(
            sha256="f" * 64,
            payload={"arm_label": closure.REPRESENTATIVE_ARM},
        ),
        score_ledger_artifact=SimpleNamespace(sha256="0" * 64),
        checkpoint_hits=0,
        correct=60,
        physical_provider_calls=0,
    )
    monkeypatch.setattr(
        closure_judging,
        "_replay_plan",
        lambda *_args, **_kwargs: replay_result,
    )
    monkeypatch.setattr(
        closure_judging.judging,
        "_make_provider_client",
        lambda *_args: pytest.fail("signature test reached a provider client"),
    )

    assert spine_cli.main(
        _closure_judge_cli_args(
            command,
            output_root=output,
            answer_run_sha256=answer_sha256,
            enable_provider=False,
            authorized_provider_calls=0,
        )
    ) == 0


def test_closure_judge_cli_requires_existing_answer_replays_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "missing-replay"
    monkeypatch.setattr(
        closure_live,
        "replay_closure_answers",
        lambda **_kwargs: pytest.fail("missing replay reached answer replay"),
    )
    monkeypatch.setattr(
        closure_judging,
        "run_closure_changed_only_judge",
        lambda **_kwargs: pytest.fail("missing replay reached judge dispatch"),
    )

    with pytest.raises(SealedArtifactError, match="answer-run-replay.json"):
        spine_cli.main(
            _closure_judge_cli_args(
                "closure-judge",
                output_root=output,
                answer_run_sha256="d" * 64,
            )
        )

    assert not output.exists()
