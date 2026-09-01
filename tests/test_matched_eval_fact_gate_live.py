from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.fast_completion_runtime import (
    preflight_fast_completion_prompts,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    FastProviderMessage,
)
from tests.test_fast_completion_runtime import _FakeClient
from tests.test_matched_eval_closure import _sealed_campaign
from tests.test_matched_eval_closure_live import _parent_plane
from tools import run_matched_eval_spine as spine_cli
from tools.matched_eval import fact_gate_live
from tools.matched_eval.contracts import (
    ArtifactRef,
    MatchedEvalContractError,
    identity_sha256,
)
from tools.matched_eval.fact_gate import FactGateResult, MAX_PROMPT_TOKENS


def _synthetic_plan() -> fact_gate_live._FactGateAnswerPlan:
    population, *_rest = _sealed_campaign()
    parent = _parent_plane(population)
    policy_sha = "3" * 64
    em_run_sha = "4" * 64
    compression_sha = "5" * 64
    snapshot = replace(
        population.snapshot,
        overlay_revisions=(
            *population.snapshot.overlay_revisions,
            ArtifactRef(role="fact_gate_em_run", sha256=em_run_sha),
            ArtifactRef(role="fact_gate_em_compression", sha256=compression_sha),
            ArtifactRef(role="fact_gate_route_policy", sha256=policy_sha),
        ),
        policy_id="matched_fact_gate_positive_cells_v1",
        renderer_id=fact_gate_live.RENDERER_ID,
        implementation_id="tools_matched_eval_fact_gate_v1",
    )
    rows = []
    prompt_mappings: tuple[dict[str, str], ...] | None = None
    for source, parent_row in zip(population.rows, parent.rows, strict=True):
        submitted = source.ordinal == 6
        prompt = None
        admitted: tuple[str, ...] = ()
        selected: tuple[str, ...] = ()
        if submitted:
            messages = tuple(
                FastProviderMessage(
                    role=str(message["role"]),
                    content=str(message["content"]),
                )
                for message in source.rendered_prompt.messages
            )
            prompt_mappings = tuple(
                {"role": row.role, "content": row.content} for row in messages
            )
            admitted = (identity_sha256({"delta": source.ordinal}),)
            selected = admitted
            prompt = SimpleNamespace(
                messages=messages,
                messages_sha256=identity_sha256(list(prompt_mappings)),
                prompt_token_proxy=source.rendered_prompt.total_prompt_token_proxy,
            )
        gate = FactGateResult(
            adapter_id="fixed_s1_em_fact_memory_v1",
            question_id=source.packet.question_id,
            dated_question_sha256=source.packet.dated_question_sha256,
            route_id="numeric_reduce" if submitted else "point_lookup",
            route_admitted=submitted,
            route_policy_id="matched_fact_gate_positive_cells_v1",
            route_policy_sha256=policy_sha,
            route_reason="synthetic_question_only_route",
            route_receipt_sha256=identity_sha256(
                {"route": source.ordinal}
            ),
            disposition="compiled" if submitted else "parent_fallback",
            reason=(
                "positive_cell_exact_cited_fact_delta"
                if submitted
                else "question_route_not_admitted"
            ),
            parent_prediction=parent_row.prediction,
            protected_evidence_ids=tuple(
                row.evidence_id for row in source.packet.protected_evidence
            ),
            selected_evidence_ids_before_dedup=selected,
            dedup_excluded_evidence_ids=(),
            admitted_delta_evidence_ids=admitted,
            facts=(),
            compression_receipt_sha256="6" * 64 if submitted else None,
            source_representation_messages_sha256=(
                None if prompt is None else prompt.messages_sha256
            ),
            prompt=prompt,
        )
        rows.append(
            fact_gate_live._FactGatePlanRow(
                source=source,
                parent=parent_row,
                em_question=SimpleNamespace(question_id=source.packet.question_id),
                gate=gate,
            )
        )
    assert prompt_mappings is not None
    prompt_population = preflight_fast_completion_prompts(
        [prompt_mappings],
        max_prompt_tokens=MAX_PROMPT_TOKENS,
    )
    return fact_gate_live._FactGateAnswerPlan(
        population=population,
        parent_plane=parent,
        snapshot=snapshot,
        rows=tuple(rows),
        prompt_population=prompt_population,
        em_run_sha256=em_run_sha,
        compression_sha256=compression_sha,
        historical_population_identity_sha256="7" * 64,
        route_policy_sha256=policy_sha,
    )


def _request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    plan = _synthetic_plan()
    monkeypatch.setattr(fact_gate_live, "_build_plan", lambda **_kwargs: plan)
    return {
        "retrieval_path": tmp_path / "retrieval.json",
        "baseline_answers_path": tmp_path / "fixed-s1-answers.json",
        "expected_baseline_answers_sha256": "8" * 64,
        "em_run_path": tmp_path / "historical-em" / "run.json",
        "expected_em_run_sha256": plan.em_run_sha256,
        "parent_root": tmp_path / "parent-s0-v2",
        "expected_parent_answer_run_sha256": plan.parent_plane.run_sha256,
        "output_root": tmp_path / "fact-gate-answers",
        "max_concurrency": 2,
        "expected_retrieval_sha256": None,
        "expected_question_count": 100,
    }


def test_fact_gate_preflight_seals_exact_changed_only_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path, monkeypatch)

    artifact = fact_gate_live.preflight_fact_gate_answers(**request)

    payload = artifact.payload
    assert payload["question_count"] == 100
    assert payload["required_authorized_provider_calls"] == 1
    assert payload["logical_prompt_count"] == 1
    assert payload["unique_prompt_count"] == 1
    assert payload["provider_calls"] == 0
    assert payload["gold_loaded"] is False
    assert payload["construction_recall_claimed"] is False
    assert payload["source_target_expansion_claimed"] is False
    assert payload["retained_request_token_state_bytes"] == 0
    assert payload["ordered_rows"][6]["provider_call_planned"] is True
    assert payload["ordered_rows"][5]["provider_call_planned"] is False


def test_fact_gate_authorization_fails_before_output_or_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path, monkeypatch)
    output = Path(request["output_root"])
    monkeypatch.setattr(
        fact_gate_live,
        "_make_provider_client",
        lambda *_args: pytest.fail("authorization failure built a client"),
    )

    with pytest.raises(MatchedEvalContractError, match="exactly equal 1"):
        fact_gate_live.run_fact_gate_answers(
            **request,
            enable_provider=True,
            authorized_provider_calls=0,
            api_key_env="MATCHED_TEST_KEY",
        )

    assert not output.exists()


def test_fact_gate_run_parent_fallback_and_zero_call_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path, monkeypatch)
    plan = fact_gate_live._build_plan()
    output = Path(request["output_root"])
    client = _FakeClient(output / fact_gate_live.CHECKPOINT_DIR_NAME)
    monkeypatch.setenv("MATCHED_TEST_KEY", "test-key")
    monkeypatch.setattr(
        fact_gate_live,
        "_make_provider_client",
        lambda *_args: client,
    )

    result = fact_gate_live.run_fact_gate_answers(
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
    assert payload["submitted_fact_gate_count"] == 1
    assert payload["parent_fallback_count"] == 99
    assert payload["questions"][5]["prediction"] == plan.parent_plane.rows[5].prediction
    assert payload["questions"][5]["prediction_source"] == "sealed_parent_fallback"
    assert payload["questions"][5]["provider_calls"] == 0
    assert payload["questions"][6]["prediction_source"] == "terra_fact_gate"
    assert payload["questions"][6]["provider_calls"] == 1
    ledger = result.runtime_ledger_artifact.payload
    assert ledger["question_count"] == 100
    assert ledger["row_count"] == 200
    assert ledger["total_provider_calls"] == 1

    monkeypatch.setattr(
        fact_gate_live,
        "_make_provider_client",
        lambda *_args: pytest.fail("replay built a provider client"),
    )
    replay = fact_gate_live.replay_fact_gate_answers(
        **request,
        expected_run_sha256=result.answer_artifact.sha256,
    )

    assert replay.run_sha256 == replay.replay_sha256
    assert replay.parent_plane is plan.parent_plane
    assert replay.rows[5].prediction == plan.parent_plane.rows[5].prediction
    assert replay.rows[5].changed_from_parent is False
    assert replay.rows[6].prediction_source == "terra_fact_gate"
    assert replay.rows[6] in replay.changed_rows
    assert len(replay.runtime_ledger["rows"]) == 200

    resumed = fact_gate_live.run_fact_gate_answers(
        **request,
        enable_provider=True,
        authorized_provider_calls=1,
        api_key_env="MATCHED_TEST_KEY",
    )
    assert resumed.answer_artifact.sha256 == result.answer_artifact.sha256
    assert resumed.physical_provider_calls == 0
    assert resumed.checkpoint_hits == 1


def _cli_args(command: str) -> list[str]:
    args = [
        command,
        "--expected-em-run-sha256",
        "a" * 64,
        "--expected-parent-answer-run-sha256",
        "b" * 64,
    ]
    if command == "fact-gate-answer":
        args.extend(("--enable-provider", "--authorized-provider-calls", "25"))
    elif command == "fact-gate-answer-replay":
        args.extend(("--expected-run-sha256", "c" * 64))
    return args


def test_fact_gate_cli_defaults_pin_inputs_and_separate_output() -> None:
    parsed = spine_cli._parser().parse_args(
        _cli_args("fact-gate-answer-preflight")
    )
    root, request = spine_cli._fact_gate_answer_request(parsed)

    assert root == spine_cli.DEFAULT_FACT_GATE_ANSWER_ROOT
    assert root != spine_cli.DEFAULT_S0_V2_ROOT
    assert request["em_run_path"] == spine_cli.DEFAULT_FACT_GATE_EM_RUN
    assert request["parent_root"] == spine_cli.DEFAULT_S0_V2_ROOT
    assert request["baseline_answers_path"] == (
        spine_cli.DEFAULT_FIXED_S1_BASELINE_ANSWERS
    )
    assert request["expected_baseline_answers_sha256"] == (
        spine_cli.DEFAULT_FIXED_S1_BASELINE_ANSWERS_SHA256
    )


@pytest.mark.parametrize(
    ("command", "function_name"),
    (
        ("fact-gate-answer-preflight", "preflight_fact_gate_answers"),
        ("fact-gate-answer", "run_fact_gate_answers"),
        ("fact-gate-answer-replay", "replay_fact_gate_answers"),
    ),
)
def test_fact_gate_cli_routes_all_answer_phases(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    function_name: str,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake(**kwargs):
        calls.append(kwargs)
        if function_name == "preflight_fact_gate_answers":
            return SimpleNamespace(
                sha256="d" * 64,
                payload={
                    "arm_label": fact_gate_live.ARM_LABEL,
                    "required_authorized_provider_calls": 25,
                },
            )
        if function_name == "run_fact_gate_answers":
            return SimpleNamespace(
                answer_artifact=SimpleNamespace(
                    sha256="e" * 64,
                    payload={"arm_label": fact_gate_live.ARM_LABEL},
                ),
                runtime_ledger_artifact=SimpleNamespace(sha256="f" * 64),
                checkpoint_hits=0,
                physical_provider_calls=25,
            )
        return SimpleNamespace(
            arm_label=fact_gate_live.ARM_LABEL,
            run_sha256="e" * 64,
            replay_sha256="e" * 64,
            runtime_ledger_sha256="f" * 64,
            changed_rows=(),
        )

    monkeypatch.setattr(fact_gate_live, function_name, fake)

    assert spine_cli.main(_cli_args(command)) == 0
    assert len(calls) == 1
    assert calls[0]["output_root"] == spine_cli.DEFAULT_FACT_GATE_ANSWER_ROOT
    assert calls[0]["expected_em_run_sha256"] == "a" * 64
    assert calls[0]["expected_parent_answer_run_sha256"] == "b" * 64
    assert "physical_provider_calls=" in capsys.readouterr().out
