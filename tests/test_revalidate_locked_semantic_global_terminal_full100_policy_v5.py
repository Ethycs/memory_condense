from __future__ import annotations

import argparse
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import (
    revalidate_locked_semantic_global_terminal_full100_policy_v5 as policy,
)
from tools.matched_eval.artifacts import SealedArtifact, read_sealed_json
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.typed_memory_final_arm import render_final_messages


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sealed_row(body: dict[str, Any], receipt_key: str) -> dict[str, Any]:
    return {**body, receipt_key: identity_sha256(body)}


def _provider_input(ordinal: int) -> dict[str, Any]:
    return {
        "dated_question": (
            f"[Question asked at 2026-08-{ordinal % 28 + 1:02d}] "
            f"What did item {ordinal} resolve to?"
        ),
        "typed_evidence": {"handles": []},
    }


def _terminal_plan(ordinal: int, *, parent: str | None = None) -> dict[str, Any]:
    parent = parent or f"parent-{ordinal}"
    provider_input = _provider_input(ordinal)
    messages = [dict(row) for row in render_final_messages(provider_input)]
    body = {
        "allowed_handle_ids": ["H001"],
        "answer_mode": "semantic_global_terminal",
        "dated_question_sha256": quote_sha256(provider_input["dated_question"]),
        "format": "synthetic-prompt-row-v1",
        "handle_group_by_id": {"H001": "G001"},
        "messages": messages,
        "messages_sha256": identity_sha256(messages),
        "ordinal": ordinal,
        "parent_prediction": parent,
        "parent_prediction_sha256": quote_sha256(parent),
        "provider_input_sha256": identity_sha256(provider_input),
        "question_id": f"q-{ordinal}",
        "question_sha256": _sha(f"question-{ordinal}"),
        "route_id": "synthetic-terminal-route",
    }
    return _sealed_row(body, "prompt_row_receipt_sha256")


def _passthrough_plan(ordinal: int, *, parent: str | None = None) -> dict[str, Any]:
    parent = parent or f"parent-{ordinal}"
    body = {
        "answer_mode": "v3_passthrough",
        "dated_question_sha256": _sha(f"dated-{ordinal}"),
        "format": "synthetic-passthrough-row-v1",
        "ordinal": ordinal,
        "parent_prediction": parent,
        "parent_prediction_sha256": quote_sha256(parent),
        "prediction": parent,
        "prediction_sha256": quote_sha256(parent),
        "question_id": f"q-{ordinal}",
        "question_sha256": _sha(f"question-{ordinal}"),
        "route_id": "synthetic-passthrough-route",
    }
    return _sealed_row(body, "passthrough_plan_receipt_sha256")


def _source_question(
    plan: dict[str, Any], *, prediction: str | None = None
) -> dict[str, Any]:
    parent = str(plan["parent_prediction"])
    prediction = prediction or parent
    body = {
        "dated_question_sha256": plan["dated_question_sha256"],
        "format": "synthetic-source-result-v1",
        "ordinal": plan["ordinal"],
        "parent_prediction_sha256": quote_sha256(parent),
        "prediction": prediction,
        "prediction_sha256": quote_sha256(prediction),
        "question_id": plan["question_id"],
        "question_sha256": plan["question_sha256"],
    }
    return _sealed_row(body, "source_row_sha256")


def _record(plan: dict[str, Any], completion: str) -> dict[str, Any]:
    ordinal = int(plan["ordinal"])
    return {
        "call_key_sha256": _sha(f"call-{ordinal}"),
        "checkpoint_hit": True,
        "completion": completion,
        "completion_sha256": quote_sha256(completion),
        "messages_sha256": plan["messages_sha256"],
        "physical_call": False,
        "request_journal_sha256": _sha(f"request-{ordinal}"),
        "response_journal_sha256": _sha(f"response-{ordinal}"),
    }


def _numeric_projection(
    *, status: str, prediction: str = "", used: list[str] | None = None
) -> dict[str, Any]:
    used = used or []
    body = {
        "decision": "replace" if status == "supported" else "abstain",
        "format": policy.NUMERIC_POLICY_FORMAT,
        "prediction": prediction,
        "provider_prompt_count": 0,
        "reason": "synthetic_numeric_fixture",
        "retained_transformer_token_state_bytes": 0,
        "status": status,
        "used_handle_ids": used,
    }
    return {**body, "receipt_sha256": identity_sha256(body)}


def _v5_proof(
    plan: dict[str, Any],
    completion: str,
    *,
    accepted: bool,
    prediction: str,
    error_code: str | None = None,
) -> dict[str, Any]:
    used = ["H001"] if accepted else []
    body = {
        "accepted_replacement": accepted,
        "completion_sha256": quote_sha256(completion),
        "decision": "replace" if accepted else "keep_parent",
        "error_code": error_code,
        "final_prediction": prediction,
        "final_prediction_sha256": quote_sha256(prediction),
        "format": policy._V5_PROOF_FORMAT,  # noqa: SLF001
        "gold_loaded": False,
        "parent_prediction_sha256": quote_sha256(plan["parent_prediction"]),
        "physical_provider_calls": 0,
        "prompt_row_receipt_sha256": plan["prompt_row_receipt_sha256"],
        "retained_transformer_token_state_bytes": 0,
        "used_handle_ids": used,
        "validator_policy_format": policy.VALIDATOR_POLICY_FORMAT,
    }
    return {**body, "policy_proof_receipt_sha256": identity_sha256(body)}


def _small_source(
    terminal: dict[str, Any],
    passthrough: dict[str, Any],
    completion: str,
    *,
    terminal_source_prediction: str = "terra-candidate",
) -> SealedArtifact:
    record = _record(terminal, completion)
    payload = {
        "completion_batch": {
            "logical_completions": [completion],
            "unique_records": [record],
            "usage": {
                "checkpoint_hits": 1,
                "logical_calls": 1,
                "physical_calls": 0,
                "unique_calls": 1,
            },
        },
        "questions": [
            _source_question(terminal, prediction=terminal_source_prediction),
            _source_question(passthrough),
        ],
    }
    return SealedArtifact(Path("source.json"), _sha("small-source"), payload)


def test_numeric_supported_has_priority_and_passthrough_calls_no_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = _terminal_plan(0)
    passthrough = _passthrough_plan(1)
    completion = "sealed terra completion"
    source = _small_source(terminal, passthrough, completion)
    calls = {"numeric": 0, "v5": 0}

    def numeric(
        _provider_input: dict[str, Any], _frontier: object = None
    ) -> dict[str, Any]:
        calls["numeric"] += 1
        return _numeric_projection(
            status="supported", prediction="numeric-answer", used=["H001"]
        )

    def v5(plan: dict[str, Any], value: str) -> dict[str, Any]:
        calls["v5"] += 1
        return _v5_proof(
            plan, value, accepted=True, prediction="terra-answer"
        )

    monkeypatch.setattr(policy, "_numeric_policy_projection", numeric)
    monkeypatch.setattr(policy, "_replacement_policy_proof", v5)
    rows = policy._build_overlay_rows(  # noqa: SLF001
        source, [terminal], [passthrough], expected_ordinals=(0, 1)
    )

    assert calls == {"numeric": 1, "v5": 1}
    assert rows[0]["prediction"] == "numeric-answer"
    assert rows[0]["selected_policy"] == "operator_first_numeric"
    assert rows[0]["used_handle_ids"] == ["H001"]
    assert rows[1]["prediction"] == passthrough["parent_prediction"]
    assert rows[1]["selected_policy"] == "passthrough"
    assert rows[1]["numeric_policy_proof"] is None
    assert rows[1]["policy_v5_proof"] is None


def test_q54_style_exact_day_abstention_fill_survives_numeric_fallthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = _terminal_plan(0, parent="I don't know")
    passthrough = _passthrough_plan(1)
    completion = "sealed q54 completion"
    source = _small_source(
        terminal,
        passthrough,
        completion,
        terminal_source_prediction="August 14, 2026",
    )
    monkeypatch.setattr(
        policy,
        "_numeric_policy_projection",
        lambda _provider_input, _frontier=None: _numeric_projection(
            status="insufficient"
        ),
    )
    monkeypatch.setattr(
        policy,
        "_replacement_policy_proof",
        lambda plan, value: _v5_proof(
            plan, value, accepted=True, prediction="August 14, 2026"
        ),
    )

    rows = policy._build_overlay_rows(  # noqa: SLF001
        source, [terminal], [passthrough], expected_ordinals=(0, 1)
    )

    assert rows[0]["prediction"] == "August 14, 2026"
    assert rows[0]["selected_policy"] == "typed_final_validator_v5"
    assert rows[0]["changed_from_parent"] is True
    assert rows[0]["changed_from_source_answer"] is False


@pytest.mark.parametrize(
    "error_code",
    ("conflict_neighborhood_incomplete", "material_claim_unsupported"),
)
def test_q5_q36_style_rejections_keep_exact_parent(
    monkeypatch: pytest.MonkeyPatch, error_code: str
) -> None:
    terminal = _terminal_plan(0, parent="protected preference")
    passthrough = _passthrough_plan(1)
    completion = f"sealed rejected completion: {error_code}"
    source = _small_source(terminal, passthrough, completion)
    monkeypatch.setattr(
        policy,
        "_numeric_policy_projection",
        lambda _provider_input, _frontier=None: _numeric_projection(
            status="insufficient"
        ),
    )
    monkeypatch.setattr(
        policy,
        "_replacement_policy_proof",
        lambda plan, value: _v5_proof(
            plan,
            value,
            accepted=False,
            prediction=str(plan["parent_prediction"]),
            error_code=error_code,
        ),
    )

    row = policy._build_overlay_rows(  # noqa: SLF001
        source, [terminal], [passthrough], expected_ordinals=(0, 1)
    )[0]

    assert row["prediction"] == terminal["parent_prediction"]
    assert row["prediction"] != "terra-candidate"
    assert row["selected_policy"] == "protected_parent"
    assert row["changed_from_parent"] is False
    assert row["changed_from_source_answer"] is True


def test_provider_input_must_match_canonical_sealed_message() -> None:
    plan = _terminal_plan(0)
    assert policy.authenticated_provider_input(plan) == _provider_input(0)

    tampered = deepcopy(plan)
    tampered["messages"][1]["content"] += " "
    unsigned = dict(tampered)
    unsigned.pop("prompt_row_receipt_sha256")
    tampered["prompt_row_receipt_sha256"] = identity_sha256(unsigned)
    tampered["messages_sha256"] = identity_sha256(tampered["messages"])
    unsigned = dict(tampered)
    unsigned.pop("prompt_row_receipt_sha256")
    tampered["prompt_row_receipt_sha256"] = identity_sha256(unsigned)

    with pytest.raises(
        policy.LockedSemanticGlobalTerminalFull100PolicyV5Error,
        match="not strict JSON|not canonical|differs from its sealed prompt",
    ):
        policy.authenticated_provider_input(tampered)


def test_completion_batch_tampering_fails_closed() -> None:
    terminal = _terminal_plan(0)
    passthrough = _passthrough_plan(1)
    source = _small_source(terminal, passthrough, "sealed completion")
    payload = deepcopy(source.payload)
    payload["completion_batch"]["unique_records"][0]["physical_call"] = True
    changed = SealedArtifact(source.path, _sha("changed"), payload)

    with pytest.raises(
        policy.LockedSemanticGlobalTerminalFull100PolicyV5Error,
        match="completion record changed",
    ):
        policy._validated_completion_records(changed, [terminal])  # noqa: SLF001


def _full_source(
    *, source_prediction_by_ordinal: dict[int, str] | None = None
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    source_prediction_by_ordinal = source_prediction_by_ordinal or {}
    terminals = [_terminal_plan(ordinal) for ordinal in range(policy.TERMINAL_COUNT)]
    passthroughs = [
        _passthrough_plan(ordinal)
        for ordinal in range(policy.TERMINAL_COUNT, policy.QUESTION_COUNT)
    ]
    completions = [f"completion-{ordinal}" for ordinal in range(policy.TERMINAL_COUNT)]
    records = [
        _record(plan, completion)
        for plan, completion in zip(terminals, completions, strict=True)
    ]
    preflight = SealedArtifact(
        Path("preflight.json"),
        _sha("full-preflight"),
        {
            "eligible_ordinals": list(range(policy.TERMINAL_COUNT)),
            "passthrough_ordinals": list(
                range(policy.TERMINAL_COUNT, policy.QUESTION_COUNT)
            ),
            "source_question_population_sha256": _sha("question-population"),
        },
    )
    source_run = SealedArtifact(
        Path("run.json"),
        _sha("full-run"),
        {
            "completion_batch": {
                "logical_completions": completions,
                "unique_records": records,
                "usage": {
                    "checkpoint_hits": policy.TERMINAL_COUNT,
                    "logical_calls": policy.TERMINAL_COUNT,
                    "physical_calls": 0,
                    "unique_calls": policy.TERMINAL_COUNT,
                },
            },
            "preflight_artifact_sha256": preflight.sha256,
            "questions": [
                _source_question(
                    plan,
                    prediction=source_prediction_by_ordinal.get(int(plan["ordinal"])),
                )
                for plan in (*terminals, *passthroughs)
            ],
        },
    )
    source_replay = SealedArtifact(
        Path("replay.json"),
        _sha("full-replay"),
        {
            "expected_run_sha256": source_run.sha256,
            "replayed_run_sha256": source_run.sha256,
        },
    )
    return preflight, source_run, source_replay, terminals, passthroughs


def _numeric_lifecycle_fixture(
    bundle: tuple[
        SealedArtifact,
        SealedArtifact,
        SealedArtifact,
        list[dict[str, Any]],
        list[dict[str, Any]],
    ],
    *,
    identity: str = "numeric-lifecycle",
) -> tuple[SealedArtifact, SealedArtifact, dict[int, object]]:
    preflight, source_run, source_replay, _terminals, _passthroughs = bundle
    body = {
        "answer_preflight_artifact_sha256": preflight.sha256,
        "answer_replay_artifact_sha256": source_replay.sha256,
        "answer_run_artifact_sha256": source_run.sha256,
        "format": policy.numeric_frontier_cli.FORMAT,
    }
    payload = {**body, "identity_sha256": _sha(identity)}
    lifecycle_sha = _sha(f"artifact-{identity}")
    materialization = SealedArtifact(
        Path("numeric-materialization.json"), lifecycle_sha, payload
    )
    replay = SealedArtifact(Path("numeric-replay.json"), lifecycle_sha, payload)
    frontier = policy.RelevantNumericFrontier(
        policy_input_sha256=_sha("frontier-input"),
        candidate_population_receipt_sha256=_sha("frontier-population"),
        represented_handle_ids=(),
        unresolved_candidate_keys=("synthetic-open-frontier",),
        selection_truncated=False,
        closed=False,
    )
    return materialization, replay, {0: frontier}


def test_synthetic_full100_payload_has_100_judge_rows_and_zero_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight, source_run, source_replay, terminals, passthroughs = _full_source(
        source_prediction_by_ordinal={0: "terra-rewrite"}
    )
    monkeypatch.setattr(
        policy,
        "_numeric_policy_projection",
        lambda _provider_input, _frontier=None: _numeric_projection(
            status="insufficient"
        ),
    )
    monkeypatch.setattr(
        policy,
        "_replacement_policy_proof",
        lambda plan, completion: _v5_proof(
            plan,
            completion,
            accepted=False,
            prediction=str(plan["parent_prediction"]),
            error_code="candidate_not_accepted",
        ),
    )

    payload = policy.build_materialization_payload(
        preflight, source_run, source_replay, terminals, passthroughs
    )

    assert payload["question_count"] == 100
    assert len(payload["questions"]) == len(payload["judge_rows"]) == 100
    assert [row["ordinal"] for row in payload["judge_rows"]] == list(range(100))
    assert payload["physical_provider_calls_during_revalidation"] == 0
    assert payload["gold_loaded"] is False
    assert payload["provider_execution_command_available"] is False
    assert payload["caller_ordinal_routing_available"] is False
    assert payload["changed_prediction_count"] == payload["changed_from_parent_count"]
    assert payload["changed_prediction_count_basis"] == "protected_parent"
    assert payload["changed_from_parent_count"] == 0
    assert payload["changed_from_source_count"] == 1
    assert payload["questions"][0]["changed_from_parent"] is False
    assert payload["questions"][0]["changed_from_source_answer"] is True
    assert all(
        payload["questions"][int(plan["ordinal"])]["prediction"]
        == plan["parent_prediction"]
        for plan in passthroughs
    )


def test_synthetic_lifecycle_publishes_distinct_sealed_run_and_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = _full_source()
    numeric_bundle = _numeric_lifecycle_fixture(bundle)
    frontier_calls: list[object] = []
    monkeypatch.setattr(policy, "_read_source", lambda _args: bundle)
    monkeypatch.setattr(
        policy.numeric_frontier_cli,
        "load_verified_numeric_frontiers",
        lambda _root, _materialization_sha, _replay_sha: numeric_bundle,
    )

    def numeric_projection(
        _provider_input: dict[str, Any], frontier: object = None
    ) -> dict[str, Any]:
        if frontier is not None:
            frontier_calls.append(frontier)
        return _numeric_projection(status="insufficient")

    monkeypatch.setattr(
        policy,
        "_numeric_policy_projection",
        numeric_projection,
    )
    monkeypatch.setattr(
        policy,
        "_replacement_policy_proof",
        lambda plan, completion: _v5_proof(
            plan,
            completion,
            accepted=False,
            prediction=str(plan["parent_prediction"]),
            error_code="candidate_not_accepted",
        ),
    )
    args = argparse.Namespace(
        expected_numeric_frontier_materialization_sha256=numeric_bundle[0].sha256,
        expected_numeric_frontier_replay_sha256=numeric_bundle[1].sha256,
        numeric_frontier_root=tmp_path / "numeric-frontier",
        output_root=tmp_path,
    )

    materialized = policy.run_materialize(args)
    args.expected_policy_run_sha256 = materialized["run_sha256"]
    replayed = policy.run_replay(args)

    run = read_sealed_json(tmp_path / policy.RUN_NAME)
    replay = read_sealed_json(tmp_path / policy.REPLAY_NAME)
    assert run.sha256 == materialized["run_sha256"] == replayed["run_sha256"]
    assert replay.sha256 == replayed["replay_sha256"]
    assert run.path != replay.path
    assert run.payload["format"] == policy.RUN_FORMAT
    binding = run.payload["numeric_frontier_binding"]
    assert binding["materialization_artifact_sha256"] == numeric_bundle[0].sha256
    assert binding["replay_artifact_sha256"] == numeric_bundle[1].sha256
    assert binding["lifecycle_identity_sha256"] == numeric_bundle[0].payload[
        "identity_sha256"
    ]
    assert binding["frontier_ordinals"] == [0]
    assert frontier_calls == [numeric_bundle[2][0], numeric_bundle[2][0]]
    assert replay.payload["format"] == policy.REPLAY_FORMAT
    assert replay.payload["numeric_frontier_binding"] == binding
    assert replay.payload["expected_run_sha256"] == run.sha256
    assert replay.payload["replayed_run_sha256"] == run.sha256
    assert replay.payload["byte_identical"] is True
    assert replay.payload["physical_provider_calls"] == 0


def test_replay_rejects_substituted_numeric_frontier_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = _full_source()
    original = _numeric_lifecycle_fixture(bundle)
    substituted = _numeric_lifecycle_fixture(bundle, identity="substituted")
    active = {"bundle": original}
    monkeypatch.setattr(policy, "_read_source", lambda _args: bundle)
    monkeypatch.setattr(
        policy.numeric_frontier_cli,
        "load_verified_numeric_frontiers",
        lambda _root, _materialization_sha, _replay_sha: active["bundle"],
    )
    monkeypatch.setattr(
        policy,
        "_numeric_policy_projection",
        lambda _provider_input, _frontier=None: _numeric_projection(
            status="insufficient"
        ),
    )
    monkeypatch.setattr(
        policy,
        "_replacement_policy_proof",
        lambda plan, completion: _v5_proof(
            plan,
            completion,
            accepted=False,
            prediction=str(plan["parent_prediction"]),
            error_code="candidate_not_accepted",
        ),
    )
    args = argparse.Namespace(
        expected_numeric_frontier_materialization_sha256=original[0].sha256,
        expected_numeric_frontier_replay_sha256=original[1].sha256,
        numeric_frontier_root=tmp_path / "numeric-frontier",
        output_root=tmp_path,
    )
    materialized = policy.run_materialize(args)
    args.expected_policy_run_sha256 = materialized["run_sha256"]
    active["bundle"] = substituted

    with pytest.raises(
        policy.LockedSemanticGlobalTerminalFull100PolicyV5Error,
        match="run envelope changed",
    ):
        policy.run_replay(args)


def test_cli_exposes_only_provider_free_population_commands() -> None:
    parser = policy.build_parser()
    choices = parser._subparsers._group_actions[0].choices  # noqa: SLF001
    assert set(choices) == {"materialize", "replay"}
    for command in choices.values():
        options = {
            option
            for action in command._actions  # noqa: SLF001
            for option in action.option_strings
        }
        assert "--ordinal" not in options
        assert "--provider-command" not in options
        assert {
            "--numeric-frontier-root",
            "--expected-numeric-frontier-materialization-sha256",
            "--expected-numeric-frontier-replay-sha256",
        } <= options
    with pytest.raises(SystemExit):
        parser.parse_args(["provider-run"])
