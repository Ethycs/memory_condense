from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_fast_completion_runtime import _FakeClient
from tests.test_matched_eval_query_payload_live import _plan
from tools import run_locked_query_answer_judge as judge_cli
from tools import run_locked_query_payload_answers as payload_cli
from tools.matched_eval.payload_arm_identity import (
    BINDING_NAME,
    PARTITION_PAYLOAD_PROFILE,
    QUERY_GUIDED_PAYLOAD_PROFILE,
    QUERY_PAYLOAD_PROFILE,
    PayloadArmIdentityError,
    ensure_payload_semantic_arm_binding,
    load_verified_payload_semantic_arm_binding,
    profile_for_cli_arm,
    profile_for_delta_tier,
)
from tools.matched_eval.query_payload_live import (
    CHECKPOINT_DIR_NAME,
    load_query_payload_answer_provider_journals,
    materialize_query_payload_answers,
    preflight_query_payload_answers,
    replay_query_payload_answers,
    run_query_payload_answer_provider,
)


def _terminal_payload(tmp_path: Path):
    plan, _parent = _plan(tmp_path)
    output = tmp_path / "payload-answer"
    preflight = preflight_query_payload_answers(plan, output_root=output)
    provider = run_query_payload_answer_provider(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        enable_provider=True,
        authorized_provider_calls=1,
        client=_FakeClient(output / CHECKPOINT_DIR_NAME),
        max_concurrency=1,
    )
    assert provider.physical_provider_calls == 1
    journals = load_query_payload_answer_provider_journals(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        max_concurrency=1,
    )
    materialized = materialize_query_payload_answers(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        completion_batch=journals.batch,
    )
    replayed = replay_query_payload_answers(
        plan,
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        expected_run_sha256=materialized.answer_artifact.sha256,
        max_concurrency=1,
    )
    return plan, output, preflight, replayed


def _byte_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.startswith(BINDING_NAME)
    }


def test_semantic_profiles_are_distinct_and_route_by_sealed_tier() -> None:
    profiles = (
        QUERY_PAYLOAD_PROFILE,
        PARTITION_PAYLOAD_PROFILE,
        QUERY_GUIDED_PAYLOAD_PROFILE,
    )

    assert len({profile.profile_sha256 for profile in profiles}) == 3
    assert len({profile.semantic_arm_label for profile in profiles}) == 3
    for profile in profiles:
        assert profile_for_delta_tier(profile.delta_tier) is profile
    assert profile_for_cli_arm("query-payload") is QUERY_PAYLOAD_PROFILE
    assert profile_for_cli_arm("partition-payload") is PARTITION_PAYLOAD_PROFILE
    assert profile_for_cli_arm("query-guided-payload") is (
        QUERY_GUIDED_PAYLOAD_PROFILE
    )


def test_sidecar_binds_terminal_plane_without_mutating_journals_or_call_keys(
    tmp_path: Path,
) -> None:
    plan, output, preflight, replayed = _terminal_payload(tmp_path)
    before = _byte_manifest(output)
    request_paths = sorted((output / CHECKPOINT_DIR_NAME).glob("*.request.json"))
    call_keys_before = tuple(path.stem.removesuffix(".request") for path in request_paths)

    lifecycle = payload_cli._replay_loaded_plan(
        plan,
        SimpleNamespace(
            expected_answer_preflight_sha256=preflight.sha256,
            expected_run_sha256=replayed.run_sha256,
            gateway_url=payload_cli.live.DEFAULT_GATEWAY_URL,
            max_concurrency=1,
            output_root=output,
        ),
    )
    assert lifecycle["semantic_arm_binding_created"] is True
    sidecar = load_verified_payload_semantic_arm_binding(
        output,
        expected_profile=QUERY_PAYLOAD_PROFILE,
        expected_binding_sha256=lifecycle["semantic_arm_binding_sha256"],
        expected_question_count=len(plan.rows),
    )
    verified = load_verified_payload_semantic_arm_binding(
        output,
        expected_profile=QUERY_PAYLOAD_PROFILE,
        expected_binding_sha256=sidecar.sha256,
        expected_question_count=len(plan.rows),
    )

    assert verified.sha256 == sidecar.sha256
    assert verified.payload["answer_run_sha256"] == replayed.run_sha256
    assert verified.payload["answer_replay_sha256"] == replayed.replay_sha256
    assert verified.payload["sealed_prediction_bytes_mutated"] is False
    assert verified.payload["provider_prompt_content_mutated"] is False
    assert verified.payload["new_provider_calls"] == 0
    assert _byte_manifest(output) == before
    request_paths_after = sorted(
        (output / CHECKPOINT_DIR_NAME).glob("*.request.json")
    )
    call_keys_after = tuple(
        path.stem.removesuffix(".request") for path in request_paths_after
    )
    assert call_keys_after == call_keys_before

    reused, reused_created = ensure_payload_semantic_arm_binding(
        output,
        profile=QUERY_PAYLOAD_PROFILE,
        expected_question_count=len(plan.rows),
    )
    assert reused_created is False
    assert reused.sha256 == sidecar.sha256


def test_profile_swap_is_rejected_against_sealed_alias_tier(
    tmp_path: Path,
) -> None:
    plan, output, _preflight, _replayed = _terminal_payload(tmp_path)
    ensure_payload_semantic_arm_binding(
        output,
        profile=QUERY_PAYLOAD_PROFILE,
        expected_question_count=len(plan.rows),
    )

    with pytest.raises(PayloadArmIdentityError, match="requested arm"):
        load_verified_payload_semantic_arm_binding(
            output,
            expected_profile=PARTITION_PAYLOAD_PROFILE,
            expected_question_count=len(plan.rows),
        )
    with pytest.raises(PayloadArmIdentityError, match="sealed alias tier"):
        ensure_payload_semantic_arm_binding(
            output,
            profile=PARTITION_PAYLOAD_PROFILE,
            expected_question_count=len(plan.rows),
        )


def test_judge_cli_rejects_semantic_binding_before_gold_capable_core(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = judge_cli._parser().parse_args(
        [
            "preflight",
            "--arm",
            "query-payload",
            "--answer-root",
            str(tmp_path / "answer"),
            "--expected-answer-preflight-sha256",
            "a" * 64,
            "--expected-answer-run-sha256",
            "b" * 64,
        ]
    )
    sentinel_plan = object()
    sentinel_plane = SimpleNamespace(rows=(object(),))
    order: list[str] = []

    monkeypatch.setattr(judge_cli.payload_cli, "_load_plan", lambda _args: sentinel_plan)

    def replay(plan, **_kwargs):
        assert plan is sentinel_plan
        order.append("answer_replay")
        return sentinel_plane

    def reject_binding(*_args, **_kwargs):
        order.append("semantic_binding")
        raise PayloadArmIdentityError("semantic binding rejected")

    def gold_capable_core(**_kwargs):
        pytest.fail("judge core was entered before semantic binding verification")

    monkeypatch.setattr(judge_cli, "replay_query_payload_answers", replay)
    monkeypatch.setattr(
        judge_cli,
        "load_verified_payload_semantic_arm_binding",
        reject_binding,
    )
    monkeypatch.setattr(
        judge_cli,
        "preflight_query_answer_changed_only_judge",
        gold_capable_core,
    )

    with pytest.raises(PayloadArmIdentityError, match="semantic binding rejected"):
        judge_cli._preflight(args)
    assert order == ["answer_replay", "semantic_binding"]
