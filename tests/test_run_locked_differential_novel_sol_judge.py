from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.benchmark import build_judge_prompt
from tools import plan_provider_free_differential_judge as differential
from tools import run_locked_differential_novel_sol_judge as runner
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.typed_memory_final_arm import judge_row_projection


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _receipt(body: dict[str, Any], key: str) -> dict[str, Any]:
    return {**body, key: identity_sha256(body)}


def _question(ordinal: int) -> str:
    return f"What is remembered fact {ordinal}?"


def _reference(ordinal: int) -> str:
    return f"reference-{ordinal}"


def _parent(ordinal: int) -> str:
    return f"parent-{ordinal}"


def _policy_source(
    prediction_overrides: dict[int, str],
) -> tuple[SealedArtifact, SealedArtifact]:
    questions: list[dict[str, Any]] = []
    projected: list[dict[str, Any]] = []
    for ordinal in range(differential.QUESTION_COUNT):
        parent = _parent(ordinal)
        prediction = prediction_overrides.get(ordinal, parent)
        body = {
            "changed_from_parent": prediction != parent,
            "dated_question_sha256": _sha(f"dated-{ordinal}"),
            "format": "synthetic-policy-result-v1",
            "gold_loaded": False,
            "ordinal": ordinal,
            "parent_prediction_sha256": quote_sha256(parent),
            "physical_provider_calls": 0,
            "prediction": prediction,
            "prediction_sha256": quote_sha256(prediction),
            "prediction_source": "synthetic-policy",
            "question_id": f"q-{ordinal}",
            "question_sha256": quote_sha256(_question(ordinal)),
            "route_id": "synthetic-route",
        }
        row = _receipt(body, "source_row_sha256")
        questions.append(row)
        projected.append(judge_row_projection(row))
    run = SealedArtifact(
        Path("policy-run.json"),
        _sha("policy-run-" + repr(sorted(prediction_overrides.items()))),
        {
            "format": differential.policy_cli.RUN_FORMAT,
            "gold_loaded": False,
            "judge_rows": projected,
            "physical_provider_calls_during_revalidation": 0,
            "question_count": differential.QUESTION_COUNT,
            "questions": questions,
        },
    )
    replay = SealedArtifact(
        Path("policy-replay.json"),
        _sha("policy-replay-" + run.sha256),
        {
            "byte_identical": True,
            "expected_run_sha256": run.sha256,
            "format": differential.policy_cli.REPLAY_FORMAT,
            "gold_loaded": False,
            "physical_provider_calls": 0,
            "replayed_run_sha256": run.sha256,
        },
    )
    return run, replay


def _prior() -> differential.AuthenticatedJudgeRun:
    prompts: list[dict[str, Any]] = []
    judgments: list[dict[str, Any]] = []
    for ordinal in range(differential.QUESTION_COUNT):
        question = _question(ordinal)
        reference = _reference(ordinal)
        prediction = _parent(ordinal)
        messages = build_judge_prompt(question, reference, prediction)
        prompt = _receipt(
            {
                "messages": messages,
                "messages_sha256": identity_sha256(messages),
                "ordinal": ordinal,
                "prediction": prediction,
                "prediction_sha256": quote_sha256(prediction),
                "question": question,
                "question_id": f"q-{ordinal}",
                "question_sha256": quote_sha256(question),
                "reference": reference,
                "reference_sha256": quote_sha256(reference),
            },
            "prompt_row_receipt_sha256",
        )
        prompts.append(prompt)
        output = "CORRECT: synthetic prior."
        judgments.append(
            _receipt(
                {
                    "correct": True,
                    "judge_output": output,
                    "judge_output_sha256": quote_sha256(output),
                    "messages_sha256": prompt["messages_sha256"],
                    "ordinal": ordinal,
                    "prediction_sha256": prompt["prediction_sha256"],
                    "question_id": prompt["question_id"],
                    "question_sha256": prompt["question_sha256"],
                    "reference_sha256": prompt["reference_sha256"],
                },
                "judge_row_sha256",
            )
        )
    preflight = SealedArtifact(
        Path("prior-preflight.json"),
        _sha("prior-preflight"),
        {
            "gold_loaded": True,
            "model": differential.DEFAULT_JUDGE_MODEL,
            "physical_provider_calls": 0,
            "prompt_rows": prompts,
        },
    )
    judge_payload = {
        "gold_loaded": True,
        "preflight_artifact_sha256": preflight.sha256,
        "questions": judgments,
    }
    judge = SealedArtifact(
        Path("prior-judge.json"), _sha("prior-judge"), judge_payload
    )
    replay = SealedArtifact(
        Path("prior-replay.json"), judge.sha256, judge_payload
    )
    return differential.authenticate_prior_judge_run(preflight, judge, replay)


def _plan(
    tmp_path: Path, novel_ordinals: tuple[int, ...]
) -> SealedArtifact:
    overrides = {ordinal: f"novel-{ordinal}" for ordinal in novel_ordinals}
    run, replay = _policy_source(overrides)
    payload = differential.build_differential_judge_plan(
        run, replay, (_prior(),)
    )
    artifact, _created = publish_sealed_json(tmp_path / "plan.json", payload)
    assert tuple(
        row["ordinal"] for row in artifact.payload["novel_prompt_rows"]
    ) == novel_ordinals
    return artifact


def _runtime_args(root: Path, **overrides: Any) -> SimpleNamespace:
    values = {
        "gateway_url": runner.DEFAULT_GATEWAY_URL,
        "max_concurrency": 3,
        "model": runner.DEFAULT_MODEL,
        "output_root": root,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _preflight_and_release(
    tmp_path: Path, novel_ordinals: tuple[int, ...] = (5, 36, 97)
) -> tuple[SealedArtifact, dict[str, Any], dict[str, Any], Path]:
    plan = _plan(tmp_path, novel_ordinals)
    root = tmp_path / "execution"
    preflight_args = _runtime_args(
        root,
        plan=plan.path,
        expected_plan_sha256=plan.sha256,
    )
    preflight_result = runner.run_preflight(preflight_args)
    release_args = _runtime_args(
        root,
        plan=plan.path,
        expected_plan_sha256=plan.sha256,
        expected_preflight_sha256=preflight_result["preflight_sha256"],
        approve_provider_release=True,
    )
    release_result = runner.run_approve_release(release_args)
    return plan, preflight_result, release_result, root


class _FakeCompletions:
    def __init__(self, output: str = "CORRECT: equivalent.") -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def create(self, **request: Any) -> SimpleNamespace:
        with self._lock:
            self.calls.append(dict(request))
            call_number = len(self.calls)
        return SimpleNamespace(
            choices=(
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=self.output),
                ),
            ),
            id=f"fake-sol-{call_number:03d}",
            model=runner.DEFAULT_MODEL,
            usage=None,
        )


class _FakeClient:
    max_retries = 0

    def __init__(self, output: str = "CORRECT: equivalent.") -> None:
        self.completions = _FakeCompletions(output)
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_preflight_copies_only_sealed_noncontiguous_novel_rows(
    tmp_path: Path,
) -> None:
    plan, preflight_result, _release_result, root = _preflight_and_release(
        tmp_path
    )
    preflight = read_sealed_json(root / runner.PREFLIGHT_NAME)
    prompts, rows = runner.validate_preflight_artifact(preflight, plan=plan)

    assert len(prompts) == 3
    assert tuple(row["ordinal"] for row in rows) == (5, 36, 97)
    assert list(rows) == plan.payload["novel_prompt_rows"]
    assert preflight_result["required_authorized_provider_calls"] == 3
    assert not (root / runner.CHECKPOINT_DIR_NAME).exists()


def test_empty_novel_plan_fails_before_preflight_or_checkpoint_write(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, ())
    root = tmp_path / "execution"
    args = _runtime_args(
        root, plan=plan.path, expected_plan_sha256=plan.sha256
    )

    with pytest.raises(
        runner.LockedDifferentialNovelSolJudgeError,
        match="population or runtime policy changed",
    ):
        runner.run_preflight(args)

    assert not (root / runner.PREFLIGHT_NAME).exists()
    assert not (root / runner.CHECKPOINT_DIR_NAME).exists()


def test_release_binds_output_root_and_cli_has_no_ordinal_route(
    tmp_path: Path,
) -> None:
    _plan_artifact, preflight_result, release_result, root = (
        _preflight_and_release(tmp_path)
    )
    preflight = read_sealed_json(root / runner.PREFLIGHT_NAME)
    release = read_sealed_json(root / runner.RELEASE_NAME)
    assert release.sha256 == release_result["release_sha256"]
    with pytest.raises(
        runner.LockedDifferentialNovelSolJudgeError, match="release changed"
    ):
        runner._validate_release(  # noqa: SLF001
            release, preflight=preflight, output_root=tmp_path / "other"
        )

    parser = runner.build_parser()
    commands = next(
        action
        for action in parser._actions  # noqa: SLF001
        if getattr(action, "choices", None)
    )
    assert set(commands.choices) == {
        "approve-release",
        "materialize",
        "preflight",
        "provider-run",
        "replay",
    }
    for command in commands.choices.values():
        assert not any(
            "ordinal" in option or "question" in option or "prompt" in option
            for action in command._actions  # noqa: SLF001
            for option in action.option_strings
        )
    assert preflight_result["selected_ordinals"] == [5, 36, 97]


def test_authorization_is_exact_remaining_and_checked_before_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan_artifact, preflight_result, release_result, root = (
        _preflight_and_release(tmp_path)
    )
    preflight, prompts, _rows = runner._read_preflight(  # noqa: SLF001
        root, preflight_result["preflight_sha256"]
    )
    release = runner._read_release(  # noqa: SLF001
        root, release_result["release_sha256"], preflight=preflight
    )
    client = _FakeClient()
    runtime = runner._runtime(  # noqa: SLF001
        preflight, release, prompts, output_root=root, client=client
    )
    try:
        runtime._provider_call(runtime._unique_order[0])  # noqa: SLF001
    finally:
        runtime.close()
    monkeypatch.setattr(
        runner,
        "load_dotenv",
        lambda: pytest.fail("environment opened before authorization matched"),
    )
    wrong = _runtime_args(
        root,
        expected_preflight_sha256=preflight.sha256,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=3,
        api_key_env="SYNTHETIC_SOL_KEY",
    )
    with pytest.raises(
        runner.LockedDifferentialNovelSolJudgeError,
        match="exactly equal remaining calls",
    ):
        runner.run_provider(wrong)


def test_incomplete_request_refuses_retry_before_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan_artifact, preflight_result, release_result, root = (
        _preflight_and_release(tmp_path)
    )
    preflight, prompts, _rows = runner._read_preflight(  # noqa: SLF001
        root, preflight_result["preflight_sha256"]
    )
    release = runner._read_release(  # noqa: SLF001
        root, release_result["release_sha256"], preflight=preflight
    )
    runtime = runner._runtime(  # noqa: SLF001
        preflight, release, prompts, output_root=root, client=_FakeClient()
    )
    try:
        runtime._reserve(runtime._unique_order[0])  # noqa: SLF001
    finally:
        runtime.close()
    monkeypatch.setattr(
        runner,
        "load_dotenv",
        lambda: pytest.fail("environment opened after incomplete request"),
    )
    args = _runtime_args(
        root,
        expected_preflight_sha256=preflight.sha256,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=3,
        api_key_env="SYNTHETIC_SOL_KEY",
    )
    with pytest.raises(
        runner.LockedDifferentialNovelSolJudgeError,
        match="unsafe retry forbidden",
    ):
        runner.run_provider(args)


def test_synthetic_lifecycle_authenticates_and_completes_100_row_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, preflight_result, release_result, root = _preflight_and_release(tmp_path)
    client = _FakeClient()
    monkeypatch.setattr(runner, "load_dotenv", lambda: None)
    monkeypatch.setenv("SYNTHETIC_SOL_KEY", "local-test-key")
    monkeypatch.setattr(
        runner.judging,
        "_make_provider_client",
        lambda *_args, **_kwargs: client,
    )
    provider_args = _runtime_args(
        root,
        expected_preflight_sha256=preflight_result["preflight_sha256"],
        expected_release_sha256=release_result["release_sha256"],
        enable_provider=True,
        authorized_provider_calls=3,
        api_key_env="SYNTHETIC_SOL_KEY",
    )
    provider = runner.run_provider(provider_args)
    assert provider["physical_provider_calls"] == 3
    assert len(client.completions.calls) == 3
    assert client.closed is True

    materialize_args = _runtime_args(
        root,
        expected_preflight_sha256=preflight_result["preflight_sha256"],
        expected_release_sha256=release_result["release_sha256"],
    )
    materialized = runner.run_materialize(materialize_args)
    replay_args = _runtime_args(
        root,
        expected_preflight_sha256=preflight_result["preflight_sha256"],
        expected_release_sha256=release_result["release_sha256"],
        expected_judge_sha256=materialized["judge_sha256"],
    )
    replayed = runner.run_replay(replay_args)
    assert replayed["byte_identical"] is True
    assert replayed["judge_replay_sha256"] == materialized["judge_sha256"]

    authenticated = runner.load_verified_novel_judge_run(
        root,
        plan_path=plan.path,
        expected_plan_sha256=plan.sha256,
        expected_preflight_sha256=preflight_result["preflight_sha256"],
        expected_release_sha256=release_result["release_sha256"],
        expected_judge_sha256=materialized["judge_sha256"],
        expected_replay_sha256=replayed["judge_replay_sha256"],
    )
    assert authenticated.model == runner.DEFAULT_MODEL
    assert tuple(row["ordinal"] for row in authenticated.entries) == (5, 36, 97)
    merged = differential.merge_differential_judgments(plan, (authenticated,))
    assert merged["question_count"] == 100
    assert len(merged["questions"]) == 100
    assert merged["correct"] == 100


def test_invalid_binary_completion_cannot_materialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan_artifact, preflight_result, release_result, root = (
        _preflight_and_release(tmp_path, (97,))
    )
    client = _FakeClient("AMBIGUOUS")
    monkeypatch.setattr(runner, "load_dotenv", lambda: None)
    monkeypatch.setenv("SYNTHETIC_SOL_KEY", "local-test-key")
    monkeypatch.setattr(
        runner.judging,
        "_make_provider_client",
        lambda *_args, **_kwargs: client,
    )
    runner.run_provider(
        _runtime_args(
            root,
            expected_preflight_sha256=preflight_result["preflight_sha256"],
            expected_release_sha256=release_result["release_sha256"],
            enable_provider=True,
            authorized_provider_calls=1,
            api_key_env="SYNTHETIC_SOL_KEY",
        )
    )
    with pytest.raises(
        runner.LockedDifferentialNovelSolJudgeError,
        match="invalid binary verdict",
    ):
        runner.run_materialize(
            _runtime_args(
                root,
                expected_preflight_sha256=preflight_result["preflight_sha256"],
                expected_release_sha256=release_result["release_sha256"],
            )
        )
