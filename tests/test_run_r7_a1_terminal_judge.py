from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.benchmark import build_judge_prompt
from memory_condense.eval.fast_completion_runtime import (
    preflight_fast_completion_prompts,
)
from tools import run_r7_a1_terminal_judge as judge
from tools.matched_eval.artifacts import SealedArtifact, read_sealed_json
from tools.matched_eval.contracts import identity_sha256


ARMS = (
    "raw_retained_no_operator",
    "raw_retained_full_operator",
    "typed_facts_plus_unresolved_raw_full_operator",
)


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _source_fixture() -> tuple[
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
    dict[str, tuple[judge._GoldRow, ...]],  # noqa: SLF001
]:
    rows: list[dict[str, Any]] = []
    gold: dict[str, list[judge._GoldRow]] = {arm: [] for arm in ARMS}  # noqa: SLF001
    for index in range(judge.QUESTION_COUNT):
        question_id = f"q-{index:02d}"
        question = f"What is fact {index}?"
        dated = f"2026-08-{index + 1:02d}: {question}"
        for arm_index, arm in enumerate(ARMS):
            prediction = f"prediction-{arm_index}-{index}"
            rows.append(
                {
                    "arm": arm,
                    "dated_question_sha256": quote_sha256(dated),
                    "format": judge.answer_cli.JUDGE_ROW_FORMAT,
                    "prediction": prediction,
                    "prediction_sha256": quote_sha256(prediction),
                    "question_id": question_id,
                    "question_sha256": quote_sha256(dated),
                    "source_row_sha256": _sha(f"source {arm} {index}"),
                }
            )
            gold[arm].append(
                judge._GoldRow(  # noqa: SLF001
                    question_id=question_id,
                    question_sha256=quote_sha256(dated),
                    dated_question=dated,
                    dated_question_sha256=quote_sha256(dated),
                    reference=f"reference-{index}",
                    reference_sha256=quote_sha256(f"reference-{index}"),
                    category=f"category-{index % 3}",
                )
            )
    run_payload = {
        "arm_prediction_population_sha256s": {
            arm: identity_sha256(
                [row["prediction_sha256"] for row in rows if row["arm"] == arm]
            )
            for arm in ARMS
        },
        "compiler_outputs_artifact_sha256": _sha("compiler outputs"),
        "compiler_outputs_replay_artifact_sha256": _sha("compiler replay"),
        "judge_row_population_sha256": identity_sha256(rows),
        "judge_rows": rows,
        "preflight_construction_artifact_sha256": _sha("answer preflight"),
        "preflight_replay_artifact_sha256": _sha("answer preflight replay"),
        "release_authorization_artifact_sha256": _sha("answer release"),
        "source_a1_construction_artifact_sha256": _sha("source A1"),
        "source_a1_replay_artifact_sha256": _sha("source A1 replay"),
    }
    run_sha = _sha("answer run")
    run = SealedArtifact(Path("answer-run.json"), run_sha, run_payload)
    replay = SealedArtifact(Path("answer-replay.json"), run_sha, run_payload)
    return run, replay, tuple(rows), {key: tuple(value) for key, value in gold.items()}


def test_answer_schema_exposes_expected_three_factorial_arms() -> None:
    assert tuple(judge.answer_cli.ARM_LABELS) == ARMS


def _build_preflight(arm: str = ARMS[1]):
    run, replay, rows, gold = _source_fixture()
    selected = tuple(row for row in rows if row["arm"] == arm)
    payload, prompts = judge.build_preflight_payload(
        run,
        replay,
        selected,
        gold[arm],
        answer_root=Path("answer-root"),
        arm=arm,
        gold_population_sha256=_sha("gold population"),
        model=judge.DEFAULT_MODEL,
        gateway_url=judge.DEFAULT_GATEWAY_URL,
        max_concurrency=4,
    )
    artifact = SealedArtifact(Path("preflight.json"), _sha("preflight"), payload)
    return run, replay, rows, gold, artifact, prompts


def test_preflight_is_exact11_question_id_derived_and_qrp_only() -> None:
    _run, _replay, _rows, _gold, artifact, prompts = _build_preflight()
    rebuilt, prompt_rows = judge._validate_preflight(  # noqa: SLF001
        artifact, expected_arm=ARMS[1]
    )

    assert rebuilt == prompts
    assert artifact.payload["required_authorized_provider_calls"] == 11
    assert artifact.payload["caller_ordinal_routing_available"] is False
    assert artifact.payload["answer_arm"] == ARMS[1]
    assert len(prompt_rows) == len({row["question_id"] for row in prompt_rows}) == 11
    for messages, row in zip(prompts, prompt_rows, strict=True):
        assert list(messages) == build_judge_prompt(
            row["dated_question"], row["reference"], row["prediction"]
        )
        assert "ordinal" not in row
        assert not any(
            key in row
            for key in (
                "context",
                "evidence",
                "facts",
                "handles",
                "messages_source",
                "used_handle_ids",
            )
        )


def test_each_arm_has_an_independent_preflight_identity() -> None:
    identities: set[str] = set()
    source_populations: set[str] = set()
    for arm in ARMS:
        _run, _replay, _rows, _gold, artifact, _prompts = _build_preflight(arm)
        identities.add(identity_sha256(artifact.payload))
        source_populations.add(artifact.payload["source_population_sha256"])
    assert len(identities) == len(source_populations) == 3


def test_resealed_source_prediction_row_must_still_match_answer_run() -> None:
    run, replay, rows, gold = _source_fixture()
    selected = [dict(row) for row in rows if row["arm"] == ARMS[1]]
    selected[0]["source_row_sha256"] = _sha("foreign source row")

    with pytest.raises(judge.R7A1TerminalJudgeError, match="source population"):
        judge.build_preflight_payload(
            run,
            replay,
            selected,
            gold[ARMS[1]],
            answer_root=Path("answer-root"),
            arm=ARMS[1],
            gold_population_sha256=_sha("gold population"),
            model=judge.DEFAULT_MODEL,
            gateway_url=judge.DEFAULT_GATEWAY_URL,
            max_concurrency=4,
        )


def test_fully_resealed_extra_message_is_rejected() -> None:
    _run, _replay, _rows, _gold, artifact, _prompts = _build_preflight()
    payload = dict(artifact.payload)
    rows = [dict(row) for row in payload["prompt_rows"]]
    messages_by_row = [list(row["messages"]) for row in rows]
    messages_by_row[0].append(
        {"role": "assistant", "content": "EVIDENCE_SENTINEL"}
    )
    population = preflight_fast_completion_prompts(
        messages_by_row, max_prompt_tokens=judge.DEFAULT_MAX_PROMPT_TOKENS
    )
    inputs: list[str] = []
    for row, messages, receipt in zip(
        rows, messages_by_row, population.ordered_rows, strict=True
    ):
        body = dict(row)
        body.pop("prompt_row_receipt_sha256")
        body["messages"] = messages
        body["messages_sha256"] = receipt.messages_sha256
        body["prompt_token_proxy"] = receipt.prompt_token_proxy
        body["judge_input_receipt_sha256"] = identity_sha256(
            judge._judge_input_body(body, messages)  # noqa: SLF001
        )
        inputs.append(body["judge_input_receipt_sha256"])
        row.clear()
        row.update(
            {**body, "prompt_row_receipt_sha256": identity_sha256(body)}
        )
    payload["prompt_rows"] = rows
    payload["prompt_population"] = population.model_dump()
    payload["prompt_population_sha256"] = population.prompt_population_sha256
    payload["judge_input_population_sha256"] = identity_sha256(inputs)
    changed = SealedArtifact(Path("changed.json"), _sha("changed"), payload)

    with pytest.raises(judge.R7A1TerminalJudgeError, match="non-contract input"):
        judge._validate_preflight(changed, expected_arm=ARMS[1])  # noqa: SLF001


def _preflight_args(tmp_path: Path, arm: str = ARMS[1]) -> SimpleNamespace:
    run, replay, _rows, _gold = _source_fixture()
    return SimpleNamespace(
        answer_arm=arm,
        answer_root=tmp_path / "answer",
        dataset=tmp_path / "dataset.json",
        expected_answer_preflight_construction_sha256=run.payload[
            "preflight_construction_artifact_sha256"
        ],
        expected_answer_preflight_replay_sha256=run.payload[
            "preflight_replay_artifact_sha256"
        ],
        expected_answer_release_sha256=run.payload[
            "release_authorization_artifact_sha256"
        ],
        expected_answer_replay_sha256=replay.sha256,
        expected_answer_run_sha256=run.sha256,
        gateway_url=judge.DEFAULT_GATEWAY_URL,
        judge_output_root=tmp_path / f"judge-{arm}",
        max_concurrency=4,
        model=judge.DEFAULT_MODEL,
        split=tmp_path / "split.json",
    )


def _install_source(
    monkeypatch: pytest.MonkeyPatch,
    *,
    arm: str = ARMS[1],
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    run, replay, rows, gold = _source_fixture()
    monkeypatch.setattr(
        judge.answer_cli,
        "load_verified_answer_run",
        lambda *_args, **_kwargs: (run, replay, rows),
    )

    def gold_reader(source_rows: Any, **_kwargs: Any):
        assert tuple(row["question_id"] for row in source_rows) == tuple(
            row.question_id for row in gold[arm]
        )
        return gold[arm], _sha("gold population")

    monkeypatch.setattr(judge, "_load_locked_gold", gold_reader)
    return run, replay, rows


class _FakeCompletions:
    def __init__(self, output: str = "CORRECT: semantically equivalent.") -> None:
        self.calls: list[dict[str, Any]] = []
        self.output = output
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
            id=f"fake-sol-{call_number:02d}",
            model=judge.DEFAULT_MODEL,
            usage=None,
        )


class _FakeClient:
    max_retries = 0

    def __init__(self, output: str = "CORRECT: semantically equivalent.") -> None:
        self.completions = _FakeCompletions(output)
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _seal_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str = ARMS[1]
) -> tuple[SimpleNamespace, dict[str, Any]]:
    _install_source(monkeypatch, arm=arm)
    args = _preflight_args(tmp_path, arm)
    return args, judge.run_preflight(args)


def _seal_release(args: SimpleNamespace, preflight_sha: str) -> dict[str, Any]:
    return judge.run_approve_release(
        SimpleNamespace(
            answer_arm=args.answer_arm,
            answer_root=args.answer_root,
            approve_provider_release=True,
            expected_judge_preflight_sha256=preflight_sha,
            gateway_url=judge.DEFAULT_GATEWAY_URL,
            judge_output_root=args.judge_output_root,
            max_concurrency=4,
            model=judge.DEFAULT_MODEL,
        )
    )


def test_complete_seven_phase_lifecycle_is_exact11_and_offline_after_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, preflight = _seal_preflight(tmp_path, monkeypatch)
    release = _seal_release(args, preflight["preflight_sha256"])
    client = _FakeClient()
    monkeypatch.setattr(judge, "load_dotenv", lambda: None)
    monkeypatch.setenv("TEST_SOL_KEY", "local-test-key")
    monkeypatch.setattr(judge, "_make_provider_client", lambda *_args: client)
    provider_args = SimpleNamespace(
        answer_arm=args.answer_arm,
        answer_root=args.answer_root,
        api_key_env="TEST_SOL_KEY",
        authorized_provider_calls=11,
        enable_provider=True,
        expected_judge_preflight_sha256=preflight["preflight_sha256"],
        expected_release_sha256=release["release_sha256"],
        gateway_url=judge.DEFAULT_GATEWAY_URL,
        judge_output_root=args.judge_output_root,
        max_concurrency=4,
        model=judge.DEFAULT_MODEL,
    )
    provider = judge.run_provider(provider_args)
    assert provider["physical_provider_calls"] == 11
    assert len(client.completions.calls) == 11
    assert client.closed is True
    assert all(
        set(call) == {"max_tokens", "messages", "model"}
        for call in client.completions.calls
    )
    assert all(
        "EVIDENCE_SENTINEL" not in str(call) for call in client.completions.calls
    )

    provider_args.authorized_provider_calls = 0
    monkeypatch.setattr(
        judge,
        "load_dotenv",
        lambda: pytest.fail("environment opened for checkpoint replay"),
    )
    assert judge.run_provider(provider_args)["physical_provider_calls"] == 0

    offline = SimpleNamespace(
        answer_arm=args.answer_arm,
        answer_root=args.answer_root,
        expected_judge_preflight_sha256=preflight["preflight_sha256"],
        expected_release_sha256=release["release_sha256"],
        gateway_url=judge.DEFAULT_GATEWAY_URL,
        judge_output_root=args.judge_output_root,
        max_concurrency=4,
        model=judge.DEFAULT_MODEL,
    )
    materialized = judge.run_materialize(offline)
    replay = judge.run_replay(
        SimpleNamespace(
            **vars(offline), expected_judge_sha256=materialized["judge_sha256"]
        )
    )
    monkeypatch.setattr(
        judge,
        "_checkpoint_batch",
        lambda *_args, **_kwargs: pytest.fail("score opened provider journals"),
    )
    score_args = SimpleNamespace(
        answer_arm=args.answer_arm,
        answer_root=args.answer_root,
        expected_judge_preflight_sha256=preflight["preflight_sha256"],
        expected_judge_replay_sha256=replay["judge_replay_sha256"],
        expected_judge_sha256=materialized["judge_sha256"],
        expected_release_sha256=release["release_sha256"],
        judge_output_root=args.judge_output_root,
    )
    score = judge.run_score(score_args)
    score_replay = judge.run_score_replay(
        SimpleNamespace(**vars(score_args), expected_score_sha256=score["score_sha256"])
    )
    _judge, sealed_score, rows = judge.load_verified_judge_score(
        args.judge_output_root,
        arm=args.answer_arm,
        expected_preflight_sha256=preflight["preflight_sha256"],
        expected_release_sha256=release["release_sha256"],
        expected_judge_sha256=materialized["judge_sha256"],
        expected_judge_replay_sha256=replay["judge_replay_sha256"],
        expected_score_sha256=score["score_sha256"],
        expected_score_replay_sha256=score_replay["score_replay_sha256"],
    )
    assert score["correct"] == 11
    assert score["accuracy"] == 1.0
    assert sealed_score.payload["answer_arm"] == ARMS[1]
    assert len(rows) == 11


def test_invalid_verdict_fails_closed_at_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, preflight = _seal_preflight(tmp_path, monkeypatch)
    release = _seal_release(args, preflight["preflight_sha256"])
    client = _FakeClient("MAYBE")
    monkeypatch.setattr(judge, "load_dotenv", lambda: None)
    monkeypatch.setenv("TEST_SOL_KEY", "local-test-key")
    monkeypatch.setattr(judge, "_make_provider_client", lambda *_args: client)
    common = dict(
        answer_arm=args.answer_arm,
        answer_root=args.answer_root,
        expected_judge_preflight_sha256=preflight["preflight_sha256"],
        expected_release_sha256=release["release_sha256"],
        gateway_url=judge.DEFAULT_GATEWAY_URL,
        judge_output_root=args.judge_output_root,
        max_concurrency=4,
        model=judge.DEFAULT_MODEL,
    )
    judge.run_provider(
        SimpleNamespace(
            **common,
            api_key_env="TEST_SOL_KEY",
            authorized_provider_calls=11,
            enable_provider=True,
        )
    )
    with pytest.raises(judge.R7A1TerminalJudgeError, match="invalid Sol verdict"):
        judge.run_materialize(SimpleNamespace(**common))


def test_unsafe_retry_and_cli_has_no_ordinal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, preflight_result = _seal_preflight(tmp_path, monkeypatch)
    release_result = _seal_release(args, preflight_result["preflight_sha256"])
    preflight, prompts, _rows = judge._read_preflight(  # noqa: SLF001
        args.judge_output_root,
        preflight_result["preflight_sha256"],
        expected_arm=args.answer_arm,
    )
    release = judge._read_release(  # noqa: SLF001
        args.judge_output_root,
        release_result["release_sha256"],
        preflight=preflight,
    )
    runtime = judge._runtime(  # noqa: SLF001
        preflight,
        release,
        prompts,
        output_root=args.judge_output_root,
        model=judge.DEFAULT_MODEL,
        gateway_url=judge.DEFAULT_GATEWAY_URL,
        max_concurrency=4,
        client=_FakeClient(),
    )
    try:
        runtime._reserve(runtime._unique_order[0])  # noqa: SLF001
    finally:
        runtime.close()
    with pytest.raises(judge.R7A1TerminalJudgeError, match="unsafe retry forbidden"):
        judge._read_only_checkpoint_count(args.judge_output_root)  # noqa: SLF001

    parser = judge.build_parser()
    subparsers = next(
        action for action in parser._actions if getattr(action, "choices", None)  # noqa: SLF001
    )
    assert set(subparsers.choices) == {
        "approve-release",
        "materialize",
        "preflight",
        "provider-run",
        "replay",
        "score",
        "score-replay",
    }
    for command in subparsers.choices.values():
        assert not any(
            "ordinal" in option
            for action in command._actions  # noqa: SLF001
            for option in action.option_strings
        )
