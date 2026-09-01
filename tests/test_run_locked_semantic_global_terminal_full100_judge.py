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
from tools import run_locked_semantic_global_terminal_full100_judge as judge
from tools.matched_eval.artifacts import (
    SealedArtifact,
    read_sealed_json,
)
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.typed_memory_final_arm import JUDGE_ROW_FORMAT
from tools.matched_eval.typed_memory_final_judging import TypedFinalJudgeGoldRow


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _source_fixture() -> tuple[
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
    tuple[TypedFinalJudgeGoldRow, ...],
    str,
]:
    rows: list[dict[str, Any]] = []
    gold: list[TypedFinalJudgeGoldRow] = []
    for ordinal in judge.ALL_ORDINALS:
        question_id = f"question-{ordinal:03d}"
        question = f"Which sealed fact belongs to row {ordinal}?"
        dated = f"2026-08-30: {question}"
        reference = f"reference-{ordinal:03d}"
        prediction = f"prediction-{ordinal:03d}"
        rows.append(
            {
                "changed_from_parent": ordinal % 3 == 0,
                "dated_question_sha256": quote_sha256(dated),
                "format": JUDGE_ROW_FORMAT,
                "ordinal": ordinal,
                "parent_prediction_sha256": _sha(f"parent {ordinal}"),
                "prediction": prediction,
                "prediction_sha256": quote_sha256(prediction),
                "prediction_source": "sealed-full100-test-source",
                "question_id": question_id,
                "question_sha256": quote_sha256(question),
                "route_id": "full100-test-route",
                "source_row_sha256": _sha(f"source row {ordinal}"),
            }
        )
        gold.append(
            TypedFinalJudgeGoldRow(
                ordinal=ordinal,
                question_id=question_id,
                question=question,
                question_sha256=quote_sha256(question),
                dated_question=dated,
                dated_question_sha256=quote_sha256(dated),
                reference=reference,
                reference_sha256=quote_sha256(reference),
                category=f"category-{ordinal % 4}",
            )
        )
    run_sha = _sha("full100 answer run")
    source_bindings = {
        key: (
            _sha(f"answer binding {key}")
            if key.endswith("_sha256")
            else 10 + index
        )
        for index, key in enumerate(judge.answer_cli.SOURCE_BINDING_KEYS)
    }
    run_payload = {
        "preflight_artifact_sha256": _sha("full100 answer preflight"),
        "release_authorization_artifact_sha256": _sha(
            "full100 answer release"
        ),
        **source_bindings,
    }
    replay_payload = {
        "expected_run_sha256": run_sha,
        "preflight_artifact_sha256": run_payload[
            "preflight_artifact_sha256"
        ],
        "release_authorization_artifact_sha256": run_payload[
            "release_authorization_artifact_sha256"
        ],
        "replayed_run_sha256": run_sha,
        **source_bindings,
    }
    run = SealedArtifact(Path("answer-run.json"), run_sha, run_payload)
    replay = SealedArtifact(
        Path("answer-replay.json"), _sha("full100 answer replay"), replay_payload
    )
    return run, replay, tuple(rows), tuple(gold), _sha("postseal audit")


def _preflight_args(
    tmp_path: Path, run: SealedArtifact, replay: SealedArtifact, postseal: str
) -> SimpleNamespace:
    return SimpleNamespace(
        answer_root=tmp_path / "answer",
        dataset=tmp_path / "dataset.json",
        expected_answer_preflight_sha256=run.payload[
            "preflight_artifact_sha256"
        ],
        expected_answer_replay_sha256=replay.sha256,
        expected_answer_run_sha256=run.sha256,
        expected_postseal_audit_sha256=postseal,
        gateway_url=judge.DEFAULT_GATEWAY_URL,
        judge_output_root=tmp_path / "judge",
        max_concurrency=4,
        model=judge.DEFAULT_MODEL,
        postseal_audit=tmp_path / "postseal.json",
        split=tmp_path / "split.json",
    )


def _install_sources(
    monkeypatch: pytest.MonkeyPatch,
    run: SealedArtifact,
    replay: SealedArtifact,
    rows: tuple[dict[str, Any], ...],
    gold: tuple[TypedFinalJudgeGoldRow, ...],
) -> list[str]:
    calls: list[str] = []

    def answer_reader(*_args: Any, **_kwargs: Any):
        calls.append("answer")
        return run, replay, rows

    def gold_reader(**kwargs: Any):
        calls.append("gold")
        assert kwargs["allow_subset"] is False
        assert kwargs["source_rows"] == rows
        return gold, _sha("gold population")

    monkeypatch.setattr(
        judge.answer_cli, "load_verified_answer_run", answer_reader
    )
    monkeypatch.setattr(judge, "load_locked_typed_final_gold", gold_reader)
    return calls


def _make_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[SimpleNamespace, dict[str, Any], tuple[dict[str, Any], ...]]:
    run, replay, rows, gold, postseal = _source_fixture()
    _install_sources(monkeypatch, run, replay, rows, gold)
    args = _preflight_args(tmp_path, run, replay, postseal)
    return args, judge.run_preflight(args), rows


def _release(
    args: SimpleNamespace, preflight_sha: str
) -> tuple[SimpleNamespace, dict[str, Any]]:
    release_args = SimpleNamespace(
        **vars(args),
        approve_provider_release=True,
        expected_judge_preflight_sha256=preflight_sha,
    )
    return release_args, judge.run_approve_release(release_args)


def test_preflight_authenticates_full100_before_gold_and_seals_only_qrp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, replay, rows, gold, postseal = _source_fixture()
    calls = _install_sources(monkeypatch, run, replay, rows, gold)
    args = _preflight_args(tmp_path, run, replay, postseal)

    result = judge.run_preflight(args)
    artifact = read_sealed_json(
        Path(args.judge_output_root) / judge.PREFLIGHT_NAME
    )
    prompts, prompt_rows = judge._validate_preflight(artifact)  # noqa: SLF001

    assert calls == ["answer", "gold"]
    assert result["physical_provider_calls"] == 0
    assert result["required_authorized_provider_calls"] == 100
    assert artifact.payload["question_count"] == 100
    assert artifact.payload["selected_question_count"] == 100
    assert artifact.payload["ordinal_cli_routing_available"] is False
    assert artifact.payload["production_ordinal_routing_enabled"] is False
    assert tuple(row["ordinal"] for row in prompt_rows) == tuple(range(100))
    assert len({row["messages_sha256"] for row in prompt_rows}) == 100
    for messages, row in zip(prompts, prompt_rows, strict=True):
        assert list(messages) == build_judge_prompt(
            row["question"], row["reference"], row["prediction"]
        )
        assert not any(
            key in row
            for key in ("evidence", "handles", "used_handle_ids", "context")
        )
    assert not (
        Path(args.judge_output_root) / judge.CHECKPOINT_DIR_NAME
    ).exists()


@pytest.mark.parametrize("mutation", ("short", "reordered", "duplicate"))
def test_preflight_rejects_noncanonical_population_before_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    run, replay, rows, _gold, postseal = _source_fixture()
    if mutation == "short":
        changed = rows[:-1]
    elif mutation == "reordered":
        changed = tuple(reversed(rows))
    else:
        duplicate = dict(rows[-1])
        duplicate["question_id"] = rows[0]["question_id"]
        changed = (*rows[:-1], duplicate)
    monkeypatch.setattr(
        judge.answer_cli,
        "load_verified_answer_run",
        lambda *_args, **_kwargs: (run, replay, changed),
    )
    monkeypatch.setattr(
        judge,
        "load_locked_typed_final_gold",
        lambda **_kwargs: pytest.fail("gold opened for invalid answer population"),
    )

    with pytest.raises(
        judge.LockedSemanticGlobalTerminalFull100JudgeError,
        match="population/order",
    ):
        judge.run_preflight(_preflight_args(tmp_path, run, replay, postseal))


def test_fully_resealed_extra_provider_message_is_still_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, result, _ = _make_preflight(tmp_path, monkeypatch)
    artifact = read_sealed_json(
        Path(args.judge_output_root) / judge.PREFLIGHT_NAME
    )
    payload = dict(artifact.payload)
    rows = [dict(row) for row in payload["prompt_rows"]]
    messages_by_row = [list(row["messages"]) for row in rows]
    messages_by_row[0] = [
        *messages_by_row[0],
        {"role": "assistant", "content": "EVIDENCE_SENTINEL"},
    ]
    population = preflight_fast_completion_prompts(
        messages_by_row, max_prompt_tokens=judge.DEFAULT_MAX_PROMPT_TOKENS
    )
    input_receipts: list[str] = []
    for row, messages, receipt in zip(
        rows, messages_by_row, population.ordered_rows, strict=True
    ):
        body = dict(row)
        body.pop("prompt_row_receipt_sha256")
        body["messages"] = messages
        body["messages_sha256"] = receipt.messages_sha256
        body["prompt_token_proxy"] = receipt.prompt_token_proxy
        input_receipt = identity_sha256(judge._judge_input_body(body, messages))  # noqa: SLF001
        body["judge_input_receipt_sha256"] = input_receipt
        row.clear()
        row.update(
            {**body, "prompt_row_receipt_sha256": identity_sha256(body)}
        )
        input_receipts.append(input_receipt)
    payload["prompt_rows"] = rows
    payload["prompt_population"] = population.model_dump()
    payload["prompt_population_sha256"] = population.prompt_population_sha256
    payload["judge_input_population_sha256"] = identity_sha256(input_receipts)
    resealed = SealedArtifact(
        Path("resealed-preflight.json"), _sha("resealed preflight"), payload
    )

    assert result["question_count"] == 100
    with pytest.raises(
        judge.LockedSemanticGlobalTerminalFull100JudgeError,
        match="leaked non-contract input",
    ):
        judge._validate_preflight(resealed)  # noqa: SLF001


class _FakeCompletions:
    def __init__(self) -> None:
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
                    message=SimpleNamespace(content="CORRECT: equivalent."),
                ),
            ),
            id=f"fake-sol-{call_number:03d}",
            model=judge.DEFAULT_MODEL,
            usage=None,
        )


class _FakeClient:
    max_retries = 0

    def __init__(self) -> None:
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _provider_args(
    args: SimpleNamespace, preflight_sha: str, release_sha: str
) -> SimpleNamespace:
    return SimpleNamespace(
        api_key_env="SEALED_SOL_KEY",
        authorized_provider_calls=100,
        enable_provider=True,
        expected_judge_preflight_sha256=preflight_sha,
        expected_release_sha256=release_sha,
        gateway_url=judge.DEFAULT_GATEWAY_URL,
        judge_output_root=args.judge_output_root,
        max_concurrency=4,
        model=judge.DEFAULT_MODEL,
    )


def test_complete_seven_phase_lifecycle_is_exact100_and_score_is_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, preflight_result, _rows = _make_preflight(tmp_path, monkeypatch)
    _release_args, release_result = _release(
        args, preflight_result["preflight_sha256"]
    )
    client = _FakeClient()
    monkeypatch.setattr(judge, "load_dotenv", lambda: None)
    monkeypatch.setenv("SEALED_SOL_KEY", "local-test-key")
    monkeypatch.setattr(
        judge.judging,
        "_make_provider_client",
        lambda *_args, **_kwargs: client,
    )
    provider_args = _provider_args(
        args,
        preflight_result["preflight_sha256"],
        release_result["release_sha256"],
    )

    provider_result = judge.run_provider(provider_args)
    assert provider_result["physical_provider_calls"] == 100
    assert provider_result["checkpoint_hits"] == 0
    assert len(client.completions.calls) == 100
    assert client.closed is True
    rendered = {
        tuple((row["role"], row["content"]) for row in call["messages"])
        for call in client.completions.calls
    }
    assert len(rendered) == 100
    assert all(
        set(call) == {"max_tokens", "messages", "model"}
        for call in client.completions.calls
    )
    assert all("EVIDENCE_SENTINEL" not in str(call) for call in client.completions.calls)

    checkpoint = Path(args.judge_output_root) / judge.CHECKPOINT_DIR_NAME
    assert len(tuple(checkpoint.glob("*.request.json"))) == 100
    assert len(tuple(checkpoint.glob("*.response.json"))) == 100

    provider_args.authorized_provider_calls = 0
    monkeypatch.setattr(
        judge,
        "load_dotenv",
        lambda: pytest.fail("environment opened for complete checkpoint replay"),
    )
    complete = judge.run_provider(provider_args)
    assert complete["physical_provider_calls"] == 0
    assert complete["checkpoint_hits"] == 100

    offline = SimpleNamespace(
        expected_judge_preflight_sha256=preflight_result["preflight_sha256"],
        expected_release_sha256=release_result["release_sha256"],
        gateway_url=judge.DEFAULT_GATEWAY_URL,
        judge_output_root=args.judge_output_root,
        max_concurrency=4,
        model=judge.DEFAULT_MODEL,
    )
    materialized = judge.run_materialize(offline)
    replayed = judge.run_replay(
        SimpleNamespace(
            **vars(offline), expected_judge_sha256=materialized["judge_sha256"]
        )
    )

    # Scoring and score replay must not inspect or instantiate provider state.
    monkeypatch.setattr(
        judge,
        "_checkpoint_batch",
        lambda *_args, **_kwargs: pytest.fail("score opened provider journals"),
    )
    score_args = SimpleNamespace(
        expected_judge_preflight_sha256=preflight_result["preflight_sha256"],
        expected_judge_replay_sha256=replayed["judge_replay_sha256"],
        expected_judge_sha256=materialized["judge_sha256"],
        expected_release_sha256=release_result["release_sha256"],
        judge_output_root=args.judge_output_root,
    )
    scored = judge.run_score(score_args)
    score_replayed = judge.run_score_replay(
        SimpleNamespace(
            **vars(score_args), expected_score_sha256=scored["score_sha256"]
        )
    )
    judge_artifact, score_artifact, verdicts = judge.load_verified_judge_score(
        args.judge_output_root,
        expected_preflight_sha256=preflight_result["preflight_sha256"],
        expected_release_sha256=release_result["release_sha256"],
        expected_judge_sha256=materialized["judge_sha256"],
        expected_judge_replay_sha256=replayed["judge_replay_sha256"],
        expected_score_sha256=scored["score_sha256"],
        expected_score_replay_sha256=score_replayed["score_replay_sha256"],
    )

    assert materialized["physical_provider_calls"] == 0
    assert replayed["byte_identical"] is True
    assert replayed["judge_replay_sha256"] == judge_artifact.sha256
    assert scored["physical_provider_calls"] == 0
    assert scored["correct"] == 100
    assert scored["accuracy"] == 1.0
    assert score_replayed["byte_identical"] is True
    assert score_replayed["score_replay_sha256"] == score_artifact.sha256
    assert len(verdicts) == 100
    with pytest.raises(
        judge.LockedSemanticGlobalTerminalFull100JudgeError,
        match="score/replay artifacts changed",
    ):
        judge.load_verified_judge_score(
            args.judge_output_root,
            expected_preflight_sha256=preflight_result["preflight_sha256"],
            expected_release_sha256=release_result["release_sha256"],
            expected_judge_sha256=materialized["judge_sha256"],
            expected_judge_replay_sha256=replayed["judge_replay_sha256"],
            expected_score_sha256=scored["score_sha256"],
            expected_score_replay_sha256=_sha("foreign score replay"),
        )


def test_release_owns_root_unsafe_retry_fails_and_cli_has_no_ordinal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, preflight_result, _rows = _make_preflight(tmp_path, monkeypatch)
    _release_args, release_result = _release(
        args, preflight_result["preflight_sha256"]
    )
    preflight, prompts, _ = judge._read_preflight(  # noqa: SLF001
        args.judge_output_root, preflight_result["preflight_sha256"]
    )
    release = judge._read_release(  # noqa: SLF001
        args.judge_output_root,
        release_result["release_sha256"],
        preflight=preflight,
    )
    with pytest.raises(
        judge.LockedSemanticGlobalTerminalFull100JudgeError,
        match="release changed",
    ):
        judge._validate_release(  # noqa: SLF001
            release, preflight=preflight, output_root=tmp_path / "foreign-root"
        )

    provider_args = _provider_args(
        args,
        preflight_result["preflight_sha256"],
        release_result["release_sha256"],
    )
    runtime = judge._runtime(  # noqa: SLF001
        preflight, release, prompts, args=provider_args, client=_FakeClient()
    )
    try:
        runtime._reserve(runtime._unique_order[0])  # noqa: SLF001
    finally:
        runtime.close()
    monkeypatch.setattr(
        judge,
        "load_dotenv",
        lambda: pytest.fail("environment opened after incomplete request"),
    )
    with pytest.raises(
        judge.LockedSemanticGlobalTerminalFull100JudgeError,
        match="unsafe retry forbidden",
    ):
        judge.run_provider(provider_args)

    parser = judge.build_parser()
    actions = next(
        action
        for action in parser._actions  # noqa: SLF001
        if getattr(action, "choices", None)
    )
    assert set(actions.choices) == {
        "approve-release",
        "materialize",
        "preflight",
        "provider-run",
        "replay",
        "score",
        "score-replay",
    }
    for command in actions.choices.values():
        assert not any(
            "ordinal" in option
            for action in command._actions  # noqa: SLF001
            for option in action.option_strings
        )
