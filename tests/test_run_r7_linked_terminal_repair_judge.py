from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.benchmark import build_judge_prompt
from tools import run_r7_linked_terminal_repair_judge as judge
from tools.matched_eval.artifacts import SealedArtifact, publish_sealed_json
from tools.matched_eval.contracts import identity_sha256


def _sha(label: str) -> str:
    return identity_sha256({"label": label})


def _source_fixture() -> tuple[
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, Any], ...],
    tuple[judge._GoldRow, ...],  # noqa: SLF001
]:
    results: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    gold: list[judge._GoldRow] = []  # noqa: SLF001
    for index in range(judge.QUESTION_COUNT):
        question_id = f"q-{index:02d}"
        dated_question = f"2026-08-{index + 1:02d}: What is fact {index}?"
        prediction = f"prediction-{index}"
        reference = f"reference-{index}"
        source_sha = _sha(f"answer source {index}")
        results.append(
            {
                "question_id": question_id,
                "source_row_sha256": source_sha,
                "prediction": prediction,
                "prediction_sha256": quote_sha256(prediction),
            }
        )
        source_rows.append(
            {
                "dated_question_sha256": quote_sha256(dated_question),
                "format": judge.answer_cli.JUDGE_ROW_FORMAT,
                "prediction": prediction,
                "prediction_sha256": quote_sha256(prediction),
                "question_id": question_id,
                "question_sha256": quote_sha256(dated_question),
                "source_row_sha256": source_sha,
            }
        )
        gold.append(
            judge._GoldRow(  # noqa: SLF001
                question_id=question_id,
                question_sha256=quote_sha256(dated_question),
                dated_question=dated_question,
                dated_question_sha256=quote_sha256(dated_question),
                reference=reference,
                reference_sha256=quote_sha256(reference),
                category=f"category-{index % 3}",
            )
        )
    payload = {
        "a1_construction_artifact_sha256": _sha("A1 construction"),
        "a1_replay_artifact_sha256": _sha("A1 replay"),
        "format": judge.answer_cli.RUN_FORMAT,
        "gold_loaded": False,
        "judge_row_population_sha256": identity_sha256(source_rows),
        "judge_rows": source_rows,
        "physical_provider_calls_during_materialization": 0,
        "preflight_construction_artifact_sha256": _sha("answer preflight"),
        "preflight_replay_artifact_sha256": _sha("answer preflight replay"),
        "prompt_population_sha256": _sha("answer prompts"),
        "question_count": judge.QUESTION_COUNT,
        "result_count": judge.QUESTION_COUNT,
        "retained_population_sha256": _sha("answer retained"),
        "retained_transformer_token_state_bytes": 0,
        "source_construction_artifact_sha256": _sha("source construction"),
        "source_replay_artifact_sha256": _sha("source replay"),
    }
    run = SealedArtifact(Path("answer-run.json"), _sha("answer run"), payload)
    replay = SealedArtifact(Path("answer-replay.json"), run.sha256, payload)
    # The real answer loader returns result rows; the judge seam is in payload.
    judge._validate_answer_rows(run, results)  # noqa: SLF001
    return run, replay, tuple(source_rows), tuple(gold)


def _preflight_payload():
    run, replay, source, gold = _source_fixture()
    payload, prompts = judge.build_preflight_payload(
        run,
        replay,
        source,
        gold,
        gold_population_sha256=_sha("gold population"),
    )
    return run, replay, source, gold, payload, prompts


def test_preflight_boundary_is_exact_question_reference_prediction() -> None:
    _run, _replay, _source, _gold, payload, prompts = _preflight_payload()
    artifact = SealedArtifact(Path("preflight.json"), _sha("preflight"), payload)
    rebuilt, rows = judge._validate_preflight(artifact)  # noqa: SLF001

    assert rebuilt == prompts
    assert payload["required_authorized_provider_calls"] == 11
    assert payload["ordinal_cli_routing_available"] is False
    assert len(rows) == len({row["question_id"] for row in rows}) == 11
    for messages, row in zip(prompts, rows, strict=True):
        assert list(messages) == build_judge_prompt(
            row["dated_question"], row["reference"], row["prediction"]
        )
        assert "ordinal" not in row
        assert not any(
            key in row
            for key in (
                "allowed_handle_ids",
                "context",
                "evidence",
                "facts",
                "graph_links",
                "handles",
                "typed_links",
                "used_handle_ids",
            )
        )


def test_answer_replay_and_source_membership_fail_closed() -> None:
    run, replay, source, gold = _source_fixture()
    changed_payload = dict(replay.payload)
    changed_payload["question_count"] = 10
    changed_replay = SealedArtifact(replay.path, replay.sha256, changed_payload)
    with pytest.raises(
        judge.R7LinkedTerminalRepairJudgeError, match="not byte-identical"
    ):
        judge.build_preflight_payload(
            run,
            changed_replay,
            source,
            gold,
            gold_population_sha256=_sha("gold population"),
        )

    changed_source = [dict(row) for row in source]
    changed_source[0]["source_row_sha256"] = _sha("foreign source")
    with pytest.raises(
        judge.R7LinkedTerminalRepairJudgeError, match="judge source changed"
    ):
        judge.build_preflight_payload(
            run,
            replay,
            changed_source,
            gold,
            gold_population_sha256=_sha("gold population"),
        )


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


def _publish_preflight(tmp_path: Path) -> tuple[SealedArtifact, SealedArtifact]:
    _run, _replay, _source, _gold, payload, _prompts = _preflight_payload()
    construction, _ = publish_sealed_json(tmp_path / judge.PREFLIGHT_NAME, payload)
    replay, _ = publish_sealed_json(tmp_path / judge.PREFLIGHT_REPLAY_NAME, payload)
    return construction, replay


def _runtime_args(
    tmp_path: Path, construction: SealedArtifact, replay: SealedArtifact
) -> dict[str, Any]:
    return {
        "expected_judge_preflight_construction_sha256": construction.sha256,
        "expected_judge_preflight_replay_sha256": replay.sha256,
        "gateway_url": judge.DEFAULT_GATEWAY_URL,
        "judge_output_root": tmp_path,
        "max_concurrency": 4,
        "model": judge.DEFAULT_MODEL,
    }


def test_fake_provider_lifecycle_is_exact11_and_offline_after_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    construction, replay = _publish_preflight(tmp_path)
    common = _runtime_args(tmp_path, construction, replay)
    client = _FakeClient()
    monkeypatch.setattr(judge, "load_dotenv", lambda: None)
    monkeypatch.setenv("TEST_SOL_KEY", "local-test-key")
    monkeypatch.setattr(judge, "_make_provider_client", lambda *_args: client)

    provider = judge.run_provider(
        SimpleNamespace(
            **common,
            api_key_env="TEST_SOL_KEY",
            authorized_provider_calls=11,
            enable_provider=True,
        )
    )
    assert provider["physical_provider_calls"] == 11
    assert len(client.completions.calls) == 11
    assert client.closed is True
    assert all(
        set(call) == {"max_tokens", "messages", "model"}
        for call in client.completions.calls
    )
    assert all("evidence" not in str(call).casefold() for call in client.completions.calls)

    monkeypatch.setattr(
        judge,
        "load_dotenv",
        lambda: pytest.fail("environment opened for checkpoint replay"),
    )
    resumed = judge.run_provider(
        SimpleNamespace(
            **common,
            api_key_env="TEST_SOL_KEY",
            authorized_provider_calls=0,
            enable_provider=True,
        )
    )
    assert resumed["physical_provider_calls"] == 0
    assert resumed["checkpoint_hits"] == 11

    materialized = judge.run_materialize(SimpleNamespace(**common))
    replayed = judge.run_replay(
        SimpleNamespace(
            **common,
            expected_judge_sha256=materialized["judge_sha256"],
            expected_score_sha256=materialized["score_sha256"],
        )
    )
    sealed_judge, sealed_score, rows = judge.load_verified_judge_run(
        tmp_path,
        expected_preflight_construction_sha256=construction.sha256,
        expected_preflight_replay_sha256=replay.sha256,
        expected_judge_sha256=materialized["judge_sha256"],
        expected_judge_replay_sha256=replayed["judge_replay_sha256"],
        expected_score_sha256=materialized["score_sha256"],
        expected_score_replay_sha256=replayed["score_replay_sha256"],
    )
    assert materialized["correct"] == 11
    assert materialized["accuracy"] == 1.0
    assert sealed_score.payload["judge_artifact_sha256"] == sealed_judge.sha256
    assert len(rows) == 11


def test_invalid_verdict_and_incomplete_journal_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    construction, replay = _publish_preflight(tmp_path)
    common = _runtime_args(tmp_path, construction, replay)
    client = _FakeClient("MAYBE")
    monkeypatch.setattr(judge, "load_dotenv", lambda: None)
    monkeypatch.setenv("TEST_SOL_KEY", "local-test-key")
    monkeypatch.setattr(judge, "_make_provider_client", lambda *_args: client)
    judge.run_provider(
        SimpleNamespace(
            **common,
            api_key_env="TEST_SOL_KEY",
            authorized_provider_calls=11,
            enable_provider=True,
        )
    )
    with pytest.raises(judge.R7LinkedTerminalRepairJudgeError, match="invalid Sol"):
        judge.run_materialize(SimpleNamespace(**common))

    second_root = tmp_path / "unsafe"
    c2, r2 = _publish_preflight(second_root)
    preflight, preflight_replay, prompts, _rows = judge._read_preflight(  # noqa: SLF001
        second_root,
        expected_construction_sha256=c2.sha256,
        expected_replay_sha256=r2.sha256,
    )
    runtime = judge._runtime(  # noqa: SLF001
        preflight,
        preflight_replay,
        prompts,
        output_root=second_root,
        model=judge.DEFAULT_MODEL,
        gateway_url=judge.DEFAULT_GATEWAY_URL,
        max_concurrency=4,
        client=_FakeClient(),
    )
    try:
        runtime._reserve(runtime._unique_order[0])  # noqa: SLF001
    finally:
        runtime.close()
    with pytest.raises(
        judge.R7LinkedTerminalRepairJudgeError, match="unsafe retry forbidden"
    ):
        judge._read_only_checkpoint_count(second_root)  # noqa: SLF001

    parser = judge.build_parser()
    subparsers = next(
        action for action in parser._actions if getattr(action, "choices", None)  # noqa: SLF001
    )
    assert set(subparsers.choices) == {
        "materialize",
        "preflight",
        "provider-run",
        "replay",
    }
    for command in subparsers.choices.values():
        assert not any(
            "ordinal" in option
            for action in command._actions  # noqa: SLF001
            for option in action.option_strings
        )
