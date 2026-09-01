from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain.discourse import identity_sha256
from tests.test_fast_completion_runtime import _FakeClient
from tests.test_matched_eval_population import _publish, _retrieval
from tools.matched_eval import judging, live
from tools.matched_eval.artifacts import publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import MatchedEvalContractError


def _run_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, count: int = 2
) -> tuple[Path, Path, live.S0V2AnswerRunResult]:
    retrieval = _publish(tmp_path, _retrieval(count))
    output = tmp_path / "matched"
    client = _FakeClient(output / live.CHECKPOINT_DIR_NAME)
    monkeypatch.setenv("MATCHED_TEST_KEY", "test-key")
    monkeypatch.setattr(live, "_make_provider_client", lambda *_args: client)
    result = live.run_s0_v2_answers(
        retrieval_path=retrieval,
        output_root=output,
        enable_provider=True,
        authorized_provider_calls=count,
        api_key_env="MATCHED_TEST_KEY",
        max_concurrency=2,
        expected_retrieval_sha256=None,
        expected_question_count=count,
    )
    return retrieval, output, result


def test_answer_run_replay_and_verified_plane_are_byte_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retrieval, output, result = _run_answers(tmp_path, monkeypatch)

    assert result.physical_provider_calls == 2
    assert result.checkpoint_hits == 0
    assert result.runtime_ledger_artifact.payload["total_provider_calls"] == 2
    assert result.answer_artifact.payload["gold_loaded"] is False

    replay = live.replay_s0_v2_answers(
        retrieval_path=retrieval,
        output_root=output,
        expected_run_sha256=result.answer_artifact.sha256,
        max_concurrency=2,
        expected_retrieval_sha256=None,
        expected_question_count=2,
    )
    assert replay.run_sha256 == replay.replay_sha256
    assert len(replay.rows) == 2
    assert read_sealed_json(output / live.RUNTIME_LEDGER_REPLAY_NAME).sha256 == (
        result.runtime_ledger_artifact.sha256
    )

    verified = live.load_verified_s0_v2_answer_plane(
        output / live.ANSWER_RUN_NAME,
        output / live.ANSWER_REPLAY_NAME,
        expected_run_sha256=result.answer_artifact.sha256,
        retrieval_path=retrieval,
        max_concurrency=2,
        expected_retrieval_sha256=None,
        expected_question_count=2,
    )
    assert verified.rows == replay.rows
    assert verified.runtime_ledger["total_provider_calls"] == 2


def test_answer_authorization_fails_before_output_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retrieval = _publish(tmp_path, _retrieval(2))
    output = tmp_path / "matched"
    monkeypatch.setenv("MATCHED_TEST_KEY", "test-key")

    with pytest.raises(MatchedEvalContractError, match="exactly equal 2"):
        live.run_s0_v2_answers(
            retrieval_path=retrieval,
            output_root=output,
            enable_provider=True,
            authorized_provider_calls=1,
            api_key_env="MATCHED_TEST_KEY",
            expected_retrieval_sha256=None,
            expected_question_count=2,
        )

    assert not output.exists()


def test_existing_answer_run_replays_before_provider_client_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retrieval, output, first = _run_answers(tmp_path, monkeypatch)

    population_loads = 0
    original_population_loader = live.load_s0_population

    def counted_population_loader(*args, **kwargs):
        nonlocal population_loads
        population_loads += 1
        return original_population_loader(*args, **kwargs)

    def forbidden_client(*_args):
        raise AssertionError("provider client must not be built for a sealed run")

    monkeypatch.setattr(live, "load_s0_population", counted_population_loader)
    monkeypatch.setattr(live, "_make_provider_client", forbidden_client)
    resumed = live.run_s0_v2_answers(
        retrieval_path=retrieval,
        output_root=output,
        enable_provider=True,
        authorized_provider_calls=2,
        api_key_env="MATCHED_TEST_KEY",
        max_concurrency=2,
        expected_retrieval_sha256=None,
        expected_question_count=2,
    )

    assert resumed.answer_artifact.sha256 == first.answer_artifact.sha256
    assert resumed.physical_provider_calls == 0
    assert resumed.checkpoint_hits == 2
    assert population_loads == 1


def test_incompatible_existing_answer_run_fails_before_provider_client_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retrieval, output, _first = _run_answers(tmp_path, monkeypatch)

    def forbidden_client(*_args):
        raise AssertionError("provider client must not be built for a sealed run")

    monkeypatch.setattr(live, "_make_provider_client", forbidden_client)
    with pytest.raises((MatchedEvalContractError, ValueError)):
        live.run_s0_v2_answers(
            retrieval_path=retrieval,
            output_root=output,
            enable_provider=True,
            authorized_provider_calls=2,
            api_key_env="MATCHED_TEST_KEY",
            max_concurrency=1,
            expected_retrieval_sha256=None,
            expected_question_count=2,
        )


class _SolCompletions:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(
            id=f"sol-{len(self.requests)}",
            model="codex_sdk/gpt-5.6-sol-test",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="CORRECT — equivalent."),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )


class _SolClient:
    def __init__(self) -> None:
        self.max_retries = 0
        self.completions = _SolCompletions()
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _synthetic_gold_loader(
    *, answer_plane: live.VerifiedS0V2AnswerPlane, **_kwargs
):
    rows = []
    for row in answer_plane.rows:
        question = f"What was choice {row.ordinal}?"
        dated = f"[Question asked at 2026/08/{row.ordinal + 1:02d}]\n{question}"
        rows.append(
            judging._GoldRow(
                ordinal=row.ordinal,
                question_id=row.question_id,
                question=question,
                question_sha256=row.question_sha256,
                dated_question=dated,
                dated_question_sha256=row.dated_question_sha256,
                reference=row.prediction,
                reference_sha256=row.prediction_sha256,
                category="synthetic",
            )
        )
    return tuple(rows), identity_sha256(
        [{"ordinal": row.ordinal, "reference": row.reference_sha256} for row in rows]
    )


def test_judge_runs_only_after_verified_answer_replay_and_replays_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retrieval, output, answer = _run_answers(tmp_path, monkeypatch)
    live.replay_s0_v2_answers(
        retrieval_path=retrieval,
        output_root=output,
        expected_run_sha256=answer.answer_artifact.sha256,
        max_concurrency=2,
        expected_retrieval_sha256=None,
        expected_question_count=2,
    )

    order: list[str] = []
    original_loader = judging.load_verified_s0_v2_answer_plane

    def verified_loader(*args, **kwargs):
        order.append("answer_verified")
        return original_loader(*args, **kwargs)

    def gold_loader(*, answer_plane, **_kwargs):
        order.append("gold_loaded")
        return _synthetic_gold_loader(answer_plane=answer_plane)

    sol = _SolClient()
    monkeypatch.setattr(judging, "load_verified_s0_v2_answer_plane", verified_loader)
    monkeypatch.setattr(judging, "_load_gold", gold_loader)
    monkeypatch.setattr(judging, "_make_provider_client", lambda *_args: sol)
    monkeypatch.setenv("MATCHED_TEST_KEY", "test-key")

    result = judging.run_s0_v2_judge(
        answer_run_path=output / live.ANSWER_RUN_NAME,
        answer_replay_path=output / live.ANSWER_REPLAY_NAME,
        expected_answer_run_sha256=answer.answer_artifact.sha256,
        retrieval_path=retrieval,
        dataset_path=tmp_path / "unused-dataset.json",
        split_path=tmp_path / "unused-split.json",
        output_root=output,
        enable_provider=True,
        authorized_provider_calls=2,
        api_key_env="MATCHED_TEST_KEY",
        max_concurrency=2,
        expected_retrieval_sha256=None,
        expected_question_count=2,
    )
    assert order[:2] == ["answer_verified", "gold_loaded"]
    assert result.correct == 2
    assert result.physical_provider_calls == 2
    assert result.score_ledger_artifact.payload["aggregate"]["candidate_correct"] == 2
    assert result.judge_artifact.payload["aggregate"]["normalized_exact_match"] == 2

    order.clear()

    def forbidden_client(*_args):
        raise AssertionError("provider client must not be built for a sealed judge")

    monkeypatch.setattr(judging, "_make_provider_client", forbidden_client)
    resumed = judging.run_s0_v2_judge(
        answer_run_path=output / live.ANSWER_RUN_NAME,
        answer_replay_path=output / live.ANSWER_REPLAY_NAME,
        expected_answer_run_sha256=answer.answer_artifact.sha256,
        retrieval_path=retrieval,
        dataset_path=tmp_path / "unused-dataset.json",
        split_path=tmp_path / "unused-split.json",
        output_root=output,
        enable_provider=True,
        authorized_provider_calls=2,
        api_key_env="MATCHED_TEST_KEY",
        max_concurrency=2,
        expected_retrieval_sha256=None,
        expected_question_count=2,
    )
    assert order == ["answer_verified", "gold_loaded"]
    assert resumed.physical_provider_calls == 0
    assert resumed.checkpoint_hits == 2
    assert resumed.judge_artifact.sha256 == result.judge_artifact.sha256
    assert resumed.score_ledger_artifact.sha256 == result.score_ledger_artifact.sha256

    replay = judging.replay_s0_v2_judge(
        answer_run_path=output / live.ANSWER_RUN_NAME,
        answer_replay_path=output / live.ANSWER_REPLAY_NAME,
        expected_answer_run_sha256=answer.answer_artifact.sha256,
        expected_judge_sha256=result.judge_artifact.sha256,
        retrieval_path=retrieval,
        dataset_path=tmp_path / "unused-dataset.json",
        split_path=tmp_path / "unused-split.json",
        output_root=output,
        max_concurrency=2,
        expected_retrieval_sha256=None,
        expected_question_count=2,
    )
    assert replay.correct == 2
    assert replay.physical_provider_calls == 0
    assert replay.judge_artifact.sha256 == result.judge_artifact.sha256
    assert replay.score_ledger_artifact.sha256 == result.score_ledger_artifact.sha256


def test_existing_judge_fails_closed_before_provider_client_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retrieval, output, answer = _run_answers(tmp_path, monkeypatch)
    live.replay_s0_v2_answers(
        retrieval_path=retrieval,
        output_root=output,
        expected_run_sha256=answer.answer_artifact.sha256,
        max_concurrency=2,
        expected_retrieval_sha256=None,
        expected_question_count=2,
    )
    publish_sealed_json(output / judging.JUDGE_NAME, {"format": "incompatible"})
    monkeypatch.setattr(judging, "_load_gold", _synthetic_gold_loader)

    def forbidden_client(*_args):
        raise AssertionError("provider client must not be built for a sealed judge")

    monkeypatch.setattr(judging, "_make_provider_client", forbidden_client)
    with pytest.raises(RuntimeError, match="provider client is required"):
        judging.run_s0_v2_judge(
            answer_run_path=output / live.ANSWER_RUN_NAME,
            answer_replay_path=output / live.ANSWER_REPLAY_NAME,
            expected_answer_run_sha256=answer.answer_artifact.sha256,
            retrieval_path=retrieval,
            dataset_path=tmp_path / "unused-dataset.json",
            split_path=tmp_path / "unused-split.json",
            output_root=output,
            enable_provider=True,
            authorized_provider_calls=2,
            api_key_env="MATCHED_TEST_KEY",
            max_concurrency=2,
            expected_retrieval_sha256=None,
            expected_question_count=2,
        )
