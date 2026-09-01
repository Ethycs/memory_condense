from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.benchmark import build_judge_prompt
from tools import run_reduced_specialist_judge as judge
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.typed_memory_final_judging import TypedFinalJudgeGoldRow


def _sha(text: str) -> str:
    return quote_sha256(text)


def _source_and_gold() -> tuple[
    tuple[judge.AnswerSeamRow, ...],
    tuple[TypedFinalJudgeGoldRow, ...],
]:
    source: list[judge.AnswerSeamRow] = []
    gold: list[TypedFinalJudgeGoldRow] = []
    for index, ordinal in enumerate(judge.EXACT_ORDINALS):
        question_id = f"question-{ordinal:03d}"
        question = f"What was the value at ordinal {ordinal}?"
        dated = f"[Question asked at 2026/08/28 10:00]\n{question}"
        reference = f"reference value {ordinal}"
        prediction = reference if index % 2 == 0 else f"value {ordinal}"
        unsigned = {
            "dated_question_sha256": _sha(dated),
            "ordinal": ordinal,
            "prediction": prediction,
            "prediction_sha256": _sha(prediction),
            "question_id": question_id,
            "question_sha256": _sha(question),
        }
        source.append(
            judge.AnswerSeamRow(
                ordinal=ordinal,
                question_id=question_id,
                question_sha256=_sha(question),
                dated_question_sha256=_sha(dated),
                prediction=prediction,
                prediction_sha256=_sha(prediction),
                answer_row_sha256=identity_sha256(unsigned),
            )
        )
        gold.append(
            TypedFinalJudgeGoldRow(
                ordinal=ordinal,
                question_id=question_id,
                question=question,
                question_sha256=_sha(question),
                dated_question=dated,
                dated_question_sha256=_sha(dated),
                reference=reference,
                reference_sha256=_sha(reference),
                category="synthetic",
            )
        )
    return tuple(source), tuple(gold)


def _answer_payload(rows: tuple[judge.AnswerSeamRow, ...]) -> dict[str, Any]:
    return {
        "format": judge.ANSWER_RUN_FORMAT,
        "gold_loaded": False,
        "judge_rows": [row.projection() for row in rows],
        "physical_provider_calls_during_materialization": 0,
        "question_count": judge.QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def _artifact(label: str, payload: dict[str, Any] | None = None) -> SealedArtifact:
    return SealedArtifact(Path(f"{label}.json"), _sha(label), payload or {"label": label})


def _preflight(
    tmp_path: Path,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    source, gold = _source_and_gold()
    run = _artifact("specialist answer", _answer_payload(source))
    replay = SealedArtifact(Path("answer-replay.json"), run.sha256, run.payload)
    payload, _prompts = judge.build_preflight_payload(
        run=run,
        replay=replay,
        answer_rows=source,
        gold_rows=gold,
        gold_population_sha256=_sha("gold population"),
        model=judge.DEFAULT_SOL_MODEL,
        gateway_url="https://gateway.invalid/v1",
        max_concurrency=3,
    )
    artifact, _ = publish_sealed_json(tmp_path / judge.PREFLIGHT_NAME, payload)
    _prompts, rows = judge.validate_preflight_artifact(artifact)
    return artifact, rows


def test_answer_seam_accepts_only_exact_v2_rows_and_byte_identical_replay(
    tmp_path: Path,
) -> None:
    source, _gold = _source_and_gold()
    payload = _answer_payload(source)
    run, _ = publish_sealed_json(tmp_path / judge.ANSWER_RUN_NAME, payload)
    replay, _ = publish_sealed_json(tmp_path / judge.ANSWER_REPLAY_NAME, payload)

    loaded_run, loaded_replay, loaded_rows = judge.load_verified_answer_seam(
        answer_run_path=run.path,
        answer_replay_path=replay.path,
        expected_answer_run_sha256=run.sha256,
        expected_answer_replay_sha256=replay.sha256,
    )

    assert loaded_run.sha256 == loaded_replay.sha256 == run.sha256
    assert tuple(row.ordinal for row in loaded_rows) == judge.EXACT_ORDINALS

    extra = _answer_payload(source)
    extra["judge_rows"][0]["prediction_source"] = "unexpected alternative seam"
    damaged_run, _ = publish_sealed_json(tmp_path / "extra-run.json", extra)
    damaged_replay, _ = publish_sealed_json(tmp_path / "extra-replay.json", extra)
    with pytest.raises(
        judge.ReducedSpecialistJudgeError,
        match="row schema changed at ordinal 7",
    ):
        judge.load_verified_answer_seam(
            answer_run_path=damaged_run.path,
            answer_replay_path=damaged_replay.path,
            expected_answer_run_sha256=damaged_run.sha256,
            expected_answer_replay_sha256=damaged_replay.sha256,
        )

    with pytest.raises(
        judge.ReducedSpecialistJudgeError,
        match="not byte-identical",
    ):
        judge.load_verified_answer_seam(
            answer_run_path=run.path,
            answer_replay_path=damaged_replay.path,
            expected_answer_run_sha256=run.sha256,
            expected_answer_replay_sha256=damaged_replay.sha256,
        )


def test_preflight_verifies_answer_before_gold_and_seals_dated_standard_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, gold = _source_and_gold()
    run = _artifact("verified specialist answer", _answer_payload(source))
    replay = SealedArtifact(Path("answer-replay.json"), run.sha256, run.payload)
    calls: list[str] = []

    def verified(**_kwargs: Any):
        calls.append("answer_replay_verified")
        return run, replay, source

    def locked_gold(**kwargs: Any):
        calls.append("gold_opened")
        assert kwargs["allow_subset"] is True
        assert tuple(row["ordinal"] for row in kwargs["source_rows"]) == (
            judge.EXACT_ORDINALS
        )
        return gold, _sha("locked gold")

    monkeypatch.setattr(judge, "load_verified_answer_seam", verified)
    monkeypatch.setattr(judge, "load_locked_typed_final_gold", locked_gold)
    args = SimpleNamespace(
        answer_replay=None,
        answer_root=tmp_path / "answer",
        answer_run=None,
        dataset=tmp_path / "locked.json",
        expected_answer_replay_sha256=replay.sha256,
        expected_answer_run_sha256=run.sha256,
        gateway_url="https://gateway.invalid/v1",
        judge_output_root=tmp_path / "judge",
        max_concurrency=3,
        model=judge.DEFAULT_SOL_MODEL,
        split=tmp_path / "split.json",
    )

    result = judge.run_preflight(args)
    artifact = read_sealed_json(Path(args.judge_output_root) / judge.PREFLIGHT_NAME)
    prompts, rows = judge.validate_preflight_artifact(artifact)

    assert calls == ["answer_replay_verified", "gold_opened"]
    assert result["required_authorized_provider_calls"] == 10
    assert result["physical_provider_calls"] == 0
    assert tuple(row["ordinal"] for row in rows) == judge.EXACT_ORDINALS
    assert list(prompts[0]) == build_judge_prompt(
        gold[0].dated_question,
        gold[0].reference,
        source[0].prediction,
    )
    assert artifact.payload["prompt_population"]["unique_prompt_count"] == 10


def test_provider_requires_exact_ten_call_authorization_before_environment_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, _rows = _preflight(tmp_path)
    monkeypatch.setattr(
        judge,
        "load_dotenv",
        lambda: pytest.fail("environment accessed before exact authorization"),
    )
    args = SimpleNamespace(
        api_key_env="LITELLM_KEY",
        authorized_provider_calls=9,
        enable_provider=True,
        expected_judge_preflight_sha256=artifact.sha256,
        gateway_url=artifact.payload["gateway_url"],
        judge_output_root=tmp_path,
        max_concurrency=artifact.payload["max_concurrency"],
        model=artifact.payload["model"],
    )

    with pytest.raises(
        judge.ReducedSpecialistJudgeError,
        match="exact authorization for 10 calls",
    ):
        judge.run_provider(args)


class _FakeBatch:
    def __init__(self, rows: tuple[dict[str, Any], ...]) -> None:
        self.logical_completions = tuple(
            "CORRECT\nThe answer is supported." if index % 2 == 0 else "INCORRECT"
            for index in range(judge.QUESTION_COUNT)
        )
        self.unique_records = tuple(
            SimpleNamespace(
                call_key_sha256=_sha(f"call {index}"),
                checkpoint_hit=True,
                completion=completion,
                messages_sha256=row["messages_sha256"],
                physical_call=False,
                request_journal_sha256=_sha(f"request {index}"),
                response_journal_sha256=_sha(f"response {index}"),
            )
            for index, (row, completion) in enumerate(
                zip(rows, self.logical_completions, strict=True)
            )
        )
        self.usage = SimpleNamespace(
            checkpoint_hits=judge.QUESTION_COUNT,
            logical_calls=judge.QUESTION_COUNT,
            physical_calls=0,
            unique_calls=judge.QUESTION_COUNT,
        )

    def model_dump(self) -> dict[str, Any]:
        return {
            "logical_completions": list(self.logical_completions),
            "prompt_population": {"logical_prompt_count": judge.QUESTION_COUNT},
            "provenance": {"model": judge.DEFAULT_SOL_MODEL},
            "runtime_identity_sha256": _sha("runtime"),
            "unique_records": [
                {
                    "call_key_sha256": row.call_key_sha256,
                    "checkpoint_hit": row.checkpoint_hit,
                    "completion": row.completion,
                    "messages_sha256": row.messages_sha256,
                    "physical_call": row.physical_call,
                    "request_journal_sha256": row.request_journal_sha256,
                    "response_journal_sha256": row.response_journal_sha256,
                }
                for row in self.unique_records
            ],
            "usage": {
                "checkpoint_hits": judge.QUESTION_COUNT,
                "logical_calls": judge.QUESTION_COUNT,
                "physical_calls": 0,
                "unique_calls": judge.QUESTION_COUNT,
            },
        }


def test_materialize_and_replay_are_checkpoint_only_and_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, rows = _preflight(tmp_path)
    batch = _FakeBatch(rows)
    monkeypatch.setattr(judge, "_run_batch", lambda *_args, **_kwargs: batch)
    common = {
        "expected_judge_preflight_sha256": artifact.sha256,
        "gateway_url": artifact.payload["gateway_url"],
        "judge_output_root": tmp_path,
        "max_concurrency": artifact.payload["max_concurrency"],
        "model": artifact.payload["model"],
    }

    materialized = judge.run_materialize(SimpleNamespace(**common))
    replayed = judge.run_replay(
        SimpleNamespace(
            **common,
            expected_judge_sha256=materialized["judge_sha256"],
            expected_score_sha256=materialized["score_sha256"],
        )
    )

    assert materialized["correct"] == 5
    assert materialized["physical_provider_calls"] == 0
    assert replayed["byte_identical"] is True
    assert replayed["judge_replay_sha256"] == materialized["judge_sha256"]
    assert replayed["score_replay_sha256"] == materialized["score_sha256"]


def test_provider_parser_has_no_answer_or_gold_authority() -> None:
    provider = judge.build_parser().parse_args(
        [
            "provider-run",
            "--expected-judge-preflight-sha256",
            "a" * 64,
            "--authorized-provider-calls",
            "10",
        ]
    )

    assert provider.model == judge.DEFAULT_SOL_MODEL
    assert provider.judge_output_root == judge.DEFAULT_JUDGE_ROOT
    assert not hasattr(provider, "dataset")
    assert not hasattr(provider, "answer_run")
    assert not hasattr(provider, "expected_answer_run_sha256")
