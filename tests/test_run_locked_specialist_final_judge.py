from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.benchmark import build_judge_prompt
from tools import run_locked_specialist_final_judge as judge
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.typed_memory_final_arm import judge_row_projection
from tools.matched_eval.typed_memory_final_judging import TypedFinalJudgeGoldRow


def _sha(text: str) -> str:
    return quote_sha256(text)


def _source_and_gold() -> tuple[
    tuple[dict[str, Any], ...],
    tuple[TypedFinalJudgeGoldRow, ...],
]:
    source: list[dict[str, Any]] = []
    gold: list[TypedFinalJudgeGoldRow] = []
    for ordinal in judge.EXACT_ORDINALS:
        question_id = f"question-{ordinal:03d}"
        question = f"What was the unique value at ordinal {ordinal}?"
        dated = f"[Question asked at 2026/08/28 10:{ordinal:02d}]\n{question}"
        reference = f"reference value {ordinal}"
        parent = f"parent value {ordinal}"
        prediction = reference if ordinal % 2 == 0 else f"prediction {ordinal}"
        body = {
            "changed_from_parent": prediction != parent,
            "dated_question_sha256": _sha(dated),
            "ordinal": ordinal,
            "parent_prediction_sha256": _sha(parent),
            "prediction": prediction,
            "prediction_sha256": _sha(prediction),
            "prediction_source": (
                "specialist_scoped_validated_replacement_v1"
                if ordinal % 2 == 0
                else "sealed_parent_passthrough_v1"
            ),
            "question_id": question_id,
            "question_sha256": _sha(question),
            "result_receipt_sha256": _sha(f"result {ordinal}"),
            "route_id": "numeric_specialist_v1",
        }
        source.append({**body, "source_row_sha256": identity_sha256(body)})
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


def _answer_payload(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    return {
        "completion_batch": {"checkpoint_only": True},
        "format": judge.ANSWER_RUN_FORMAT,
        "gold_loaded": False,
        "judge_rows": [judge_row_projection(row) for row in rows],
        "model": "codex_sdk/gpt-5.6-terra",
        "physical_provider_calls_during_materialization": 0,
        "question_count": judge.QUESTION_COUNT,
        "questions": [dict(row) for row in rows],
        "retained_transformer_token_state_bytes": 0,
    }


def _artifact(label: str, payload: dict[str, Any]) -> SealedArtifact:
    return SealedArtifact(Path(f"{label}.json"), _sha(label), payload)


def _preflight(
    tmp_path: Path,
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    source, gold = _source_and_gold()
    run = _artifact("specialist answer", _answer_payload(source))
    replay = SealedArtifact(Path("answer-replay.json"), run.sha256, run.payload)
    payload, _ = judge.build_preflight_payload(
        run=run,
        replay=replay,
        source_rows=tuple(judge_row_projection(row) for row in source),
        gold_rows=gold,
        gold_population_sha256=_sha("gold population"),
        model=judge.DEFAULT_SOL_MODEL,
        gateway_url="https://gateway.invalid/v1",
        max_concurrency=3,
    )
    artifact, _ = publish_sealed_json(tmp_path / judge.PREFLIGHT_NAME, payload)
    prompts, rows = judge.validate_preflight_artifact(artifact)
    return artifact, prompts, rows


def test_answer_source_requires_typed_full100_and_byte_identical_replay(
    tmp_path: Path,
) -> None:
    source, _gold = _source_and_gold()
    payload = _answer_payload(source)
    run, _ = publish_sealed_json(tmp_path / judge.ANSWER_RUN_NAME, payload)
    replay, _ = publish_sealed_json(tmp_path / judge.ANSWER_REPLAY_NAME, payload)

    loaded_run, loaded_replay, rows = judge.load_verified_answer_judge_source(
        answer_run_path=run.path,
        answer_replay_path=replay.path,
        expected_answer_run_sha256=run.sha256,
        expected_answer_replay_sha256=replay.sha256,
    )

    assert loaded_run.sha256 == loaded_replay.sha256
    assert tuple(row["ordinal"] for row in rows) == judge.EXACT_ORDINALS
    assert rows[0] == judge_row_projection(source[0])

    changed = _answer_payload(source)
    changed["judge_rows"][12]["route_id"] = "tampered_route"
    bad_run, _ = publish_sealed_json(tmp_path / "tampered-run.json", changed)
    bad_replay, _ = publish_sealed_json(tmp_path / "tampered-replay.json", changed)
    with pytest.raises(
        judge.LockedSpecialistFinalJudgeError,
        match="answer row changed at ordinal 12",
    ):
        judge.load_verified_answer_judge_source(
            answer_run_path=bad_run.path,
            answer_replay_path=bad_replay.path,
            expected_answer_run_sha256=bad_run.sha256,
            expected_answer_replay_sha256=bad_replay.sha256,
        )

    with pytest.raises(
        judge.LockedSpecialistFinalJudgeError,
        match="not byte-identical",
    ):
        judge.load_verified_answer_judge_source(
            answer_run_path=run.path,
            answer_replay_path=bad_replay.path,
            expected_answer_run_sha256=run.sha256,
            expected_answer_replay_sha256=bad_replay.sha256,
        )


def test_preflight_verifies_answer_before_gold_and_seals_100_unique_prompts(
    tmp_path: Path,
) -> None:
    source, gold = _source_and_gold()
    payload = _answer_payload(source)
    run = _artifact("verified full100 answer", payload)
    replay = SealedArtifact(Path("answer-replay.json"), run.sha256, run.payload)
    seam = tuple(judge_row_projection(row) for row in source)
    events: list[str] = []

    def source_loader(**_kwargs: Any):
        events.append("answer_replay_verified")
        return run, replay, seam

    def gold_loader(**kwargs: Any):
        events.append("gold_opened")
        assert tuple(row["ordinal"] for row in kwargs["source_rows"]) == (
            judge.EXACT_ORDINALS
        )
        return gold, _sha("locked gold")

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

    result = judge.run_preflight(
        args,
        source_loader=source_loader,
        gold_loader=gold_loader,
    )
    artifact = read_sealed_json(Path(args.judge_output_root) / judge.PREFLIGHT_NAME)
    prompts, rows = judge.validate_preflight_artifact(artifact)

    assert events == ["answer_replay_verified", "gold_opened"]
    assert result["required_authorized_provider_calls"] == 100
    assert result["physical_provider_calls"] == 0
    assert artifact.payload["format"] == judge.PREFLIGHT_FORMAT
    assert artifact.payload["answer_run_format"] == judge.ANSWER_RUN_FORMAT
    assert artifact.payload["prompt_population"]["unique_prompt_count"] == 100
    assert len(prompts) == len(rows) == 100
    assert list(prompts[0]) == build_judge_prompt(
        gold[0].question,
        gold[0].reference,
        source[0]["prediction"],
    )
    assert tuple(row["ordinal"] for row in rows) == judge.EXACT_ORDINALS


def test_runtime_is_sol_zero_retry_and_provider_requires_exact_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, prompts, _rows = _preflight(tmp_path)
    runtime = judge.build_runtime(
        artifact,
        prompts,
        output_root=tmp_path,
        model=judge.DEFAULT_SOL_MODEL,
        gateway_url=artifact.payload["gateway_url"],
        max_concurrency=artifact.payload["max_concurrency"],
        client=None,
    )
    try:
        assert runtime.provenance.model == "codex_sdk/gpt-5.6-sol"
        assert runtime.provenance.retries == 0
        assert runtime.provenance.retained_transformer_token_state_bytes == 0
        assert runtime.provenance.benchmark_provenance[
            "authorized_unique_calls"
        ] == 100
    finally:
        runtime.close()

    monkeypatch.setattr(
        judge,
        "load_dotenv",
        lambda: pytest.fail("environment accessed before exact authorization"),
    )
    args = SimpleNamespace(
        api_key_env="LITELLM_KEY",
        authorized_provider_calls=99,
        enable_provider=True,
        expected_judge_preflight_sha256=artifact.sha256,
        gateway_url=artifact.payload["gateway_url"],
        judge_output_root=tmp_path,
        max_concurrency=artifact.payload["max_concurrency"],
        model=artifact.payload["model"],
    )

    with pytest.raises(
        judge.LockedSpecialistFinalJudgeError,
        match="exact authorization for 100 calls",
    ):
        judge.run_provider(args)


class _FakeBatch:
    def __init__(self, rows: tuple[dict[str, Any], ...]) -> None:
        self.logical_completions = tuple(
            "CORRECT\nThe prediction is supported."
            if ordinal % 2 == 0
            else "INCORRECT"
            for ordinal in judge.EXACT_ORDINALS
        )
        self.unique_records = tuple(
            SimpleNamespace(
                call_key_sha256=_sha(f"call {ordinal}"),
                checkpoint_hit=True,
                completion=completion,
                messages_sha256=row["messages_sha256"],
                physical_call=False,
                request_journal_sha256=_sha(f"request {ordinal}"),
                response_journal_sha256=_sha(f"response {ordinal}"),
            )
            for ordinal, (row, completion) in enumerate(
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
            "provenance": {"model": judge.DEFAULT_SOL_MODEL, "retries": 0},
            "runtime_identity_sha256": _sha("runtime"),
            "unique_records": [vars(row) for row in self.unique_records],
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
    artifact, _prompts, rows = _preflight(tmp_path)
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

    judge_payload = read_sealed_json(tmp_path / judge.JUDGE_NAME).payload
    score_payload = read_sealed_json(tmp_path / judge.SCORE_NAME).payload
    assert materialized["correct"] == 50
    assert materialized["physical_provider_calls"] == 0
    assert judge_payload["format"] == judge.JUDGE_FORMAT
    assert score_payload["format"] == judge.SCORE_FORMAT
    assert score_payload["gate_passed"] is False
    assert replayed["byte_identical"] is True
    assert replayed["physical_provider_calls"] == 0
    assert replayed["judge_replay_sha256"] == materialized["judge_sha256"]
    assert replayed["score_replay_sha256"] == materialized["score_sha256"]


def test_provider_parser_exposes_no_gold_or_answer_authority() -> None:
    provider = judge.build_parser().parse_args(
        [
            "provider-run",
            "--expected-judge-preflight-sha256",
            "a" * 64,
            "--authorized-provider-calls",
            "100",
        ]
    )

    assert provider.model == judge.DEFAULT_SOL_MODEL
    assert provider.judge_output_root == judge.DEFAULT_JUDGE_ROOT
    assert not hasattr(provider, "dataset")
    assert not hasattr(provider, "answer_run")
    assert not hasattr(provider, "expected_answer_run_sha256")
