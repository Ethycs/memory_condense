from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import run_locked_specialist_final_judge_v3 as judge
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import MatchedEvalContractError, identity_sha256
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
        dated = f"[Question asked at 2026/08/28 10:00]\n{question}"
        reference = f"reference value {ordinal}"
        prediction = reference if ordinal % 2 == 0 else f"prediction {ordinal}"
        body = {
            "changed_from_parent": ordinal % 2 == 0,
            "changed_from_v2": ordinal % 3 == 0,
            "dated_question_sha256": _sha(dated),
            "decision_lane": "v2_fallback",
            "gold_loaded": False,
            "ordinal": ordinal,
            "parent_prediction_sha256": _sha(f"parent {ordinal}"),
            "physical_provider_calls": 0,
            "prediction": prediction,
            "prediction_sha256": _sha(prediction),
            "prediction_source": "locked_v3_v2_fallback",
            "question_id": question_id,
            "question_sha256": _sha(question),
            "retained_transformer_token_state_bytes": 0,
            "route_id": "synthetic",
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
        "format": judge.ANSWER_RUN_FORMAT,
        "gold_loaded": False,
        "judge_rows": [judge_row_projection(row) for row in rows],
        "physical_provider_calls_during_materialization": 0,
        "question_count": judge.QUESTION_COUNT,
        "questions": [dict(row) for row in rows],
        "retained_transformer_token_state_bytes": 0,
    }


def _artifact(label: str, payload: dict[str, Any]) -> SealedArtifact:
    return SealedArtifact(Path(f"{label}.json"), _sha(label), payload)


def test_v3_answer_source_requires_byte_identical_authenticated_replay(
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
    changed = _answer_payload(source)
    changed["questions"][12]["prediction"] = "tampered"
    bad, _ = publish_sealed_json(tmp_path / "tampered-replay.json", changed)
    with pytest.raises(MatchedEvalContractError, match="not byte-identical"):
        judge.load_verified_answer_judge_source(
            answer_run_path=run.path,
            answer_replay_path=bad.path,
            expected_answer_run_sha256=run.sha256,
            expected_answer_replay_sha256=bad.sha256,
        )


def test_preflight_authenticates_v3_answer_before_opening_gold(
    tmp_path: Path,
) -> None:
    source, gold = _source_and_gold()
    run = _artifact("verified-v3-answer", _answer_payload(source))
    replay = SealedArtifact(Path("answer-replay.json"), run.sha256, run.payload)
    seam = tuple(judge_row_projection(row) for row in source)
    events: list[str] = []

    def source_loader(**_kwargs: Any):
        events.append("answer_replay_authenticated")
        return run, replay, seam

    def gold_loader(**_kwargs: Any):
        events.append("gold_opened")
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

    assert events == ["answer_replay_authenticated", "gold_opened"]
    assert result["physical_provider_calls"] == 0
    assert result["required_authorized_provider_calls"] == 100
    assert artifact.payload["answer_run_format"] == judge.ANSWER_RUN_FORMAT
    assert len(prompts) == len(rows) == 100


def test_v3_runtime_is_exact_100_call_sol_and_zero_state(tmp_path: Path) -> None:
    source, gold = _source_and_gold()
    run = _artifact("runtime-v3-answer", _answer_payload(source))
    replay = SealedArtifact(Path("runtime-replay.json"), run.sha256, run.payload)
    payload, prompts = judge.build_preflight_payload(
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
        assert runtime.provenance.retries == 0
        assert runtime.provenance.retained_transformer_token_state_bytes == 0
        assert runtime.provenance.benchmark_provenance["authorized_unique_calls"] == 100
        assert runtime.provenance.benchmark_provenance["arm"] == (
            "locked_specialist_final_reconciliation_sol_judge_v3"
        )
    finally:
        runtime.close()


def test_v3_contract_restores_shared_judge_globals() -> None:
    before = (
        judge.base.ANSWER_RUN_FORMAT,
        judge.base.PREFLIGHT_NAME,
        judge.base.build_runtime,
    )
    with judge._v3_base_contract():  # noqa: SLF001
        assert judge.base.ANSWER_RUN_FORMAT == judge.ANSWER_RUN_FORMAT
        assert judge.base.PREFLIGHT_NAME == judge.PREFLIGHT_NAME
        assert judge.base.build_runtime is judge.build_runtime
    assert (
        judge.base.ANSWER_RUN_FORMAT,
        judge.base.PREFLIGHT_NAME,
        judge.base.build_runtime,
    ) == before
