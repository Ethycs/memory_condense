from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import run_locked_semantic_residual_judge_v4 as judge_v4
from tools import run_locked_semantic_residual_judge_v5 as judge
from tools import run_locked_specialist_final_judge_v2 as judge_v2
from tools.matched_eval.artifacts import SealedArtifact, publish_sealed_json, read_sealed_json
from tools.matched_eval.contracts import MatchedEvalContractError, identity_sha256
from tools.matched_eval.typed_memory_final_arm import judge_row_projection
from tools.matched_eval.typed_memory_final_judging import TypedFinalJudgeGoldRow


def _sha(text: str) -> str:
    return quote_sha256(text)


def _source_and_gold() -> tuple[tuple[dict[str, Any], ...], tuple[TypedFinalJudgeGoldRow, ...]]:
    source: list[dict[str, Any]] = []
    gold: list[TypedFinalJudgeGoldRow] = []
    for ordinal in judge.EXACT_ORDINALS:
        question_id = f"question-{ordinal:03d}"
        question = f"What was the unique value at ordinal {ordinal}?"
        dated = f"[Question asked at 2026/08/28 10:00]\n{question}"
        reference = f"reference value {ordinal}"
        prediction = reference if ordinal % 2 == 0 else f"prediction {ordinal}"
        parent = f"V4 prediction {ordinal}"
        body = {
            "changed_from_parent": prediction != parent,
            "changed_from_v3": True,
            "changed_from_v4": prediction != parent,
            "dated_question_sha256": _sha(dated),
            "decision": "candidate",
            "format": "synthetic-v5-row",
            "ordinal": ordinal,
            "parent_prediction_sha256": _sha(parent),
            "physical_provider_calls": 0,
            "prediction": prediction,
            "prediction_sha256": _sha(prediction),
            "prediction_source": "locked_sol_selected_candidate_v5",
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


def test_v5_answer_source_requires_gold_free_byte_identical_replay(tmp_path: Path) -> None:
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
    changed["gold_loaded"] = True
    bad, _ = publish_sealed_json(tmp_path / "gold-loaded-replay.json", changed)
    with pytest.raises(MatchedEvalContractError, match="not byte-identical"):
        judge.load_verified_answer_judge_source(
            answer_run_path=run.path,
            answer_replay_path=bad.path,
            expected_answer_run_sha256=run.sha256,
            expected_answer_replay_sha256=bad.sha256,
        )


def test_preflight_authenticates_v5_before_opening_gold_and_seals_100(tmp_path: Path) -> None:
    source, gold = _source_and_gold()
    run = _artifact("verified-v5-answer", _answer_payload(source))
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
        max_concurrency=4,
        model=judge.DEFAULT_SOL_MODEL,
        split=tmp_path / "split.json",
    )
    result = judge.run_preflight(args, source_loader=source_loader, gold_loader=gold_loader)
    artifact = read_sealed_json(Path(args.judge_output_root) / judge.PREFLIGHT_NAME)
    prompts, rows = judge.validate_preflight_artifact(artifact)

    assert events == ["answer_replay_authenticated", "gold_opened"]
    assert result["physical_provider_calls"] == 0
    assert result["required_authorized_provider_calls"] == 100
    assert artifact.payload["answer_run_format"] == judge.ANSWER_RUN_FORMAT
    assert artifact.payload["answer_run_sha256"] == run.sha256
    assert artifact.payload["answer_replay_sha256"] == replay.sha256
    assert len(prompts) == len(rows) == len(set(row["messages_sha256"] for row in rows)) == 100


def test_v5_runtime_is_exact_100_call_sol_retry_free_and_zero_state(tmp_path: Path) -> None:
    source, gold = _source_and_gold()
    run = _artifact("runtime-v5-answer", _answer_payload(source))
    replay = SealedArtifact(Path("runtime-replay.json"), run.sha256, run.payload)
    payload, prompts = judge.build_preflight_payload(
        run=run,
        replay=replay,
        source_rows=tuple(judge_row_projection(row) for row in source),
        gold_rows=gold,
        gold_population_sha256=_sha("gold population"),
        model=judge.DEFAULT_SOL_MODEL,
        gateway_url="https://gateway.invalid/v1",
        max_concurrency=4,
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
        assert runtime.provenance.benchmark_provenance["arm"] == "locked_semantic_residual_sol_judge_v5"
    finally:
        runtime.close()


def test_provider_fails_before_dispatch_when_checkpoint_root_is_not_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / judge.CHECKPOINT_DIR_NAME
    checkpoint.mkdir()
    called = False

    def forbidden(_args: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(judge.base, "run_provider", forbidden)
    with pytest.raises(judge.LockedSemanticResidualJudgeV5Error, match="fresh"):
        judge.run_provider(SimpleNamespace(judge_output_root=tmp_path))
    assert called is False


def test_v5_contract_restores_shared_globals_and_uses_shared_reentrant_lock() -> None:
    shared = judge.base._VERSION_CONTRACT_LOCK  # noqa: SLF001
    assert judge._BASE_LOCK is shared  # noqa: SLF001
    assert judge_v4._BASE_LOCK is shared  # noqa: SLF001
    assert judge_v2._BASE_LOCK is shared  # noqa: SLF001
    before = (judge.base.ANSWER_RUN_FORMAT, judge.base.PREFLIGHT_NAME, judge.base.build_runtime)
    with judge_v4._v4_base_contract():  # noqa: SLF001
        assert judge.base.ANSWER_RUN_FORMAT == judge_v4.ANSWER_RUN_FORMAT
        with judge._v5_base_contract():  # noqa: SLF001
            assert judge.base.ANSWER_RUN_FORMAT == judge.ANSWER_RUN_FORMAT
            assert judge.base.PREFLIGHT_NAME == judge.PREFLIGHT_NAME
        assert judge.base.ANSWER_RUN_FORMAT == judge_v4.ANSWER_RUN_FORMAT
    assert (judge.base.ANSWER_RUN_FORMAT, judge.base.PREFLIGHT_NAME, judge.base.build_runtime) == before


def test_cross_version_contracts_cannot_interleave() -> None:
    v4_entered = Event()
    release_v4 = Event()
    v5_entered = Event()
    failures: list[BaseException] = []

    def hold_v4() -> None:
        try:
            with judge_v4._v4_base_contract():  # noqa: SLF001
                v4_entered.set()
                assert release_v4.wait(5)
                assert judge.base.ANSWER_RUN_FORMAT == judge_v4.ANSWER_RUN_FORMAT
        except BaseException as error:  # pragma: no cover
            failures.append(error)

    def enter_v5() -> None:
        try:
            assert v4_entered.wait(5)
            with judge._v5_base_contract():  # noqa: SLF001
                assert judge.base.ANSWER_RUN_FORMAT == judge.ANSWER_RUN_FORMAT
                v5_entered.set()
        except BaseException as error:  # pragma: no cover
            failures.append(error)

    old_thread = Thread(target=hold_v4)
    new_thread = Thread(target=enter_v5)
    old_thread.start()
    assert v4_entered.wait(5)
    new_thread.start()
    assert not v5_entered.wait(0.2)
    release_v4.set()
    old_thread.join(5)
    new_thread.join(5)
    assert not old_thread.is_alive() and not new_thread.is_alive()
    assert not failures
    assert v5_entered.is_set()
