from __future__ import annotations

from pathlib import Path

import pytest

from memory_condense.domain.discourse import quote_sha256
from tests.test_matched_eval_closure_judging import _VerdictClient
from tools.matched_eval.artifacts import SealedArtifact
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.typed_memory_final_arm import judge_row_projection
from tools.matched_eval.typed_memory_final_judging import (
    TypedFinalJudgeGoldRow,
    build_runtime,
    materialization_projection,
    preflight_projection,
    validate_preflight_artifact,
    validate_typed_final_run_artifact,
)
from tools import run_locked_typed_memory_final_judge as judge_cli


def _sha(value: str) -> str:
    return quote_sha256(value)


def _source() -> tuple[SealedArtifact, tuple[dict, ...], tuple[TypedFinalJudgeGoldRow, ...]]:
    questions = []
    judge_rows = []
    gold = []
    for ordinal in range(100):
        question_id = f"typed-question-{ordinal:03d}"
        question = f"What is memory value {ordinal}?"
        dated = f"[Question asked at 2023/05/30 16:15]\n{question}"
        changed = ordinal < 2
        prediction = f"answer {ordinal}" if changed else f"parent {ordinal}"
        body = {
            "changed_from_parent": changed,
            "dated_question_sha256": _sha(dated),
            "decision": "replace" if changed else "keep_parent",
            "format": "synthetic-typed-final-result-row",
            "ordinal": ordinal,
            "parent_prediction_sha256": _sha(f"parent {ordinal}"),
            "prediction": prediction,
            "prediction_sha256": _sha(prediction),
            "prediction_source": (
                "typed_final_model_attested_replacement_v1"
                if changed
                else "typed_final_validated_keep_parent_v1"
            ),
            "question_id": question_id,
            "question_sha256": _sha(question),
            "route_id": "extract",
        }
        body["source_row_sha256"] = identity_sha256(body)
        questions.append(body)
        judge_rows.append(judge_row_projection(body))
        reference = f"answer {ordinal}"
        gold.append(
            TypedFinalJudgeGoldRow(
                ordinal,
                question_id,
                question,
                _sha(question),
                dated,
                _sha(dated),
                reference,
                _sha(reference),
                "synthetic",
            )
        )
    payload = {
        "format": "memory-condense-locked-typed-memory-final-arm-v1-run-v1",
        "gold_loaded": False,
        "judge_rows": judge_rows,
        "physical_provider_calls_during_materialization": 0,
        "question_count": 100,
        "questions": questions,
        "retained_transformer_token_state_bytes": 0,
    }
    artifact = SealedArtifact(Path("synthetic-run.json"), identity_sha256(payload), payload)
    return artifact, tuple(judge_rows), tuple(gold)


def test_full100_and_changed_only_use_exact_exported_prediction_identities() -> None:
    run, source_rows, gold = _source()
    assert validate_typed_final_run_artifact(run) == source_rows
    for mode, expected in (("full100", 100), ("changed_only", 2)):
        payload, prompts = preflight_projection(
            run_artifact=run,
            replay_artifact_sha256=_sha("replay"),
            source_rows=source_rows,
            gold_rows=gold,
            gold_population_sha256=_sha("gold"),
            mode=mode,
            model="sol-model",
            gateway_url="http://sealed-gateway",
            max_concurrency=2,
        )
        assert len(prompts) == expected
        assert payload["required_authorized_provider_calls"] == expected
        assert payload["typed_final_run_sha256"] == run.sha256
        assert all(
            row["source_row_sha256"]
            == source_rows[row["ordinal"]]["source_row_sha256"]
            and row["prediction_source"]
            == source_rows[row["ordinal"]]["prediction_source"]
            for row in payload["prompt_rows"]
        )


def test_selected_subset_preserves_original_ordinals_and_dynamic_call_count() -> None:
    run, source_rows, gold = _source()
    ordinals = (6, 17, 97)
    selected_source = tuple(source_rows[ordinal] for ordinal in ordinals)
    selected_gold = tuple(gold[ordinal] for ordinal in ordinals)
    payload, prompts = preflight_projection(
        run_artifact=run,
        replay_artifact_sha256=_sha("subset-replay"),
        source_rows=selected_source,
        gold_rows=selected_gold,
        gold_population_sha256=_sha("subset-gold"),
        mode="selected_subset",
        model="sol-model",
        gateway_url="http://sealed-gateway",
        max_concurrency=2,
    )

    assert len(prompts) == 3
    assert payload["question_count"] == 100
    assert payload["selected_question_count"] == 3
    assert payload["required_authorized_provider_calls"] == 3
    assert tuple(row["ordinal"] for row in payload["prompt_rows"]) == ordinals
    artifact = SealedArtifact(
        Path("subset-preflight.json"), identity_sha256(payload), payload
    )
    validated_prompts, validated_rows = validate_preflight_artifact(artifact)
    assert validated_prompts == prompts
    assert tuple(row["ordinal"] for row in validated_rows) == ordinals


def test_changed_only_checkpoint_materialization_and_replay_are_stable(
    tmp_path: Path,
) -> None:
    run, source_rows, gold = _source()
    payload, _ = preflight_projection(
        run_artifact=run,
        replay_artifact_sha256=_sha("replay"),
        source_rows=source_rows,
        gold_rows=gold,
        gold_population_sha256=_sha("gold"),
        mode="changed_only",
        model="sol-model",
        gateway_url="http://sealed-gateway",
        max_concurrency=2,
    )
    artifact = SealedArtifact(
        tmp_path / "preflight.json", identity_sha256(payload), payload
    )
    prompts, rows = validate_preflight_artifact(artifact)
    client = _VerdictClient(
        ["CORRECT - exact typed replacement.", "CORRECT - exact typed replacement."]
    )
    runtime = build_runtime(
        artifact,
        prompts,
        output_root=tmp_path,
        model="sol-model",
        gateway_url="http://sealed-gateway",
        max_concurrency=2,
        client=client,
    )
    try:
        provider_batch = runtime.run()
    finally:
        runtime.close()
    assert provider_batch.usage.physical_calls == 2

    replay_runtime = build_runtime(
        artifact,
        prompts,
        output_root=tmp_path,
        model="sol-model",
        gateway_url="http://sealed-gateway",
        max_concurrency=2,
        client=None,
    )
    try:
        checkpoint_batch = replay_runtime.run()
    finally:
        replay_runtime.close()
    judge, score = materialization_projection(artifact, rows, checkpoint_batch)
    rebuilt_judge, rebuilt_score = materialization_projection(
        artifact, rows, checkpoint_batch
    )
    assert checkpoint_batch.usage.physical_calls == 0
    assert checkpoint_batch.usage.checkpoint_hits == 2
    assert score["correct"] == score["selected_question_count"] == 2
    assert judge == rebuilt_judge
    assert score == rebuilt_score


def test_provider_authorization_fails_before_environment_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        judge_cli,
        "_read_preflight",
        lambda *_args, **_kwargs: (
            SealedArtifact(Path("p"), _sha("preflight"), {"required_authorized_provider_calls": 2}),
            (({"role": "user", "content": "one"},), ({"role": "user", "content": "two"},)),
            ({}, {}),
        ),
    )
    monkeypatch.setattr(
        judge_cli,
        "load_dotenv",
        lambda: pytest.fail("unauthorized provider path accessed environment"),
    )
    args = type(
        "Args",
        (),
        {
            "authorized_provider_calls": 1,
            "enable_provider": True,
            "expected_judge_preflight_sha256": _sha("preflight"),
            "judge_output_root": Path("unused"),
        },
    )()
    with pytest.raises(Exception, match="exact authorization for 2"):
        judge_cli._provider(args)
