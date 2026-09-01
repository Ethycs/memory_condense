from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_fast_completion_runtime import _FakeClient
from tests.test_matched_eval_live_execution import _SolClient, _synthetic_gold_loader
from tests.test_matched_eval_population import _publish, _retrieval
from tools import run_matched_eval_spine as spine
from tools.matched_eval import judging, live
from tools.matched_eval.artifacts import read_sealed_json


def _run_pipeline(
    *,
    retrieval: Path,
    output: Path,
    tmp_path: Path,
    answer_calls: int = 2,
    judge_calls: int = 2,
) -> dict[str, object]:
    return spine.run_s0_v4_pipeline(
        retrieval_path=retrieval,
        dataset_path=tmp_path / "unused-dataset.json",
        split_path=tmp_path / "unused-split.json",
        output_root=output,
        enable_answer_provider=True,
        authorized_answer_provider_calls=answer_calls,
        enable_judge_provider=True,
        authorized_judge_provider_calls=judge_calls,
        api_key_env="MATCHED_TEST_KEY",
        max_concurrency=2,
        selected_ordinals=None,
        expected_retrieval_sha256=None,
        expected_question_count=2,
    )


def test_v4_pipeline_reuses_one_population_and_plan_then_reverifies_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval = _publish(tmp_path, _retrieval(2))
    output = tmp_path / "pipeline"
    terra = _FakeClient(output / live.CHECKPOINT_DIR_NAME)
    sol = _SolClient()
    monkeypatch.setenv("MATCHED_TEST_KEY", "test-key")
    monkeypatch.setattr(live, "_make_provider_client", lambda *_args: terra)
    monkeypatch.setattr(judging, "_make_provider_client", lambda *_args: sol)

    population_loads = 0
    gold_loads = 0
    plan_builds = 0
    events: list[str] = []
    original_population_loader = live.load_s0_population
    original_gold_loader = judging._load_gold
    original_plan_builder = judging._build_plan_from_answer_plane
    original_answer_replay = live._replay_s0_v2_answers_for_population
    original_judge_replay = judging._replay_prebuilt_judge_plan
    original_final_read = spine.read_sealed_json

    def counted_population_loader(*args, **kwargs):
        nonlocal population_loads
        population_loads += 1
        return original_population_loader(*args, **kwargs)

    def counted_gold_loader(**kwargs):
        nonlocal gold_loads
        gold_loads += 1
        events.append("gold_loaded")
        return _synthetic_gold_loader(**kwargs)

    def counted_plan_builder(**kwargs):
        nonlocal plan_builds
        plan_builds += 1
        return original_plan_builder(**kwargs)

    def traced_answer_replay(**kwargs):
        result = original_answer_replay(**kwargs)
        events.append("answer_replayed")
        return result

    def traced_judge_replay(*args, **kwargs):
        result = original_judge_replay(*args, **kwargs)
        events.append("judge_replayed")
        return result

    def traced_final_read(path):
        result = original_final_read(path)
        if Path(path).resolve() == retrieval.resolve():
            events.append("retrieval_reverified")
        return result

    monkeypatch.setattr(live, "load_s0_population", counted_population_loader)
    monkeypatch.setattr(judging, "_load_gold", counted_gold_loader)
    monkeypatch.setattr(
        judging,
        "_build_plan_from_answer_plane",
        counted_plan_builder,
    )
    monkeypatch.setattr(
        live,
        "_replay_s0_v2_answers_for_population",
        traced_answer_replay,
    )
    monkeypatch.setattr(
        judging,
        "_replay_prebuilt_judge_plan",
        traced_judge_replay,
    )
    monkeypatch.setattr(spine, "read_sealed_json", traced_final_read)

    result = _run_pipeline(
        retrieval=retrieval,
        output=output,
        tmp_path=tmp_path,
    )

    assert population_loads == gold_loads == plan_builds == 1
    assert events.index("answer_replayed") < events.index("gold_loaded")
    assert events[-2:] == ["judge_replayed", "retrieval_reverified"]
    assert len(terra.chat.completions.requests) == 2
    assert len(sol.chat.completions.requests) == 2
    assert result["answer_physical_provider_calls"] == 2
    assert result["judge_physical_provider_calls"] == 2
    assert result["total_physical_provider_calls"] == 4
    assert result["correct"] == 2
    assert result["answer_run_sha256"] == result["answer_replay_sha256"]
    assert result["judge_run_sha256"] == result["judge_replay_sha256"]
    assert result["score_ledger_sha256"] == result["score_ledger_replay_sha256"]
    assert result["retrieval_reverified_after_judge_replay"] is True
    assert read_sealed_json(output / live.ANSWER_RUN_NAME).sha256 == result[
        "answer_run_sha256"
    ]
    assert read_sealed_json(output / judging.JUDGE_NAME).sha256 == result[
        "judge_run_sha256"
    ]

    first = dict(result)
    population_loads = gold_loads = plan_builds = 0

    def forbidden_client(*_args):
        raise AssertionError("sealed pipeline attempted another provider client")

    monkeypatch.setattr(live, "_make_provider_client", forbidden_client)
    monkeypatch.setattr(judging, "_make_provider_client", forbidden_client)
    resumed = _run_pipeline(
        retrieval=retrieval,
        output=output,
        tmp_path=tmp_path,
    )
    assert population_loads == gold_loads == plan_builds == 1
    assert resumed["answer_physical_provider_calls"] == 0
    assert resumed["judge_physical_provider_calls"] == 0
    assert resumed["answer_run_sha256"] == first["answer_run_sha256"]
    assert resumed["judge_run_sha256"] == first["judge_run_sha256"]
    assert resumed["score_ledger_sha256"] == first["score_ledger_sha256"]


def test_v4_pipeline_rejects_mismatched_separate_budget_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval = _publish(tmp_path, _retrieval(2))
    output = tmp_path / "pipeline"

    def forbidden_client(*_args):
        raise AssertionError("invalid authorization reached a provider client")

    monkeypatch.setattr(live, "_make_provider_client", forbidden_client)
    monkeypatch.setattr(judging, "_make_provider_client", forbidden_client)
    with pytest.raises(
        ValueError,
        match="authorized judge-provider calls must exactly equal 2",
    ):
        _run_pipeline(
            retrieval=retrieval,
            output=output,
            tmp_path=tmp_path,
            answer_calls=2,
            judge_calls=1,
        )
    assert not output.exists()


def test_v4_pipeline_cli_dispatches_two_exact_authorizations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_pipeline(**kwargs):
        observed.update(kwargs)
        return {"format": "fake-v4-pipeline", "total_physical_provider_calls": 0}

    monkeypatch.setattr(spine, "run_s0_v4_pipeline", fake_pipeline)
    exit_code = spine.main(
        [
            "s0-v4-pipeline",
            "--dataset",
            str(tmp_path / "dataset.json"),
            "--output-root",
            str(tmp_path / "output"),
            "--enable-answer-provider",
            "--authorized-answer-provider-calls",
            "100",
            "--enable-judge-provider",
            "--authorized-judge-provider-calls",
            "100",
        ]
    )
    assert exit_code == 0
    assert observed["enable_answer_provider"] is True
    assert observed["authorized_answer_provider_calls"] == 100
    assert observed["enable_judge_provider"] is True
    assert observed["authorized_judge_provider_calls"] == 100
    assert "format=fake-v4-pipeline" in capsys.readouterr().out
