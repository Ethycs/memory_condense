from __future__ import annotations

import argparse
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import FastEvidence
from tools import run_routed_full_source_repair as runner
from tools._locked_em_repair_adapter import (
    LockedBaselinePrediction,
    LockedEMQuestionView,
    LockedEMRepairPopulation,
    LockedEMRepairRow,
    LockedEMStageView,
)


def _stage(stage_id: str, evidence: tuple[FastEvidence, ...], seed: str):
    return LockedEMStageView(
        stage_id=stage_id,
        stage_receipt_sha256=seed * 64,
        evidence_projection_sha256=seed.swapcase().lower() * 64,
        evidence=evidence,
    )


def _question(
    ordinal: int,
    question_id: str,
    dated_question: str,
    root: FastEvidence,
    addition: FastEvidence,
    baseline: str,
) -> LockedEMRepairRow:
    s0 = _stage("causal_graph_coverage_predecessor", (root,), "a")
    s1 = _stage("direct_episode_additions", (root, addition), "b")
    view = LockedEMQuestionView(
        ordinal=ordinal,
        question_id=question_id,
        question_sha256=quote_sha256(dated_question.split("\n", 1)[1]),
        dated_question_sha256=quote_sha256(dated_question),
        retrieval_question_part_sha256=("c" if ordinal == 0 else "d") * 64,
        dated_question=dated_question,
        stages=(s0, s1),
    )
    prediction = LockedBaselinePrediction(
        text=baseline,
        text_sha256=quote_sha256(baseline),
        final_answer_row_sha256=("e" if ordinal == 0 else "f") * 64,
    )
    return LockedEMRepairRow(
        question=view,
        baseline=prediction,
        binding_sha256=("1" if ordinal == 0 else "2") * 64,
    )


def _population() -> LockedEMRepairPopulation:
    numeric = _question(
        0,
        "q-numeric",
        "[Question asked at 2026/08/26]\nHow many widgets are there in total?",
        FastEvidence(
            evidence_id="root-n", source_id="s-root-n", text="There were 3 widgets."
        ),
        FastEvidence(
            evidence_id="add-n", source_id="s-add-n", text="Then 2 widgets were added."
        ),
        "BASELINE-NUMERIC",
    )
    direct = _question(
        1,
        "q-direct",
        "[Question asked at 2026/08/26]\nWhat color was the box?",
        FastEvidence(
            evidence_id="root-d", source_id="s-root-d", text="The box was blue."
        ),
        FastEvidence(
            evidence_id="add-d", source_id="s-add-d", text="The lid was square."
        ),
        "blue",
    )
    return LockedEMRepairPopulation(
        retrieval_sha256="3" * 64,
        baseline_final_answers_sha256="4" * 64,
        population_identity_sha256="5" * 64,
        rows=(numeric, direct),
        binding_sha256="6" * 64,
    )


class _Completions:
    def __init__(self, *, unsupported: bool = False) -> None:
        self.unsupported = unsupported
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def create(self, **request: Any) -> Any:
        with self.lock:
            self.requests.append(dict(request))
        system = str(request["messages"][0]["content"])
        if str(request["model"]).endswith("gpt-5.6-sol"):
            completion = "CORRECT - the candidate matches the reference."
        elif "Convert a retrieved episodic-memory neighborhood" in system:
            fact = "999 widgets were added." if self.unsupported else "2 widgets were added."
            completion = (
                '{"facts":[{"text":"'
                + fact
                + '","citations":[{"evidence_alias":"E001",'
                '"quote":"2 widgets were added"}]}]}'
            )
        else:
            completion = "5"
        return SimpleNamespace(
            id="fake-response",
            model="fake-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=completion),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=0, completion_tokens=0, total_tokens=0
            ),
        )


class _Client:
    def __init__(self, *, unsupported: bool = False) -> None:
        self.max_retries = 0
        self.completions = _Completions(unsupported=unsupported)
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _args(root: Path, phase: str, *, provider: bool = False, calls: int = 0):
    return argparse.Namespace(
        phase=phase,
        style="numeric_reduce",
        retrieval=root / "unused-retrieval.json",
        expected_retrieval_sha256="3" * 64,
        baseline_answers=root / "unused-answers.json",
        expected_baseline_answers_sha256="4" * 64,
        baseline_judge=root / "baseline-judge.json",
        expected_baseline_judge_sha256="0" * 64,
        dataset=root / "unused-dataset.json",
        split=root / "unused-split.json",
        output_root=root,
        run_artifact=None,
        run_replay=None,
        judge_artifact=None,
        judge_replay=None,
        expected_question_count=2,
        gateway_url=runner.DEFAULT_GATEWAY_URL,
        terra_model=runner.DEFAULT_TERRA_MODEL,
        sol_model=runner.DEFAULT_SOL_MODEL,
        api_key_env="TEST_ROUTED_KEY",
        max_concurrency=2,
        enable_provider=provider,
        authorized_provider_calls=calls,
    )


def _install_population(monkeypatch: pytest.MonkeyPatch) -> LockedEMRepairPopulation:
    population = _population()
    monkeypatch.setattr(runner, "_load_population", lambda _args: population)
    return population


def _baseline_judge(
    root: Path, args: argparse.Namespace, population: LockedEMRepairPopulation
) -> None:
    payload = {
        "format": "memory-condense-fixed-stage-final-answer-semantic-judge-score-v1",
        "retrieval_sha256": population.retrieval_sha256,
        "final_answer_artifact_sha256": population.baseline_final_answers_sha256,
        "population_identity_sha256": population.population_identity_sha256,
        "question_count": 2,
        "questions": [
            {
                "ordinal": ordinal,
                "question_id": row.question.question_id,
                "prediction_sha256": row.baseline.text_sha256,
                "correct": ordinal == 1,
                "response_journal_sha256": str(ordinal + 7) * 64,
            }
            for ordinal, row in enumerate(population.rows)
        ],
    }
    digest = runner._publish(root / "baseline-judge.json", payload)
    args.expected_baseline_judge_sha256 = digest


def _gold() -> tuple[Any, ...]:
    return (
        SimpleNamespace(
            question_id="q-numeric",
            question="How many widgets are there in total?",
            dated_question=(
                "[Question asked at 2026/08/26]\n"
                "How many widgets are there in total?"
            ),
            answer="5",
        ),
        SimpleNamespace(
            question_id="q-direct",
            question="What color was the box?",
            dated_question=(
                "[Question asked at 2026/08/26]\nWhat color was the box?"
            ),
            answer="blue",
        ),
    )


def test_staged_run_has_exact_budgets_and_byte_identical_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_population(monkeypatch)
    args = _args(tmp_path, "preflight")
    preflight = runner.run_preflight(args)
    assert preflight["eligible_question_count"] == 1
    assert preflight["required_authorized_provider_calls"] == 1
    assert preflight["authorized_call_kind"] == "terra_compression"
    assert preflight["method_budget"]["shared_unbounded_tail_attached"] is False

    made_client = False

    def forbidden_client(*_args: Any) -> Any:
        nonlocal made_client
        made_client = True
        raise AssertionError("authorization must fail before client creation")

    monkeypatch.setattr(runner, "_make_provider_client", forbidden_client)
    with pytest.raises(ValueError, match="compression population"):
        runner.run_compression(_args(tmp_path, "compression-run", provider=True))
    assert made_client is False

    client = _Client()
    monkeypatch.setenv("TEST_ROUTED_KEY", "test-secret")
    monkeypatch.setattr(runner, "_make_provider_client", lambda *_args: client)
    compression, compression_sha = runner.run_compression(
        _args(tmp_path, "compression-run", provider=True, calls=1)
    )
    assert compression["status_counts"] == {"valid": 1}
    replay, replay_sha = runner.run_compression_replay(
        _args(tmp_path, "compression-replay")
    )
    assert replay == compression
    assert replay_sha == compression_sha

    answer_preflight = runner.run_answer_preflight(
        _args(tmp_path, "answer-preflight")
    )
    assert answer_preflight["required_authorized_provider_calls"] == 1
    client = _Client()
    monkeypatch.setattr(runner, "_make_provider_client", lambda *_args: client)
    run, run_sha = runner.run_treatment(
        _args(tmp_path, "run", provider=True, calls=1)
    )
    assert [row["prediction"] for row in run["questions"]] == ["5", "blue"]
    assert run["questions"][1]["prediction_kind"] == "sealed_baseline_preserved"
    assert run["budget"]["shared_unbounded_tail_attached"] is False
    assert run["budget"]["eligible_rows"][0]["validated_fact_count"] == 1
    assert run["budget"]["eligible_rows"][0]["answer_prompt_token_proxy"] > 0
    replay, replay_sha = runner.run_replay(_args(tmp_path, "replay"))
    assert replay == run
    assert replay_sha == run_sha


def test_unsupported_numeric_fact_preserves_baseline_without_answer_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_population(monkeypatch)
    monkeypatch.setenv("TEST_ROUTED_KEY", "test-secret")
    monkeypatch.setattr(
        runner, "_make_provider_client", lambda *_args: _Client(unsupported=True)
    )
    compression, _ = runner.run_compression(
        _args(tmp_path, "compression-run", provider=True, calls=1)
    )
    assert compression["status_counts"] == {"unsupported_numeric_literal": 1}
    preflight = runner.run_answer_preflight(_args(tmp_path, "answer-preflight"))
    assert preflight["required_authorized_provider_calls"] == 0
    assert preflight["baseline_fallback_count"] == 1

    def no_answer_client(*_args: Any) -> Any:
        raise AssertionError("baseline fallback must not create an answer client")

    monkeypatch.setattr(runner, "_make_provider_client", no_answer_client)
    run, _ = runner.run_treatment(_args(tmp_path, "run", provider=True, calls=0))
    row = run["questions"][0]
    assert row["prediction"] == "BASELINE-NUMERIC"
    assert row["prediction_kind"] == "sealed_baseline_fallback"
    assert row["baseline_fallback_reason"] == (
        "unsupported_numeric_literal_compression"
    )
    assert run["answer_completion_batch"] is None


def test_score_and_sol_judge_only_changed_eligible_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    population = _install_population(monkeypatch)
    monkeypatch.setenv("TEST_ROUTED_KEY", "test-secret")
    monkeypatch.setattr(runner, "_make_provider_client", lambda *_args: _Client())
    runner.run_compression(_args(tmp_path, "compression-run", provider=True, calls=1))
    runner.run_treatment(_args(tmp_path, "run", provider=True, calls=1))
    monkeypatch.setattr(
        runner,
        "_load_gold_population",
        lambda _dataset, _split, _population: _gold(),
    )
    args = _args(tmp_path, "score")
    score, _ = runner.run_score(args)
    assert score["aggregate"]["candidate_exact_matches"] == 2
    assert score["aggregate"]["eligible_exact_rescues"] == 1

    _baseline_judge(tmp_path, args, population)
    preflight = runner.run_judge_preflight(args)
    assert preflight["changed_eligible_prediction_count"] == 1
    assert preflight["required_authorized_provider_calls"] == 1
    client = _Client()
    monkeypatch.setattr(runner, "_make_provider_client", lambda *_args: client)
    args.enable_provider = True
    args.authorized_provider_calls = 1
    judge, judge_sha = runner.run_judge(args)
    assert len(client.completions.requests) == 1
    assert judge["aggregate"] == {
        "baseline_correct": 1,
        "candidate_correct": 2,
        "eligible_rescued": 1,
        "eligible_regressed": 0,
        "eligible_net_marginal": 1,
    }
    assert judge["questions"][1]["verdict_source"] == "sealed_baseline_judge"
    args.enable_provider = False
    args.authorized_provider_calls = 0
    replay, replay_sha = runner.run_judge_replay(args)
    assert replay == judge
    assert replay_sha == judge_sha
