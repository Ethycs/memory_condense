from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest

from memory_condense.eval.__main__ import (
    _make_answer_fn,
    _make_judge_fn,
    _planned_provider_calls,
    _verified_policy_sha256,
    build_parser,
)
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from memory_condense.loader import BenchmarkQuestion, BenchmarkSample


def _sample(question_count: int) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=f"sample-{question_count}",
        turns=[("user", "history")],
        questions=[
            BenchmarkQuestion(
                question_id=f"q-{index}",
                question="question",
                answer="answer",
            )
            for index in range(question_count)
        ],
    )


def test_remote_benchmark_defaults_refuse_calls_and_retries():
    args = build_parser().parse_args(["--benchmark-file", "sample.json"])

    assert args.max_provider_calls == 0
    assert args.provider_retries == 0


def test_planned_provider_calls_respects_sample_limit_and_local_answerer():
    samples = [_sample(2), _sample(3)]

    assert _planned_provider_calls(
        samples,
        max_samples=1,
        local_answerer=False,
        use_judge=True,
    ) == 4
    assert _planned_provider_calls(
        samples,
        max_samples=None,
        local_answerer=True,
        use_judge=True,
    ) == 5


def test_metered_provider_wrappers_disable_retries(monkeypatch):
    calls = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="CORRECT"))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
    )

    def completion(**kwargs):
        calls.append(kwargs)
        return response

    monkeypatch.setattr(litellm, "completion", completion)
    answer, answer_usage = _make_answer_fn("anthropic/test")([
        {"role": "user", "content": "question"}
    ])
    correct, reasoning, judge_usage = _make_judge_fn("anthropic/judge")(
        "question", "gold", "prediction"
    )

    assert answer == "CORRECT"
    assert correct is True
    assert reasoning == "CORRECT"
    assert answer_usage.calls == 1
    assert judge_usage.calls == 1
    assert [call["num_retries"] for call in calls] == [0, 0]


def test_policy_manifest_must_match_active_retrieval_config(tmp_path: Path):
    config = EvalConfig(
        retrieval=RetrievalConfig(
            mode="hybrid_neighbor",
            k=10,
            neighbor_radius=6,
            neighbor_slots=23,
        ),
        max_prompt_tokens=8000,
    )
    retrieval = {
        "mode": "hybrid_neighbor",
        "k": 10,
        "ef_search": 50,
        "alpha": 0.65,
        "candidates": 100,
        "neighbor_radius": 6,
        "neighbor_slots": 23,
        "neighbor_replacement_slots": 0,
        "max_prompt_tokens": 8000,
        "chunker_min_tokens": 120,
        "chunker_max_tokens": 250,
    }
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "status": "development_candidate_not_validated",
                "dataset_sha256": "a" * 64,
                "split_manifest": "split.json",
                "retrieval": retrieval,
            }
        ),
        encoding="utf-8",
    )

    assert len(
        _verified_policy_sha256(
            path,
            config=config,
            dataset_sha256="a" * 64,
            split_manifest=str(tmp_path / "split.json"),
        )
    ) == 64

    path.write_text(
        json.dumps(
            {
                "status": "superseded_after_fix",
                "dataset_sha256": "a" * 64,
                "split_manifest": "split.json",
                "retrieval": retrieval,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not active"):
        _verified_policy_sha256(
            path,
            config=config,
            dataset_sha256="a" * 64,
            split_manifest=str(tmp_path / "split.json"),
        )
