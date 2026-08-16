"""Tests for the QA-probe benchmark eval path.

All fixtures are inline; nothing is downloaded and no network call is made.
The default path uses a fake retriever so the tests stay fast — the one test
that exercises a real MemoryCondenser (and therefore bge-m3) is marked slow.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.eval.benchmark import (
    BenchmarkRunResult,
    build_qa_prompt,
    cap_context_to_prompt_budget,
    exact_match,
    f1_score,
    ingest_sample,
    normalize_answer,
    run_benchmark,
    save_benchmark_report,
)
from memory_condense.eval.schemas import (
    ChunkerConfig,
    EvalConfig,
    RetrievalConfig,
    UsageStats,
)
from memory_condense.loader import BenchmarkQuestion, BenchmarkSample, load_benchmark

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeCondenser:
    """In-memory stand-in for MemoryCondenser: no embeddings, no downloads.

    Retrieval is naive token overlap, which is enough to assert that the
    right turns reach the QA prompt.
    """

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.closed = False

    def ingest(self, role: str, text: str) -> None:
        self.texts.append(text)

    def search(self, query: str, k: int = 10, ef_search: int = 50):
        query_tokens = set(normalize_answer(query).split())
        scored = [
            (len(query_tokens & set(normalize_answer(t).split())), t)
            for t in self.texts
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            SimpleNamespace(chunk=SimpleNamespace(text=text), score=float(score))
            for score, text in scored[:k]
        ]

    def search_sources(self, query: str, k_sources: int = 4):
        return self.search(query, k=max(k_sources, 1))

    def search_anchored_sources(self, query: str, k: int = 10, **kwargs):
        return self.search(query, k=k)

    def search_hybrid_sources(self, query: str, k: int = 10, **kwargs):
        return self.search(query, k=k)

    def search_hybrid_graph(self, query: str, k: int = 10, **kwargs):
        return self.search(query, k=k)

    def search_hybrid_neighbors(self, query: str, k: int = 10, **kwargs):
        return self.search(query, k=k)

    def close(self) -> None:
        self.closed = True


def fake_ingest_fn(sample: BenchmarkSample, config: EvalConfig, data_dir: Path):
    mc = FakeCondenser()
    for role, text in sample.turns:
        mc.ingest(role, text)
    return mc


def make_config(k: int = 3) -> EvalConfig:
    return EvalConfig(
        chunker=ChunkerConfig(min_tokens=20, max_tokens=60),
        retrieval=RetrievalConfig(k=k, ef_search=20),
    )


SAMPLE = BenchmarkSample(
    sample_id="s1",
    turns=[
        ("user", "I moved to Boston last spring."),
        ("assistant", "That is a big change!"),
        ("user", "My cat is named Pepper."),
        ("assistant", "Pepper is a great name."),
    ],
    questions=[
        BenchmarkQuestion(
            question_id="s1_q0",
            question="Which city did I move to?",
            answer="Boston",
            category="single-session",
        ),
        BenchmarkQuestion(
            question_id="s1_q1",
            question="What is my cat's name?",
            answer="Pepper",
            category="multi-session",
        ),
    ],
)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def test_f1_exact_match_is_one():
    assert f1_score("Boston", "Boston") == 1.0


def test_f1_disjoint_is_zero():
    assert f1_score("Boston", "Chicago") == 0.0


def test_f1_partial_overlap_is_between():
    score = f1_score("the red bicycle", "red bicycle in the garage")
    assert 0.0 < score < 1.0


def test_f1_normalizes_case_punctuation_and_articles():
    assert f1_score("The Answer.", "answer") == 1.0
    assert f1_score("  a  BOSTON!! ", "boston") == 1.0


def test_f1_empty_handling():
    assert f1_score("", "") == 1.0
    assert f1_score("", "Boston") == 0.0
    assert f1_score("Boston", "") == 0.0


def test_normalize_answer():
    assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"


def test_exact_match_uses_normalization():
    assert exact_match("The Answer.", "answer") is True
    assert exact_match("answer two", "answer") is False


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


def test_build_qa_prompt_includes_context_and_question():
    messages = build_qa_prompt("Where?", ["chunk one", "chunk two"])
    assert messages[0]["role"] == "system"
    assert "chunk one" in messages[1]["content"]
    assert "chunk two" in messages[1]["content"]
    assert "Where?" in messages[1]["content"]


def test_build_qa_prompt_without_chunks():
    messages = build_qa_prompt("Where?", [])
    assert "no excerpts" in messages[1]["content"]


def test_dated_question_reaches_retrieval_and_prompt():
    dated = SAMPLE.questions[0].model_copy(
        update={"question_date": "2026/08/16 10:00"}
    )
    sample = SAMPLE.model_copy(update={"questions": [dated]})
    seen: list[str] = []

    run_benchmark(
        samples=[sample],
        config=make_config(k=1),
        answer_fn=lambda messages: seen.append(messages[1]["content"]) or "Boston",
        ingest_fn=fake_ingest_fn,
    )

    assert "Question asked at 2026/08/16" in seen[0]


def test_source_mode_reaches_the_benchmark_prompt():
    config = make_config(k=1).model_copy(
        update={"retrieval": RetrievalConfig(mode="source", k_sources=1)}
    )
    seen: list[str] = []
    run_benchmark(
        samples=[SAMPLE],
        config=config,
        answer_fn=lambda messages: seen.append(messages[1]["content"]) or "x",
        ingest_fn=fake_ingest_fn,
    )
    assert len(seen) == 2


def test_prompt_budget_caps_and_truncates_context():
    from memory_condense._tokenizer import count_tokens

    uncapped = ["Boston " * 200, "Pepper " * 200]
    capped = cap_context_to_prompt_budget("Where?", uncapped, 120)
    prompt = build_qa_prompt("Where?", capped)

    assert capped
    assert len(capped) == 1
    assert sum(count_tokens(message["content"]) for message in prompt) <= 120


# ---------------------------------------------------------------------------
# run_benchmark (fake retriever)
# ---------------------------------------------------------------------------


def test_run_benchmark_end_to_end_with_stub_answer_fn():
    def answer_fn(messages: list[dict[str, str]]) -> str:
        return "Boston"

    result = run_benchmark(
        samples=[SAMPLE],
        config=make_config(),
        answer_fn=answer_fn,
        ingest_fn=fake_ingest_fn,
        benchmark="fixture",
    )

    assert isinstance(result, BenchmarkRunResult)
    assert result.num_samples == 1
    assert result.num_questions == 2
    # One question matches ("Boston"), one does not ("Pepper").
    assert result.mean_f1 == pytest.approx(0.5)
    assert result.exact_match_rate == pytest.approx(0.5)
    assert result.judge_accuracy is None
    assert result.target_status == "ungraded"
    assert result.accuracy_target_met is None
    assert result.mean_prompt_tokens > result.mean_context_tokens
    assert result.mean_transcript_tokens > result.mean_context_tokens
    assert 0.0 < result.mean_context_fraction < 1.0
    assert result.mean_transcript_token_savings == pytest.approx(
        1.0 - result.mean_context_fraction
    )
    assert result.prompt_budget_compliance is True
    assert result.max_prompt_tokens_observed > 0
    assert result.benchmark == "fixture"
    assert result.run_timestamp


def test_run_benchmark_category_breakdown():
    def answer_fn(messages: list[dict[str, str]]) -> str:
        return "Boston"

    result = run_benchmark(
        samples=[SAMPLE],
        config=make_config(),
        answer_fn=answer_fn,
        ingest_fn=fake_ingest_fn,
    )

    assert set(result.by_category) == {"single-session", "multi-session"}
    assert result.by_category["single-session"].mean_f1 == pytest.approx(1.0)
    assert result.by_category["multi-session"].mean_f1 == pytest.approx(0.0)
    assert result.by_category["single-session"].num_questions == 1


def test_run_benchmark_uncategorized_bucket():
    sample = BenchmarkSample(
        sample_id="s2",
        turns=[("user", "hello")],
        questions=[
            BenchmarkQuestion(question_id="q", question="hi?", answer="hello")
        ],
    )
    result = run_benchmark(
        samples=[sample],
        config=make_config(),
        answer_fn=lambda messages: "hello",
        ingest_fn=fake_ingest_fn,
    )
    assert "uncategorized" in result.by_category


def test_run_benchmark_with_judge_fn():
    def answer_fn(messages: list[dict[str, str]]) -> str:
        return "the city of Boston"

    def judge_fn(question: str, gold: str, prediction: str) -> tuple[bool, str]:
        return (gold.lower() in prediction.lower(), "substring check")

    result = run_benchmark(
        samples=[SAMPLE],
        config=make_config(),
        answer_fn=answer_fn,
        judge_fn=judge_fn,
        ingest_fn=fake_ingest_fn,
    )

    assert result.judge_accuracy == pytest.approx(0.5)
    assert result.exact_match_rate == pytest.approx(0.0)  # verbose answer
    qr = result.samples[0].question_results[0]
    assert qr.judge_correct is True
    assert qr.judge_reasoning == "substring check"
    assert result.by_category["multi-session"].judge_accuracy == pytest.approx(0.0)
    assert result.target_status == "insufficient_questions"
    assert result.accuracy_target_met is False


def test_accuracy_target_requires_enough_questions_and_passes():
    sample = SAMPLE.model_copy(update={"questions": [SAMPLE.questions[0]] * 3})
    config = make_config().model_copy(
        update={"accuracy_target": 0.95, "min_target_questions": 3}
    )
    result = run_benchmark(
        samples=[sample],
        config=config,
        answer_fn=lambda messages: "Boston",
        judge_fn=lambda question, gold, prediction: (True, "correct"),
        ingest_fn=fake_ingest_fn,
    )

    assert result.judge_accuracy == 1.0
    assert result.accuracy_target_met is True
    assert result.target_status == "passed"
    assert result.p95_prompt_tokens >= result.mean_prompt_tokens
    assert result.prompt_budget_compliance is True


def test_accuracy_target_fails_when_operational_prompt_budget_is_exceeded(
    monkeypatch,
):
    import memory_condense.eval.benchmark as benchmark_module

    sample = SAMPLE.model_copy(update={"questions": [SAMPLE.questions[0]]})
    config = make_config().model_copy(
        update={
            "accuracy_target": 0.95,
            "min_target_questions": 1,
            "max_prompt_tokens": 10,
        }
    )
    monkeypatch.setattr(
        benchmark_module,
        "answer_question",
        lambda mc, question, config, answer_fn: (
            "Boston",
            [],
            config.max_prompt_tokens + 1,
            UsageStats(),
        ),
    )

    result = run_benchmark(
        samples=[sample],
        config=config,
        answer_fn=lambda messages: "Boston",
        judge_fn=lambda question, gold, prediction: (True, "correct"),
        ingest_fn=fake_ingest_fn,
    )

    assert result.prompt_budget_compliance is False
    assert result.accuracy_target_met is False
    assert result.target_status == "prompt_budget_exceeded"


def test_run_benchmark_max_samples():
    samples = [
        SAMPLE.model_copy(update={"sample_id": f"s{i}"}) for i in range(5)
    ]
    result = run_benchmark(
        samples=samples,
        config=make_config(),
        answer_fn=lambda messages: "Boston",
        ingest_fn=fake_ingest_fn,
        max_samples=2,
    )
    assert result.num_samples == 2


def test_run_benchmark_retrieval_reaches_the_prompt():
    seen: list[str] = []

    def answer_fn(messages: list[dict[str, str]]) -> str:
        seen.append(messages[1]["content"])
        return "x"

    run_benchmark(
        samples=[SAMPLE],
        config=make_config(k=1),
        answer_fn=answer_fn,
        ingest_fn=fake_ingest_fn,
    )

    assert len(seen) == 2
    assert "Boston" in seen[0]
    assert "Pepper" in seen[1]


def test_run_benchmark_closes_the_store():
    created: list[FakeCondenser] = []

    def tracking_ingest(sample, config, data_dir):
        mc = fake_ingest_fn(sample, config, data_dir)
        created.append(mc)
        return mc

    run_benchmark(
        samples=[SAMPLE],
        config=make_config(),
        answer_fn=lambda messages: "Boston",
        ingest_fn=tracking_ingest,
    )

    assert created and all(mc.closed for mc in created)


def test_run_benchmark_aggregates_metered_answer_and_judge_usage():
    answer_usage = UsageStats(input_tokens=20, output_tokens=2, calls=1)
    judge_usage = UsageStats(input_tokens=5, output_tokens=3, calls=1)

    result = run_benchmark(
        samples=[SAMPLE],
        config=make_config(),
        answer_fn=lambda messages: ("Boston", answer_usage),
        judge_fn=lambda question, gold, prediction: (
            True,
            "CORRECT",
            judge_usage,
        ),
        ingest_fn=fake_ingest_fn,
    )

    assert result.responder_usage == UsageStats(
        input_tokens=40,
        output_tokens=4,
        calls=2,
    )
    assert result.judge_usage == UsageStats(
        input_tokens=10,
        output_tokens=6,
        calls=2,
    )


def test_run_benchmark_empty_samples():
    result = run_benchmark(
        samples=[],
        config=make_config(),
        answer_fn=lambda messages: "x",
        ingest_fn=fake_ingest_fn,
    )
    assert result.num_samples == 0
    assert result.mean_f1 == 0.0
    assert result.by_category == {}


def test_save_benchmark_report(tmp_path: Path):
    result = run_benchmark(
        samples=[SAMPLE],
        config=make_config(),
        answer_fn=lambda messages: "Boston",
        ingest_fn=fake_ingest_fn,
        benchmark="longmemeval_s",
    )
    path = save_benchmark_report(result, tmp_path / "out")
    assert path.exists()
    assert path.name.startswith("benchmark_longmemeval_s_")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["num_questions"] == 2
    assert "by_category" in payload
    checksum = path.parent / f"{path.name}.sha256"
    assert checksum.exists()
    assert checksum.read_text(encoding="ascii").strip().endswith(path.name)


# ---------------------------------------------------------------------------
# Loader -> benchmark integration (still offline)
# ---------------------------------------------------------------------------


def test_loaded_longmemeval_file_runs_through_the_benchmark(tmp_path: Path):
    record = {
        "question_id": "lme_x",
        "question_type": "temporal-reasoning",
        "question": "Which city did I move to?",
        "answer": "Boston",
        "haystack_sessions": [
            [
                {"role": "user", "content": "I moved to Boston last spring."},
                {"role": "assistant", "content": "Nice!"},
            ]
        ],
    }
    f = tmp_path / "lme.json"
    f.write_text(json.dumps([record]), encoding="utf-8")

    samples = load_benchmark(f)
    result = run_benchmark(
        samples=samples,
        config=make_config(),
        answer_fn=lambda messages: "Boston",
        ingest_fn=fake_ingest_fn,
        benchmark="longmemeval",
    )
    assert result.mean_f1 == pytest.approx(1.0)
    assert "temporal-reasoning" in result.by_category


# ---------------------------------------------------------------------------
# Real MemoryCondenser (downloads bge-m3)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ingest_sample_with_real_condenser(tmp_path: Path):
    config = make_config()
    mc = ingest_sample(SAMPLE, config, tmp_path / "store")
    try:
        results = mc.search("Which city did I move to?", k=2)
        assert results
        assert any("Boston" in r.chunk.text for r in results)
    finally:
        mc.close()


@pytest.mark.slow
def test_run_benchmark_with_real_condenser(tmp_path: Path):
    result = run_benchmark(
        samples=[SAMPLE],
        config=make_config(),
        answer_fn=lambda messages: "Boston",
        benchmark="real",
    )
    assert result.num_questions == 2
    assert result.by_category["single-session"].mean_f1 == pytest.approx(1.0)
