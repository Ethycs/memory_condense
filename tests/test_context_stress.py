import pytest

from memory_condense._tokenizer import count_tokens
from memory_condense.eval.context_stress import (
    compose_context_stress_sample,
    transcript_tokens,
)
from memory_condense.loader import BenchmarkQuestion, BenchmarkSample


def _sample(sample_id: str, text: str) -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=sample_id,
        turns=[("user", text)],
        turn_source_ids=["shared-source"],
        questions=[
            BenchmarkQuestion(
                question_id=f"q-{sample_id}",
                question="What happened?",
                answer="Something",
                evidence_sources=["shared-source"],
            )
        ],
    )


def test_compose_context_stress_sample_builds_one_namespaced_memory():
    first = _sample("first", "alpha " * 20)
    second = _sample("second", "beta " * 20)
    target = transcript_tokens(first) + 1

    combined = compose_context_stress_sample(
        [first, second], target_tokens=target, max_questions=2
    )

    assert transcript_tokens(combined) >= target
    assert combined.turn_source_ids == [
        "first::shared-source",
        "second::shared-source",
    ]
    assert [q.evidence_sources for q in combined.questions] == [
        ["first::shared-source"],
        ["second::shared-source"],
    ]
    assert count_tokens(combined.turns[0][1]) == transcript_tokens(first)


def test_compose_context_stress_sample_rejects_insufficient_history():
    sample = _sample("only", "short")

    with pytest.raises(ValueError, match="not enough benchmark history"):
        compose_context_stress_sample(
            [sample], target_tokens=transcript_tokens(sample) + 1
        )


def test_compose_context_stress_sample_can_target_later_question():
    samples = [_sample("a", "alpha"), _sample("b", "beta")]

    combined = compose_context_stress_sample(
        samples,
        target_tokens=2,
        max_questions=1,
        question_offset=1,
    )

    assert [question.question_id for question in combined.questions] == ["q-b"]
    assert combined.questions[0].evidence_sources == ["b::shared-source"]
