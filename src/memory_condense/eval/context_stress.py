"""Deterministic long-context stress samples built from benchmark histories.

The public LongMemEval-S records are approximately 100k tokens each.  Summing
their metrics does not test retrieval from one million-token memory, so this
module combines complete locked-split histories into one sample while keeping
their source/evidence identities disjoint.
"""

from __future__ import annotations

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.ingest.loader import BenchmarkSample


def transcript_tokens(sample: BenchmarkSample) -> int:
    """Return content tokens in a sample's completed transcript."""

    return sum(count_tokens(text) for _role, text in sample.turns)


def compose_context_stress_sample(
    samples: list[BenchmarkSample],
    *,
    target_tokens: int,
    max_questions: int = 10,
    question_offset: int = 0,
) -> BenchmarkSample:
    """Combine histories until one sample meets ``target_tokens``.

    Source IDs are namespaced by their original sample ID.  This prevents an
    identically named distractor session in another LongMemEval record from
    receiving evidence credit.  Only the number of questions is bounded; all
    turns from every admitted history remain available as retrieval noise.
    """

    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    if max_questions < 1:
        raise ValueError("max_questions must be positive")
    if question_offset < 0:
        raise ValueError("question_offset must be non-negative")

    turns: list[tuple[str, str]] = []
    turn_source_ids: list[str | None] = []
    questions = []
    total_tokens = 0

    for sample in samples:
        turns.extend(sample.turns)
        source_ids = sample.turn_source_ids or [None] * len(sample.turns)
        if len(source_ids) != len(sample.turns):
            raise ValueError(
                f"sample {sample.sample_id!r} has misaligned turn source IDs"
            )

        def namespace(source_id: str | None) -> str | None:
            if source_id is None:
                return None
            return f"{sample.sample_id}::{source_id}"

        turn_source_ids.extend(namespace(source_id) for source_id in source_ids)
        question_stop = question_offset + max_questions
        if len(questions) < question_stop:
            for question in sample.questions:
                if len(questions) >= question_stop:
                    break
                questions.append(
                    question.model_copy(
                        update={
                            "evidence_sources": [
                                namespace(source_id)
                                for source_id in question.evidence_sources
                            ]
                        }
                    )
                )

        total_tokens += transcript_tokens(sample)
        if total_tokens >= target_tokens:
            break

    if total_tokens < target_tokens:
        raise ValueError(
            "not enough benchmark history for context stress target: "
            f"requested {target_tokens:,}, available {total_tokens:,} tokens"
        )
    if not questions:
        raise ValueError("context stress sample has no questions")
    questions = questions[question_offset : question_offset + max_questions]
    if not questions:
        raise ValueError("question_offset is outside the available stress questions")

    return BenchmarkSample(
        sample_id=f"context-stress-{target_tokens}",
        turns=turns,
        turn_source_ids=turn_source_ids,
        questions=questions,
    )
