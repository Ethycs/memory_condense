from __future__ import annotations

from pathlib import Path

from memory_condense.eval.schemas import EvalConfig, RetrievalConfig, UsageStats
from memory_condense.eval.sufficiency import (
    audit_sample,
    build_sufficiency_prompt,
    run_sufficiency_audit,
)
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample
from memory_condense.domain.schemas import Chunk, RetrievalResult, Turn


SAMPLE = BenchmarkSample(
    sample_id="temporal",
    turns=[
        ("user", "I joined Book Lovers Unite three weeks ago."),
        ("assistant", "You attended its meetup last week."),
        ("user", "An unrelated source says hello."),
    ],
    turn_source_ids=["books", "books", "other"],
    questions=[
        BenchmarkQuestion(
            question_id="q1",
            question="How long had I been a member when I attended?",
            answer="two weeks",
            category="temporal-reasoning",
            evidence_sources=["books"],
        )
    ],
)


class OneTurnStore:
    def search(self, query, k=10, ef_search=50):
        turn = Turn(
            turn_id="t1",
            role="user",
            text=SAMPLE.turns[0][1],
            source_id="books",
        )
        chunk = Chunk(
            chunk_id="c1",
            turn_id=turn.turn_id,
            text=turn.text,
            start_char=0,
            end_char=len(turn.text),
            token_count=10,
        )
        return [RetrievalResult(chunk=chunk, turn=turn, score=1.0)]

    def close(self):
        pass


def _ingest(sample, config, data_dir: Path):
    return OneTurnStore()


def _temporal_judge(question, gold, context):
    text = " ".join(context)
    sufficient = "three weeks" in text and "last week" in text
    return sufficient, "both dates present" if sufficient else "one date missing", UsageStats(calls=1)


def test_source_coverage_can_be_perfect_while_evidence_is_insufficient(tmp_path):
    rows = audit_sample(
        SAMPLE,
        EvalConfig(retrieval=RetrievalConfig(mode="dense"), max_prompt_tokens=8000),
        tmp_path,
        ingest_fn=_ingest,
        sufficiency_fn=_temporal_judge,
    )

    row = rows[0]
    assert row.evidence_granularity == "source"
    assert row.evidence_source_recall == 1.0
    assert row.all_evidence_sources is True
    assert row.gold_source_literal is False
    assert row.requires_inference is True
    assert row.oracle_sufficient is True
    assert row.retrieved_sufficient is False
    assert row.gap == "retrieval_or_packing_gap"
    assert row.judge_usage.calls == 2


def test_sufficiency_report_tracks_oracle_retention():
    report = run_sufficiency_audit(
        [SAMPLE],
        EvalConfig(retrieval=RetrievalConfig(mode="dense"), max_prompt_tokens=8000),
        ingest_fn=_ingest,
        sufficiency_fn=_temporal_judge,
    )

    assert report.n_evidence_labeled == 1
    assert report.mean_evidence_source_recall == 1.0
    assert report.oracle_sufficiency == 1.0
    assert report.retrieved_sufficiency == 0.0
    assert report.sufficiency_retention == 0.0
    assert report.judge_usage.calls == 2


def test_sufficiency_prompt_explicitly_allows_derivation():
    messages = build_sufficiency_prompt("When?", "two weeks", ["three weeks", "last week"])

    assert "Arithmetic" in messages[0]["content"]
    assert "Known gold answer: two weeks" in messages[1]["content"]
