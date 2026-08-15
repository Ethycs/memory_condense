"""Answer-reachability harness. Fully offline — no API, no key, no model."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense import decay
from memory_condense.eval.recall import (
    QuestionRecall,
    best_f1,
    contains_answer,
    run_recall,
)
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from memory_condense.loader import BenchmarkQuestion, BenchmarkSample
from memory_condense.schemas import (
    CreateOp,
    MemoryType,
    PackedContext,
    Provenance,
)

SAMPLE = BenchmarkSample(
    sample_id="s1",
    turns=[
        ("user", "I prefer dark mode in all my apps."),
        ("assistant", "Noted."),
        ("user", "We decided to use SQLite for storage."),
        ("assistant", "SQLite in WAL mode never blocks readers."),
    ],
    questions=[
        BenchmarkQuestion(
            question_id="q1",
            question="What storage did we choose?",
            answer="SQLite",
            category="single-session-user",
        )
    ],
)


class TestContainsAnswer:
    def test_finds_a_span_inside_a_passage(self):
        assert contains_answer(["We decided to use SQLite for storage."], "SQLite")

    def test_normalizes_case_articles_and_punctuation(self):
        assert contains_answer(["the answer is sqlite!"], "SQLite")

    def test_absent_answer_is_false(self):
        assert not contains_answer(["We chose Postgres."], "SQLite")

    def test_empty_inputs_are_false_not_true(self):
        """A vacuous substring match would score every empty answer as found."""
        assert not contains_answer(["anything"], "")
        assert not contains_answer([], "SQLite")


class TestBestF1:
    def test_scores_a_reworded_answer_containment_would_miss(self):
        assert best_f1(["storage is handled by sqlite"], "SQLite storage") > 0.0

    def test_empty_context_scores_zero_rather_than_raising(self):
        assert best_f1([], "SQLite") == 0.0


class _FakeStore:
    """Chunk retrieval by token overlap; no embeddings, no downloads."""

    def __init__(self, sample, mode, items=()):
        self.texts = [t for _, t in sample.turns if t]
        self.mode = mode
        self._items = list(items)
        self.closed = False
        self.memory = SimpleNamespace(list_items=lambda: self._items)

    def _rank(self, query, k):
        q = set(query.lower().split())
        scored = sorted(
            self.texts, key=lambda t: len(q & set(t.lower().split())), reverse=True
        )
        return [SimpleNamespace(chunk=SimpleNamespace(text=t)) for t in scored[:k]]

    def search(self, query, k=10, ef_search=50):
        return self._rank(query, k)

    def search_hybrid(self, query, k=10, **kwargs):
        return self._rank(query, k)

    def build_context(self, query, **kwargs):
        return PackedContext(
            memory_header="Relevant memory:\n- [Decision] Storage is SQLite.",
            expansions=[r.chunk.text for r in self._rank(query, kwargs.get("k_expansions", 3))],
        )

    def close(self):
        self.closed = True


def _ingest_fn(items=()):
    def fn(sample, config, data_dir: Path):
        return _FakeStore(sample, config.retrieval.mode, items)

    return fn


def _item(content: str, days_old: float, importance: float = 0.8):
    op = CreateOp(
        type=MemoryType.DECISION,
        content=content,
        provenance=[Provenance(turn_id="t1", quote=content)],
        importance=importance,
    )
    from memory_condense.schemas import MemoryItem

    return MemoryItem(
        type=op.type,
        content=op.content,
        provenance=op.provenance,
        importance=importance,
        energy=decay.seed_energy(importance),
        last_access_at=decay.now_utc() - timedelta(days=days_old),
    )


class TestRunRecall:
    def test_dense_mode_finds_the_answer_in_chunks(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="dense"))
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())

        assert report.n_questions == 1
        assert report.recall == 1.0
        assert report.expansion_recall == 1.0
        assert report.header_recall == 0.0

    def test_memory_mode_reports_where_the_answer_came_from(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())

        assert report.header_recall == 1.0
        assert report.expansion_recall == 1.0

    def test_an_unanswerable_question_scores_zero(self):
        sample = SAMPLE.model_copy(
            update={
                "questions": [
                    BenchmarkQuestion(
                        question_id="q2",
                        question="What did we pick?",
                        answer="Cassandra",
                    )
                ]
            }
        )
        config = EvalConfig(retrieval=RetrievalConfig(k=3))
        report = run_recall([sample], config, ingest_fn=_ingest_fn())

        assert report.recall == 0.0

    def test_max_samples_limits_work(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3))
        report = run_recall(
            [SAMPLE, SAMPLE], config, max_samples=1, ingest_fn=_ingest_fn()
        )
        assert report.n_questions == 1

    def test_the_store_is_closed_even_on_the_happy_path(self):
        store = _FakeStore(SAMPLE, "dense")
        config = EvalConfig(retrieval=RetrievalConfig(k=3))
        run_recall([SAMPLE], config, ingest_fn=lambda *a, **kw: store)
        assert store.closed

    def test_categories_are_broken_out(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3))
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())
        assert report.by_category == {"single-session-user": 1.0}


class TestDecaySurvival:
    """The measurement Phase 4's gate asked for, without touching a clock.

    `08 - Analysis/01` showed COLD is unreachable in a live run — an item needs
    7–11.75 days of no access and a run lasts minutes. Replaying decay forward
    over recorded items answers the question without waiting or injecting a
    clock into the live path.
    """

    def test_a_fresh_item_holds_the_answer_now_and_loses_it_later(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))
        report = run_recall(
            [SAMPLE],
            config,
            ingest_fn=_ingest_fn([_item("Storage is SQLite.", days_old=0)]),
            horizons_days=(0, 7, 14, 30),
        )

        assert report.survival_by_horizon[0] == 1.0
        # importance 0.8 seeds energy 0.8, which reaches COLD at 11.75 days.
        assert report.survival_by_horizon[7] == 1.0
        assert report.survival_by_horizon[14] == 0.0
        assert report.survival_by_horizon[30] == 0.0

    def test_an_already_cold_item_never_counts(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))
        report = run_recall(
            [SAMPLE],
            config,
            ingest_fn=_ingest_fn([_item("Storage is SQLite.", days_old=60)]),
            horizons_days=(0, 30),
        )
        assert report.survival_by_horizon[0] == 0.0

    def test_chunk_modes_report_no_survival_because_there_are_no_items(self):
        """Not a bug: dense mode holds the answer in chunks, not memory items."""
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="dense"))
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())

        assert report.recall == 1.0
        assert all(v == 0.0 for v in report.survival_by_horizon.values())

    def test_measuring_does_not_reheat(self):
        """A measurement must not make the thing it measures hotter."""
        item = _item("Storage is SQLite.", days_old=5)
        before = (item.energy, item.last_access_at)

        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))
        run_recall([SAMPLE], config, ingest_fn=_ingest_fn([item]))

        assert (item.energy, item.last_access_at) == before


def test_report_prints_without_raising(capsys):
    from memory_condense.eval.recall import RecallReport, print_recall_report

    print_recall_report(
        RecallReport(
            benchmark="mini",
            mode="memory",
            n_questions=2,
            recall=0.5,
            survival_by_horizon={0: 1.0, 30: 0.0},
            by_category={"a": 1.0},
            questions=[QuestionRecall(question_id="q1")],
        )
    )
    out = capsys.readouterr().out
    assert "ANSWER REACHABILITY" in out
    assert "no API calls" in out
