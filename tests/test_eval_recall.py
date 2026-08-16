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

    def __init__(self, sample, mode, items=(), now_turn=100):
        self.texts = [t for _, t in sample.turns if t]
        self.mode = mode
        self._items = list(items)
        self.closed = False
        self.last_build_kwargs = None
        self.memory = SimpleNamespace(list_items=lambda: self._items)
        # Decay counts turns, so the survival projection needs to know where
        # the conversation is. 100 stands in for "a conversation has happened".
        self.transcript = SimpleNamespace(current_turn=lambda: now_turn)

    def _rank(self, query, k):
        q = set(query.lower().split())
        scored = sorted(
            self.texts, key=lambda t: len(q & set(t.lower().split())), reverse=True
        )
        return [
            SimpleNamespace(
                chunk=SimpleNamespace(text=t),
                turn=SimpleNamespace(source_id=f"source_{self.texts.index(t)}"),
            )
            for t in scored[:k]
        ]

    def search(self, query, k=10, ef_search=50):
        return self._rank(query, k)

    def search_hybrid(self, query, k=10, **kwargs):
        return self._rank(query, k)

    def search_sources(self, query, k_sources=4):
        return self._rank(query, k_sources)

    def search_anchored_sources(self, query, k=10, **kwargs):
        return self._rank(query, k)

    def search_hybrid_sources(self, query, k=10, **kwargs):
        return self._rank(query, k)

    def search_hybrid_graph(self, query, k=10, **kwargs):
        return self._rank(query, k)

    def search_hybrid_neighbors(self, query, k=10, **kwargs):
        return self._rank(query, k)

    def build_context(self, query, **kwargs):
        self.last_build_kwargs = kwargs
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


def _item(content: str, turns_old: float, importance: float = 0.8):
    """A memory item last accessed ``turns_old`` turns before turn 100."""
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
        last_access_turn=int(100 - turns_old),
    )


class TestRunRecall:
    def test_dense_mode_finds_the_answer_in_chunks(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="dense"))
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())

        assert report.n_questions == 1
        assert report.haystack_recall == 1.0
        assert report.recall == 1.0
        assert report.expansion_recall == 1.0
        assert report.header_recall == 0.0

    def test_source_mode_is_measured(self):
        config = EvalConfig(
            retrieval=RetrievalConfig(mode="source", k_sources=2)
        )
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())
        assert report.recall == 1.0

    def test_anchored_source_mode_is_measured(self):
        config = EvalConfig(
            retrieval=RetrievalConfig(mode="anchored_source", k=2)
        )
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())
        assert report.recall == 1.0

    def test_hybrid_source_mode_is_measured(self):
        config = EvalConfig(
            retrieval=RetrievalConfig(mode="hybrid_source", k=2, source_slots=4)
        )
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())
        assert report.recall == 1.0

    def test_hybrid_graph_mode_is_measured(self):
        config = EvalConfig(
            retrieval=RetrievalConfig(mode="hybrid_graph", k=2)
        )
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())
        assert report.recall == 1.0

    def test_hybrid_neighbor_mode_is_measured(self):
        config = EvalConfig(
            retrieval=RetrievalConfig(mode="hybrid_neighbor", k=2, neighbor_radius=1)
        )
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())
        assert report.recall == 1.0

    def test_evidence_source_coverage_scores_multi_source_retrieval(self):
        sample = SAMPLE.model_copy(
            update={
                "questions": [
                    SAMPLE.questions[0].model_copy(
                        update={"evidence_sources": ["source_0", "source_2"]}
                    )
                ]
            }
        )
        config = EvalConfig(retrieval=RetrievalConfig(mode="source", k_sources=1))

        report = run_recall([sample], config, ingest_fn=_ingest_fn())

        question = report.questions[0]
        assert question.evidence_source_hit is True
        assert question.evidence_source_recall == 0.5
        assert question.all_evidence_sources is False
        assert report.evidence_source_recall == 0.5
        assert report.evidence_any_source_recall == 1.0
        assert report.evidence_all_source_recall == 0.0

    def test_memory_mode_reports_where_the_answer_came_from(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())

        assert report.header_recall == 1.0
        assert report.expansion_recall == 1.0

    def test_memory_measurement_is_hybrid_and_does_not_reheat(self):
        store = _FakeStore(SAMPLE, "memory")
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))

        run_recall([SAMPLE], config, ingest_fn=lambda *args, **kwargs: store)

        assert store.last_build_kwargs["hybrid"] is True
        assert store.last_build_kwargs["reheat_memories"] is False

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
        assert report.haystack_recall == 0.0

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


class TestCostIsMeasuredAlongsideRecall:
    """Condensation's claim is fewer tokens, not more hits.

    A recall-only comparison structurally cannot show that, and actively
    rewards whichever arm sends more text — so cost is reported beside it.
    """

    def test_context_tokens_are_recorded(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3))
        report = run_recall([SAMPLE], config, ingest_fn=_ingest_fn())

        assert report.mean_context_tokens > 0
        assert all(q.context_tokens > 0 for q in report.questions)

    def test_recall_per_1k_tokens_rewards_the_cheaper_arm(self):
        """Equal recall at half the tokens must score twice as efficient."""
        from memory_condense.eval.recall import RecallReport

        cheap = RecallReport(recall=0.5, mean_context_tokens=1000)
        pricey = RecallReport(recall=0.5, mean_context_tokens=2000)

        assert cheap.recall_per_1k_tokens == pytest.approx(50.0)
        assert pricey.recall_per_1k_tokens == pytest.approx(25.0)

    def test_efficiency_is_zero_rather_than_dividing_by_zero(self):
        from memory_condense.eval.recall import RecallReport

        assert RecallReport(recall=0.5).recall_per_1k_tokens == 0.0


class TestDecaySurvival:
    """The measurement Phase 4's gate asked for.

    Before schema v4 this could not work: decay counted wall-clock seconds, an
    item needed 7-11.75 days of no access to reach COLD, and a run lasted
    minutes — so horizon 0 was always "everything survives" and the far
    horizons were always 0.0% by arithmetic. Decay now counts turns, so the
    run advances the coordinate itself and horizon 0 is a real reading.
    """

    def test_a_fresh_item_holds_the_answer_now_and_loses_it_later(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))
        report = run_recall(
            [SAMPLE],
            config,
            ingest_fn=_ingest_fn([_item("Storage is SQLite.", turns_old=0)]),
            horizons_turns=(0, 15, 30, 45),
        )

        assert report.survival_by_horizon[0] == 1.0
        # importance 0.8 seeds energy 0.8, which reaches COLD at ~50 turns.
        assert report.survival_by_horizon[15] == 1.0
        assert report.survival_by_horizon[30] == 1.0
        assert report.survival_by_horizon[45] == 1.0

    def test_an_important_item_outlives_an_ordinary_one(self):
        """The horizons must actually separate the two seed levels.

        The old day-based set could not do this: two of its four entries were
        past the theoretical ceiling for any unpinned item, so they reported
        0.0% regardless of what the store held.
        """
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))
        ordinary = run_recall(
            [SAMPLE],
            config,
            ingest_fn=_ingest_fn(
                [_item("Storage is SQLite.", turns_old=0, importance=0.2)]
            ),
            horizons_turns=(0, 45),
        )
        important = run_recall(
            [SAMPLE],
            config,
            ingest_fn=_ingest_fn(
                [_item("Storage is SQLite.", turns_old=0, importance=0.9)]
            ),
            horizons_turns=(0, 45),
        )

        assert ordinary.survival_by_horizon[0] == 1.0
        assert ordinary.survival_by_horizon[45] == 0.0
        assert important.survival_by_horizon[45] == 1.0

    def test_an_already_cold_item_never_counts(self):
        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))
        report = run_recall(
            [SAMPLE],
            config,
            ingest_fn=_ingest_fn([_item("Storage is SQLite.", turns_old=200)]),
            horizons_turns=(0, 45),
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
        item = _item("Storage is SQLite.", turns_old=5)
        before = (item.energy, item.last_access_turn)

        config = EvalConfig(retrieval=RetrievalConfig(k=3, mode="memory"))
        run_recall([SAMPLE], config, ingest_fn=_ingest_fn([item]))

        assert (item.energy, item.last_access_turn) == before


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
