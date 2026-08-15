"""Retrieval-recall: can the gold answer even reach the prompt? No API calls.

Every benchmark question ships a gold answer string, so retrieval quality can
be measured **without generating anything**: ingest the haystack, assemble the
context the responder would see, and ask a purely local question — is the
answer recoverable from it?

This matters because it is the cheap predictor of the expensive run. If the
memory arm's context contains the answer *less often* than the dense arm's,
no responder can recover the difference, and the paid comparison's outcome is
knowable in advance for zero dollars.

It also answers the question that gated design Phase 4: **do COLD items hold
answers nothing else holds?** Before schema v4 that question was unanswerable
— decay counted wall-clock seconds, so an item needed 7–11.75 days of no
access to reach COLD while a run lasted minutes, and the horizons this module
projected to were mostly incapable of returning anything but zero. Decay now
counts *turns*, so a run advances the coordinate on its own and the tiers
populate for real. The forward projection below is consequently a genuine
extrapolation of a live signal rather than a workaround for a dead one.

Deliberately no LLM, no key, no network. It is a fifth CLI mode beside
``--compare``, and both are free.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from memory_condense import decay
from memory_condense._tokenizer import count_tokens
from memory_condense.eval.benchmark import (
    IngestFn,
    ingest_sample,
    normalize_answer,
    f1_score,
)
from memory_condense.eval.schemas import EvalConfig
from memory_condense.loader import BenchmarkSample

#: Simulated ages, **in turns**, at which to re-tier the items and ask whether
#: the answer survives. 0 is "now" (the transcript's current position).
#:
#: Chosen to straddle both crossing points at the default 30-turn half-life:
#: an ordinary item (seed 0.5) crosses into COLD at 30 untouched turns and an
#: important one (seed 0.8) at ~50. So 15 is inside WARM for both, 30 is the
#: first divider, and 45 separates them.
#:
#: 60 is deliberately **excluded**. Energy is clamped to ``<= 1.0`` and COLD
#: begins below ``0.25 = 1.0 * 0.5**2``, so two half-lives is the theoretical
#: ceiling for *any* unpinned item and that horizon can only ever report 0.0%.
#: The previous day-based set had exactly that defect in two of its four
#: entries; a horizon that cannot vary is not a measurement.
DEFAULT_HORIZONS_TURNS = (0, 15, 30, 45)


class QuestionRecall(BaseModel):
    """Whether one question's answer was reachable, from where, and at what cost."""

    question_id: str
    category: str = ""
    in_context: bool = False
    best_f1: float = 0.0
    in_memory_header: bool = False
    in_expansions: bool = False
    #: tiktoken count of the assembled context. Load-bearing: condensation's
    #: claim is *the same answer for fewer tokens*, so recall alone cannot
    #: show its benefit — and can make a system that spends 10x look better.
    context_tokens: int = 0
    #: ``{horizon_turns: answer_still_in_a_non_cold_item}``
    survives_horizon: dict[int, bool] = Field(default_factory=dict)


class RecallReport(BaseModel):
    """Aggregate answer-reachability for one config over one benchmark.

    Reports **recall and cost together**. Condensation's claim is not "finds
    the answer more often" but "finds it as often for fewer tokens", so a
    recall-only comparison structurally cannot show its benefit, and rewards
    whichever arm simply sends more text. ``recall_per_1k_tokens`` is the
    efficiency figure; ``mean_context_tokens`` is what it is normalised by.
    """

    benchmark: str = ""
    mode: str = "dense"
    k: int = 10
    n_questions: int = 0
    recall: float = 0.0
    mean_best_f1: float = 0.0
    header_recall: float = 0.0
    expansion_recall: float = 0.0
    mean_context_tokens: float = 0.0
    survival_by_horizon: dict[int, float] = Field(default_factory=dict)
    by_category: dict[str, float] = Field(default_factory=dict)
    questions: list[QuestionRecall] = Field(default_factory=list)

    @property
    def recall_per_1k_tokens(self) -> float:
        """Recall points earned per 1,000 tokens of context spent."""
        if not self.mean_context_tokens:
            return 0.0
        return self.recall * 100.0 / (self.mean_context_tokens / 1000.0)


def contains_answer(texts: list[str], gold: str) -> bool:
    """Whether ``gold`` appears in any of ``texts`` under SQuAD normalization.

    Containment, not equality: the context is a passage and the answer is a
    span inside it. Normalization (lowercase, strip articles and punctuation)
    is the same one the benchmark grades with, so this measures the same notion
    of "the answer is there" that F1 does.
    """
    needle = normalize_answer(gold)
    if not needle:
        return False
    return any(needle in normalize_answer(t) for t in texts)


def best_f1(texts: list[str], gold: str) -> float:
    """Highest token-F1 between the gold answer and any single context piece.

    A softer signal than containment: it still scores when the answer is
    present but reworded, which containment misses entirely.
    """
    return max((f1_score(t, gold) for t in texts), default=0.0)


def _assemble(mc, question: str, config: EvalConfig) -> tuple[list[str], list[str]]:
    """The context this config would send. Returns ``(header_texts, body_texts)``.

    ``reheat`` is off throughout: this is a measurement, and an item must not
    become hotter merely because a measurement looked at it.
    """
    if config.retrieval.mode == "memory":
        packed = mc.build_context(
            question,
            recent_turns=0,
            k_memories=config.retrieval.k_memories,
            k_expansions=config.retrieval.k,
            hybrid=config.retrieval.effective_hybrid,
        )
        header = [packed.memory_header] if packed.memory_header else []
        return header, list(packed.expansions)

    if config.retrieval.mode == "span":
        results = mc.search_spans(
            question,
            levels=config.retrieval.span_levels,
            k_per_level=config.retrieval.k_per_level,
        )
    elif config.retrieval.effective_hybrid:
        results = mc.search_hybrid(
            question,
            k=config.retrieval.k,
            ef_search=config.retrieval.ef_search,
            candidates=config.retrieval.candidates,
            alpha=config.retrieval.alpha,
        )
    else:
        results = mc.search(
            question, k=config.retrieval.k, ef_search=config.retrieval.ef_search
        )
    return [], [r.chunk.text for r in results]


def _survival(mc, gold: str, horizons_turns) -> dict[int, bool]:
    """Would the answer still sit in a non-COLD memory item N turns from now?

    Projects :func:`decay.effective_energy` forward over the stored items from
    the transcript's current position. Horizon 0 is not a projection at all —
    it is the store as it stands, which is now a real reading because turns
    have actually elapsed during the run.

    An empty memory store (the chunk arms, where nothing is extracted) yields
    ``False`` at every horizon — correctly: there is no memory item holding
    the answer.
    """
    items = mc.memory.list_items()
    now_turn = mc.transcript.current_turn()
    out: dict[int, bool] = {}
    for turns in horizons_turns:
        alive = [
            f"{i.content} {i.details or ''}"
            for i in items
            if decay.item_heat(i, now_turn=now_turn + turns) is not decay.Heat.COLD
        ]
        out[turns] = contains_answer(alive, gold)
    return out


def measure_sample(
    sample: BenchmarkSample,
    config: EvalConfig,
    data_dir: Path,
    ingest_fn: IngestFn = ingest_sample,
    horizons_turns=DEFAULT_HORIZONS_TURNS,
) -> list[QuestionRecall]:
    """Ingest one sample and measure answer reachability for its questions."""
    mc = ingest_fn(sample, config, data_dir)
    try:
        out: list[QuestionRecall] = []
        for question in sample.questions:
            header, body = _assemble(mc, question.question, config)
            everything = header + body
            out.append(
                QuestionRecall(
                    question_id=question.question_id,
                    category=question.category or "",
                    in_context=contains_answer(everything, question.answer),
                    best_f1=best_f1(everything, question.answer),
                    in_memory_header=contains_answer(header, question.answer),
                    in_expansions=contains_answer(body, question.answer),
                    context_tokens=sum(count_tokens(t) for t in everything),
                    survives_horizon=_survival(mc, question.answer, horizons_turns),
                )
            )
        return out
    finally:
        mc.close()


def run_recall(
    samples: list[BenchmarkSample],
    config: EvalConfig,
    benchmark: str = "",
    max_samples: int | None = None,
    ingest_fn: IngestFn = ingest_sample,
    horizons_turns=DEFAULT_HORIZONS_TURNS,
) -> RecallReport:
    """Measure answer reachability across samples. Zero API calls."""
    selected = samples[:max_samples] if max_samples else samples

    results: list[QuestionRecall] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, sample in enumerate(selected):
            print(f"  [{i + 1}/{len(selected)}] {sample.sample_id} "
                  f"({len(sample.turns)} turns, {len(sample.questions)} questions)...")
            results.extend(
                measure_sample(
                    sample,
                    config,
                    Path(tmpdir) / f"sample_{i}",
                    ingest_fn=ingest_fn,
                    horizons_turns=horizons_turns,
                )
            )

    n = len(results)
    by_category: dict[str, list[bool]] = {}
    for r in results:
        by_category.setdefault(r.category or "uncategorized", []).append(r.in_context)

    return RecallReport(
        benchmark=benchmark,
        mode=config.retrieval.mode,
        k=config.retrieval.k,
        n_questions=n,
        recall=_frac(r.in_context for r in results),
        mean_best_f1=(sum(r.best_f1 for r in results) / n) if n else 0.0,
        header_recall=_frac(r.in_memory_header for r in results),
        expansion_recall=_frac(r.in_expansions for r in results),
        mean_context_tokens=(sum(r.context_tokens for r in results) / n) if n else 0.0,
        survival_by_horizon={
            days: _frac(r.survives_horizon.get(days, False) for r in results)
            for days in horizons_turns
        },
        by_category={k: _frac(v) for k, v in sorted(by_category.items())},
        questions=results,
    )


def _frac(flags) -> float:
    flags = list(flags)
    return (sum(1 for f in flags if f) / len(flags)) if flags else 0.0


def print_recall_report(report: RecallReport) -> None:
    """Human-readable summary. No API calls were made to produce any of this."""
    print()
    print("=" * 72)
    print("ANSWER REACHABILITY (offline — no API calls)")
    print(f"  benchmark: {report.benchmark or '(unnamed)'}")
    print(f"  mode     : {report.mode}  k={report.k}")
    print(f"  questions: {report.n_questions}")
    print("=" * 72)
    print(f"{'answer present in context':<34}{report.recall:>8.1%}")
    print(f"{'mean best token-F1':<34}{report.mean_best_f1:>8.3f}")
    if report.mode == "memory":
        print(f"{'  ...via the memory header':<34}{report.header_recall:>8.1%}")
        print(f"{'  ...via verbatim expansions':<34}{report.expansion_recall:>8.1%}")
    print()
    print(f"{'mean context tokens':<34}{report.mean_context_tokens:>8.0f}")
    print(f"{'recall per 1k tokens':<34}{report.recall_per_1k_tokens:>8.2f}")
    print("  (condensation wins by costing less, not only by recalling more —")
    print("   compare arms on this row as well as the one above)")

    if report.survival_by_horizon:
        print()
        print("Answer still held by a non-COLD memory item, by turns ahead:")
        for turns, frac in sorted(report.survival_by_horizon.items()):
            label = "now" if turns == 0 else f"+{turns}t"
            print(f"  {label:>5} {frac:>7.1%}")

    if report.by_category:
        print()
        print(f"{'category':<40}{'recall':>8}")
        print("-" * 48)
        for category, frac in report.by_category.items():
            print(f"{category[:40]:<40}{frac:>8.1%}")
    print()
