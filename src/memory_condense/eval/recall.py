"""Retrieval-recall: can the gold answer even reach the prompt? No API calls.

Every benchmark question ships a gold answer string, so retrieval quality can
be measured **without generating anything**: ingest the haystack, assemble the
context the responder would see, and ask a purely local question — is the
answer recoverable from it?

This matters because it is the cheap predictor of the expensive run. If the
memory arm's context contains the answer *less often* than the dense arm's,
no responder can recover the difference, and the paid comparison's outcome is
knowable in advance for zero dollars.

It also answers the question that gated design Phase 4. ``08 - Analysis/01``
showed COLD is unreachable in a live run: seeds of 0.5/0.8 against a seven-day
half-life need 7–11.75 days of no access, while a run lasts minutes. Rather
than inject a clock into the live path, this module **replays decay forward**
over the recorded item state at simulated offsets, which answers "would the
answer still be reachable once these items went cold?" with no clock
manipulation and no waiting.

Deliberately no LLM, no key, no network. It is a fifth CLI mode beside
``--compare``, and both are free.
"""

from __future__ import annotations

import tempfile
from datetime import timedelta
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

#: Simulated ages, in days, at which to re-tier the items and ask whether the
#: answer survives. 0 is "now"; 30 is comfortably past the point where an
#: untouched item at either seed energy has fallen below WARM.
DEFAULT_HORIZONS_DAYS = (0, 7, 14, 30)


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
    #: ``{horizon_days: answer_still_in_a_non_cold_item}``
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


def _survival(mc, gold: str, horizons_days) -> dict[int, bool]:
    """Would the answer still sit in a non-COLD memory item after N days?

    Replays :func:`decay.effective_energy` forward over the stored items rather
    than touching any clock the live path reads. An empty memory store (the
    chunk arms, where nothing is extracted) yields ``False`` at every horizon —
    correctly: there is no memory item holding the answer.
    """
    items = mc.memory.list_items()
    now = decay.now_utc()
    out: dict[int, bool] = {}
    for days in horizons_days:
        at = now + timedelta(days=days)
        alive = [
            f"{i.content} {i.details or ''}"
            for i in items
            if decay.item_heat(i, now=at) is not decay.Heat.COLD
        ]
        out[days] = contains_answer(alive, gold)
    return out


def measure_sample(
    sample: BenchmarkSample,
    config: EvalConfig,
    data_dir: Path,
    ingest_fn: IngestFn = ingest_sample,
    horizons_days=DEFAULT_HORIZONS_DAYS,
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
                    survives_horizon=_survival(mc, question.answer, horizons_days),
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
    horizons_days=DEFAULT_HORIZONS_DAYS,
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
                    horizons_days=horizons_days,
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
            for days in horizons_days
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
        print("Answer still held by a non-COLD memory item, by simulated age:")
        for days, frac in sorted(report.survival_by_horizon.items()):
            print(f"  +{days:>3}d {frac:>7.1%}")

    if report.by_category:
        print()
        print(f"{'category':<40}{'recall':>8}")
        print("-" * 48)
        for category, frac in report.by_category.items():
            print(f"{category[:40]:<40}{frac:>8.1%}")
    print()
