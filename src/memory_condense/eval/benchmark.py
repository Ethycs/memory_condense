"""QA-probe evaluation against public agent-memory benchmarks.

This is a different protocol from :mod:`memory_condense.eval.runner`. The
replay runner interleaves generation and ingestion turn by turn against the
user's own conversations. Here we:

1. Ingest a sample's **entire** haystack into a fresh memory store.
2. For each question about that sample, retrieve top-k chunks and ask an
   injected ``answer_fn`` to answer *only* from those chunks.
3. Grade the answer against the gold answer with token-level F1 / exact match,
   and optionally with an injected LLM ``judge_fn`` for semantic equivalence.

This is the protocol LongMemEval and LoCoMo report, so the numbers produced
here are directly comparable to published SimpleMem / Mem0 / Zep results.

This module deliberately does **not** import ``litellm``. ``answer_fn`` and
``judge_fn`` are injected callables, which keeps the module testable offline
and provider-agnostic; CLI wiring lives elsewhere.
"""

from __future__ import annotations

import re
import string
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Protocol

from pydantic import BaseModel, Field

from memory_condense.condenser import MemoryCondenser
from memory_condense.context_packer import ContextBudget
from memory_condense.eval.schemas import EvalConfig
from memory_condense.loader import BenchmarkQuestion, BenchmarkSample

# ---------------------------------------------------------------------------
# Injected callable types
# ---------------------------------------------------------------------------

#: ``answer_fn(messages) -> answer_text``. ``messages`` is an OpenAI-style
#: chat list, so a litellm-backed implementation is a one-liner.
AnswerFn = Callable[[list[dict[str, str]]], str]

#: ``judge_fn(question, gold_answer, predicted_answer) -> (is_correct, reason)``
JudgeFn = Callable[[str, str, str], tuple[bool, str]]


class SupportsSearch(Protocol):
    """Minimal surface :func:`run_benchmark` needs from an ingested store.

    :class:`memory_condense.condenser.MemoryCondenser` satisfies this; tests
    can substitute a fake to avoid downloading an embedding model.

    ``search_hybrid`` and ``build_context`` are only called in their
    corresponding modes, so a fake that implements ``search`` alone remains
    valid for the dense arm.
    """

    def search(self, query: str, k: int = ..., ef_search: int = ...): ...

    def close(self) -> None: ...


#: ``ingest_fn(sample, config, data_dir) -> SupportsSearch``
IngestFn = Callable[[BenchmarkSample, EvalConfig, Path], SupportsSearch]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = (
    "You are answering questions about a long conversation history. "
    "You are given excerpts retrieved from that history as your only source "
    "of information.\n\n"
    "Answer the question using ONLY the retrieved excerpts. Be as short as "
    "possible: reply with just the fact, name, number, or date asked for — "
    "no preamble, no explanation, no full sentences unless the question "
    "requires one. If the excerpts do not contain the answer, reply exactly: "
    "I don't know."
)

QA_USER_TEMPLATE = (
    "Retrieved excerpts from the conversation history:\n"
    "{context}\n\n"
    "Question: {question}\n"
    "Short answer:"
)

QA_NO_CONTEXT = "(no excerpts retrieved)"

JUDGE_SYSTEM_PROMPT = (
    "You grade answers to questions about a conversation history. "
    "Decide whether the predicted answer is semantically equivalent to the "
    "gold answer — same facts, regardless of wording, formatting, or extra "
    "detail. Minor paraphrase is correct; a different or missing fact is "
    "incorrect."
)

JUDGE_USER_TEMPLATE = (
    "Question: {question}\n"
    "Gold answer: {gold}\n"
    "Predicted answer: {prediction}\n\n"
    "Is the predicted answer correct? Reply CORRECT or INCORRECT, then a "
    "one-sentence reason."
)


# ---------------------------------------------------------------------------
# Result schemas
# ---------------------------------------------------------------------------


class BenchmarkQuestionResult(BaseModel):
    """Grading outcome for one QA probe."""

    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    category: Optional[str] = None
    retrieved_chunks: list[str] = Field(default_factory=list)
    f1: float = 0.0
    exact_match: bool = False
    judge_correct: Optional[bool] = None
    judge_reasoning: Optional[str] = None


class BenchmarkSampleResult(BaseModel):
    """Aggregated results for all questions on one benchmark sample."""

    sample_id: str
    num_turns: int
    num_questions: int
    question_results: list[BenchmarkQuestionResult] = Field(default_factory=list)
    mean_f1: float = 0.0
    exact_match_rate: float = 0.0
    judge_accuracy: Optional[float] = None


class CategoryMetrics(BaseModel):
    """Per-question-type breakdown (published results report these)."""

    category: str
    num_questions: int
    mean_f1: float
    exact_match_rate: float
    judge_accuracy: Optional[float] = None


class BenchmarkRunResult(BaseModel):
    """Results from one benchmark run under one config."""

    config: EvalConfig
    benchmark: str = ""
    samples: list[BenchmarkSampleResult] = Field(default_factory=list)
    num_samples: int = 0
    num_questions: int = 0
    mean_f1: float = 0.0
    exact_match_rate: float = 0.0
    judge_accuracy: Optional[float] = None
    by_category: dict[str, CategoryMetrics] = Field(default_factory=dict)
    run_timestamp: str = ""


# ---------------------------------------------------------------------------
# Grading — pure, dependency-free
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD answer normalization.

    Lowercase, strip punctuation, remove articles, collapse whitespace.
    """
    if not text:
        return ""
    lowered = text.lower()
    no_punct = lowered.translate(_PUNCT_TABLE)
    no_articles = _ARTICLES_RE.sub(" ", no_punct)
    return " ".join(no_articles.split())


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold, after SQuAD normalization.

    This is the standard LongMemEval / LoCoMo string metric. Returns 1.0 when
    both normalize to the same (possibly empty) token bag, 0.0 when they are
    disjoint or exactly one side is empty.
    """
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> bool:
    """Normalized exact-match between prediction and gold."""
    return normalize_answer(prediction) == normalize_answer(gold)


# ---------------------------------------------------------------------------
# Ingest / answer
# ---------------------------------------------------------------------------


def ingest_sample(
    sample: BenchmarkSample,
    config: EvalConfig,
    data_dir: str | Path,
) -> MemoryCondenser:
    """Build a fresh memory store from a sample's full haystack.

    The caller owns the returned condenser and must ``close()`` it.
    """
    # auto_extract follows the mode, as in runner.py: in dense/hybrid mode the
    # QA prompt is built from retrieved chunks and extraction would cost time
    # per ingest without changing a single score. Haystacks are large, so this
    # matters more here than in the replay eval — and in memory mode it is the
    # dominant local cost, which is why --max-samples exists.
    memory_mode = config.retrieval.mode == "memory"
    mc = MemoryCondenser(
        data_dir=data_dir,
        chunker_min_tokens=config.chunker.min_tokens,
        chunker_max_tokens=config.chunker.max_tokens,
        auto_extract=memory_mode,
        budget=(
            ContextBudget(max_expansions=max(config.retrieval.k, 1))
            if memory_mode
            else None
        ),
    )
    for role, text in sample.turns:
        if text:
            mc.ingest(role, text)
    return mc


def build_qa_prompt(question: str, chunk_texts: list[str]) -> list[dict[str, str]]:
    """Build the chat messages for a QA probe from retrieved chunk texts."""
    if chunk_texts:
        context = "\n".join(
            f"[{i + 1}] {text}" for i, text in enumerate(chunk_texts)
        )
    else:
        context = QA_NO_CONTEXT

    return [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": QA_USER_TEMPLATE.format(context=context, question=question),
        },
    ]


def build_judge_prompt(
    question: str, gold: str, prediction: str
) -> list[dict[str, str]]:
    """Build the chat messages for the semantic-equivalence judge.

    Provided so a CLI-side ``judge_fn`` can share this module's wording.
    """
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(
                question=question, gold=gold, prediction=prediction
            ),
        },
    ]


def answer_question(
    mc: SupportsSearch,
    question: BenchmarkQuestion,
    config: EvalConfig,
    answer_fn: AnswerFn,
) -> tuple[str, list[str]]:
    """Assemble context for ``question`` and answer from it.

    In ``dense``/``hybrid`` mode the context is the top-k chunks. In ``memory``
    mode it is what ``build_context`` produces — the memory-item header plus
    budgeted verbatim expansions — which is the only way this harness exercises
    ``ContextPacker``, ``MemoryStore.retrieve``, ``rank_score`` or ``decay``.

    Returns ``(answer_text, context_texts)``.
    """
    if config.retrieval.mode == "memory":
        packed = mc.build_context(
            question.question,
            # A haystack is not a live conversation: its "last 8 turns" are an
            # arbitrary slice of someone else's dialogue and would be noise in
            # the prompt. The QA protocol wants retrieved context only.
            recent_turns=0,
            k_memories=config.retrieval.k_memories,
            k_expansions=config.retrieval.k,
            hybrid=config.retrieval.effective_hybrid,
        )
        # `expansions` is already rendered text, unlike the chunk arms'
        # RetrievalResult objects.
        context_texts = [t for t in [packed.memory_header] if t]
        context_texts += list(packed.expansions)
    elif config.retrieval.effective_hybrid:
        retrieved = mc.search_hybrid(
            question.question,
            k=config.retrieval.k,
            ef_search=config.retrieval.ef_search,
            candidates=config.retrieval.candidates,
            alpha=config.retrieval.alpha,
        )
        context_texts = [r.chunk.text for r in retrieved]
    else:
        retrieved = mc.search(
            question.question,
            k=config.retrieval.k,
            ef_search=config.retrieval.ef_search,
        )
        context_texts = [r.chunk.text for r in retrieved]

    messages = build_qa_prompt(question.question, context_texts)
    answer = answer_fn(messages)

    return (answer or "").strip(), context_texts


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_sample(
    sample: BenchmarkSample,
    config: EvalConfig,
    answer_fn: AnswerFn,
    data_dir: str | Path,
    judge_fn: JudgeFn | None = None,
    ingest_fn: IngestFn = ingest_sample,
) -> BenchmarkSampleResult:
    """Ingest one sample's haystack, then grade every question about it."""
    question_results: list[BenchmarkQuestionResult] = []

    mc = ingest_fn(sample, config, Path(data_dir))
    try:
        for question in sample.questions:
            prediction, chunk_texts = answer_question(
                mc, question, config, answer_fn
            )

            judge_correct: bool | None = None
            judge_reasoning: str | None = None
            if judge_fn is not None:
                judge_correct, judge_reasoning = judge_fn(
                    question.question, question.answer, prediction
                )

            question_results.append(
                BenchmarkQuestionResult(
                    question_id=question.question_id,
                    question=question.question,
                    gold_answer=question.answer,
                    predicted_answer=prediction,
                    category=question.category,
                    retrieved_chunks=[t[:200] for t in chunk_texts[:5]],
                    f1=f1_score(prediction, question.answer),
                    exact_match=exact_match(prediction, question.answer),
                    judge_correct=judge_correct,
                    judge_reasoning=judge_reasoning,
                )
            )
    finally:
        close = getattr(mc, "close", None)
        if callable(close):
            close()

    judged = [
        qr.judge_correct for qr in question_results if qr.judge_correct is not None
    ]

    return BenchmarkSampleResult(
        sample_id=sample.sample_id,
        num_turns=len(sample.turns),
        num_questions=len(question_results),
        question_results=question_results,
        mean_f1=_mean([qr.f1 for qr in question_results]),
        exact_match_rate=_mean(
            [1.0 if qr.exact_match else 0.0 for qr in question_results]
        ),
        judge_accuracy=_mean([1.0 if c else 0.0 for c in judged]) if judged else None,
    )


def _category_breakdown(
    results: list[BenchmarkQuestionResult],
) -> dict[str, CategoryMetrics]:
    """Group question results by category and aggregate each group."""
    groups: dict[str, list[BenchmarkQuestionResult]] = {}
    for qr in results:
        groups.setdefault(qr.category or "uncategorized", []).append(qr)

    breakdown: dict[str, CategoryMetrics] = {}
    for category, items in sorted(groups.items()):
        judged = [qr.judge_correct for qr in items if qr.judge_correct is not None]
        breakdown[category] = CategoryMetrics(
            category=category,
            num_questions=len(items),
            mean_f1=_mean([qr.f1 for qr in items]),
            exact_match_rate=_mean([1.0 if qr.exact_match else 0.0 for qr in items]),
            judge_accuracy=(
                _mean([1.0 if c else 0.0 for c in judged]) if judged else None
            ),
        )
    return breakdown


def run_benchmark(
    samples: list[BenchmarkSample],
    config: EvalConfig,
    answer_fn: AnswerFn,
    judge_fn: JudgeFn | None = None,
    max_samples: int | None = None,
    benchmark: str = "",
    ingest_fn: IngestFn = ingest_sample,
    verbose: bool = False,
) -> BenchmarkRunResult:
    """Run the QA-probe protocol over benchmark samples.

    Each sample gets its own scratch memory store (created under a temporary
    directory that is removed when the run finishes), so haystacks never leak
    between samples.

    Args:
        samples: Parsed benchmark samples (see
            :func:`memory_condense.loader.load_benchmark`).
        config: Chunker/retrieval settings for this run.
        answer_fn: Injected QA callable, ``messages -> answer text``.
        judge_fn: Optional semantic-equivalence judge; when given, judge
            accuracy is reported alongside F1/EM.
        max_samples: Evaluate at most this many samples.
        benchmark: Free-form label recorded in the result (e.g.
            ``"longmemeval_s"``).
        ingest_fn: Override how a sample's memory store is built. Defaults to
            :func:`ingest_sample`; tests inject a fake to stay offline.
        verbose: Print per-sample progress.
    """
    sample_results: list[BenchmarkSampleResult] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, sample in enumerate(samples):
            if max_samples is not None and i >= max_samples:
                break

            if verbose:
                print(
                    f"  [{i + 1}] {sample.sample_id} "
                    f"({len(sample.turns)} turns, "
                    f"{len(sample.questions)} questions)..."
                )

            result = evaluate_sample(
                sample=sample,
                config=config,
                answer_fn=answer_fn,
                data_dir=Path(tmpdir) / f"sample_{i}",
                judge_fn=judge_fn,
                ingest_fn=ingest_fn,
            )
            sample_results.append(result)

            if verbose:
                print(
                    f"       F1: {result.mean_f1:.3f}  "
                    f"EM: {result.exact_match_rate:.1%}"
                )

    all_questions = [
        qr for sr in sample_results for qr in sr.question_results
    ]
    judged = [qr.judge_correct for qr in all_questions if qr.judge_correct is not None]

    return BenchmarkRunResult(
        config=config,
        benchmark=benchmark,
        samples=sample_results,
        num_samples=len(sample_results),
        num_questions=len(all_questions),
        mean_f1=_mean([qr.f1 for qr in all_questions]),
        exact_match_rate=_mean(
            [1.0 if qr.exact_match else 0.0 for qr in all_questions]
        ),
        judge_accuracy=_mean([1.0 if c else 0.0 for c in judged]) if judged else None,
        by_category=_category_breakdown(all_questions),
        run_timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def save_benchmark_report(
    result: BenchmarkRunResult, output_dir: str | Path
) -> Path:
    """Save a benchmark run as JSON. Mirrors ``report.save_run_result``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    c = result.config.chunker
    r = result.config.retrieval
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", result.benchmark) or "benchmark"
    # The mode is in the name because the two arms of the decision run are
    # otherwise distinguishable only by timestamp — and picking the wrong file
    # out of eval_results/ inverts the result you report.
    filename = (
        f"benchmark_{name}_{c.min_tokens}-{c.max_tokens}"
        f"_k{r.k}_ef{r.ef_search}_{r.label}_{timestamp}.json"
    )
    path = output_dir / filename

    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def print_benchmark_summary(result: BenchmarkRunResult) -> None:
    """Print an aggregate + per-category summary of a benchmark run."""
    c = result.config.chunker
    r = result.config.retrieval
    print(f"\n{'=' * 64}")
    print(f"Benchmark: {result.benchmark or '(unnamed)'}")
    print(f"Config: chunk({c.min_tokens}-{c.max_tokens}) k={r.k} ef={r.ef_search}")
    print(f"Samples: {result.num_samples}  Questions: {result.num_questions}")
    print(f"Mean F1:      {result.mean_f1:.3f}")
    print(f"Exact match:  {result.exact_match_rate:.1%}")
    if result.judge_accuracy is not None:
        print(f"Judge acc:    {result.judge_accuracy:.1%}")
    print(f"{'=' * 64}")

    if result.by_category:
        print(f"{'category':<28} {'n':>5} {'F1':>7} {'EM':>8} {'Judge':>8}")
        print(f"{'-' * 64}")
        for cat in result.by_category.values():
            judge = (
                f"{cat.judge_accuracy:>7.1%}"
                if cat.judge_accuracy is not None
                else f"{'-':>8}"
            )
            print(
                f"{cat.category:<28} {cat.num_questions:>5} "
                f"{cat.mean_f1:>7.3f} {cat.exact_match_rate:>7.1%} {judge}"
            )
        print(f"{'=' * 64}")
