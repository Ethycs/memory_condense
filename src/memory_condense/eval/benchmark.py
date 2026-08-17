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

import math
import re
import string
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Protocol

from pydantic import BaseModel, Field

from memory_condense._tokenizer import count_tokens, truncate_to_tokens
from memory_condense.condenser import MemoryCondenser
from memory_condense.context_packer import ContextBudget
from memory_condense.embedding import EmbeddingService
from memory_condense.eval.reproducibility import file_sha256
from memory_condense.eval.schemas import EvalConfig, UsageStats
from memory_condense.loader import BenchmarkQuestion, BenchmarkSample

# ---------------------------------------------------------------------------
# Injected callable types
# ---------------------------------------------------------------------------

#: ``answer_fn(messages) -> answer_text``. ``messages`` is an OpenAI-style
#: chat list, so a litellm-backed implementation is a one-liner.
AnswerFn = Callable[
    [list[dict[str, str]]],
    str | tuple[str, UsageStats],
]

#: ``judge_fn(question, gold_answer, predicted_answer) -> (is_correct, reason)``
JudgeFn = Callable[
    [str, str, str],
    tuple[bool, str] | tuple[bool, str, UsageStats],
]


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
    context_tokens: int = 0
    prompt_tokens: int = 0
    transcript_tokens: int = 0
    context_fraction: float = 0.0
    transcript_token_savings: float = 0.0
    responder_usage: UsageStats = Field(default_factory=UsageStats)
    judge_usage: UsageStats = Field(default_factory=UsageStats)


class BenchmarkSampleResult(BaseModel):
    """Aggregated results for all questions on one benchmark sample."""

    sample_id: str
    num_turns: int
    num_questions: int
    question_results: list[BenchmarkQuestionResult] = Field(default_factory=list)
    mean_f1: float = 0.0
    exact_match_rate: float = 0.0
    judge_accuracy: Optional[float] = None
    mean_context_tokens: float = 0.0
    mean_prompt_tokens: float = 0.0
    transcript_tokens: int = 0
    mean_context_fraction: float = 0.0
    mean_transcript_token_savings: float = 0.0


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
    mean_context_tokens: float = 0.0
    mean_prompt_tokens: float = 0.0
    p95_prompt_tokens: int = 0
    mean_transcript_tokens: float = 0.0
    mean_context_fraction: float = 0.0
    mean_transcript_token_savings: float = 0.0
    max_prompt_tokens_observed: int = 0
    prompt_budget_compliance: bool = True
    accuracy_target: float = 0.95
    min_target_questions: int = 100
    accuracy_target_met: Optional[bool] = None
    target_status: str = "ungraded"
    responder_usage: UsageStats = Field(default_factory=UsageStats)
    judge_usage: UsageStats = Field(default_factory=UsageStats)
    dataset_sha256: str = ""
    split_manifest_sha256: str = ""
    benchmark_split: str = ""
    implementation_sha256: str = ""
    environment_lock_sha256: str = ""
    policy_manifest_sha256: str = ""
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
    *,
    embedder: EmbeddingService | None = None,
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
        embedder=embedder,
    )
    try:
        source_ids = sample.turn_source_ids or [None] * len(sample.turns)
        if len(source_ids) != len(sample.turns):
            raise ValueError("turn_source_ids must be empty or parallel to turns")
        records = [
            (role, text, source_id)
            for (role, text), source_id in zip(sample.turns, source_ids, strict=True)
            if text
        ]
        mc.ingest_many(records)
        return mc
    except BaseException:
        # A partially ingested store still owns SQLite/WAL and HNSW handles.
        # Close it before the compiled-cache layer tries to remove its
        # temporary directory; Windows otherwise raises a second PermissionError
        # that masks the actual corpus/ingestion failure.
        try:
            mc.close()
        except Exception:
            pass
        raise


def shared_embedding_ingest_fn(device: str | None = None) -> IngestFn:
    """Create isolated stores while reusing one stateless embedding model.

    LongMemEval oracle has one question per sample. Constructing an embedder
    inside every store reloads bge-m3 hundreds of times and dominates the
    evaluation. SQLite/HNSW state remains isolated; only model weights and the
    tokenizer are shared across sequential samples.
    """

    embedder = EmbeddingService(device=device)

    def ingest(
        sample: BenchmarkSample,
        config: EvalConfig,
        data_dir: Path,
    ) -> MemoryCondenser:
        return ingest_sample(sample, config, data_dir, embedder=embedder)

    return ingest


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


def _message_content_tokens(messages: list[dict[str, str]]) -> int:
    """Stable tokenizer-proxy count over content sent to a provider."""

    return sum(count_tokens(message.get("content", "")) for message in messages)


def cap_context_to_prompt_budget(
    question: str,
    chunk_texts: list[str],
    max_prompt_tokens: int | None,
) -> list[str]:
    """Keep ranked excerpts in order under a hard prompt-content ceiling.

    The final excerpt may be truncated on a token boundary. A binary search is
    used because numbered labels and the QA template also consume tokens; the
    assembled prompt is measured rather than estimated.
    """

    if max_prompt_tokens is None:
        return list(chunk_texts)
    if max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must be positive")

    if _message_content_tokens(build_qa_prompt(question, [])) > max_prompt_tokens:
        raise ValueError(
            "max_prompt_tokens is smaller than the QA prompt without context"
        )

    selected: list[str] = []
    for excerpt in chunk_texts:
        proposal = [*selected, excerpt]
        if (
            _message_content_tokens(build_qa_prompt(question, proposal))
            <= max_prompt_tokens
        ):
            selected.append(excerpt)
            continue

        low = 0
        high = count_tokens(excerpt)
        while low < high:
            midpoint = (low + high + 1) // 2
            prefix = truncate_to_tokens(excerpt, midpoint)
            tokens = _message_content_tokens(
                build_qa_prompt(question, [*selected, prefix])
            )
            if tokens <= max_prompt_tokens:
                low = midpoint
            else:
                high = midpoint - 1
        if low:
            selected.append(truncate_to_tokens(excerpt, low))
        break
    return selected


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
) -> tuple[str, list[str], int, UsageStats]:
    """Assemble context for ``question`` and answer from it.

    In ``dense``/``hybrid`` mode the context is the top-k chunks. In ``memory``
    mode it is what ``build_context`` produces — the memory-item header plus
    budgeted verbatim expansions — which is the only way this harness exercises
    ``ContextPacker``, ``MemoryStore.retrieve``, ``rank_score`` or ``decay``.

    Returns answer text, context, prompt-content tokens, and provider usage.
    """
    query_text = question.dated_question
    if config.retrieval.mode in {
        "memory",
        "causal_consolidation",
        "causal_graph",
    }:
        causal = config.retrieval.mode in {
            "causal_consolidation",
            "causal_graph",
        }
        graph_results = (
            mc.search_hybrid_graph(
                query_text,
                k=config.retrieval.k,
                neighbor_radius=config.retrieval.neighbor_radius,
                neighbor_slots=config.retrieval.neighbor_slots,
                neighbor_direction=config.retrieval.neighbor_direction,
                source_slots=config.retrieval.source_slots,
                source_candidate_pool=config.retrieval.source_candidate_pool,
                source_activation_k=config.retrieval.source_activation_k,
                source_tfisf_activation=(
                    config.retrieval.source_tfisf_activation
                ),
                source_tfisf_slots=config.retrieval.source_tfisf_slots,
                source_hsc_activation=config.retrieval.source_hsc_activation,
                source_hsc_slots=config.retrieval.source_hsc_slots,
                source_hsc_hops=config.retrieval.source_hsc_hops,
                source_hsc_chunk_slots=(
                    config.retrieval.source_hsc_chunk_slots
                ),
                source_partition_routing=(
                    config.retrieval.source_partition_routing
                ),
                source_partition_slots=config.retrieval.source_partition_slots,
                source_partition_separator=(
                    config.retrieval.source_partition_separator
                ),
                source_local_search=config.retrieval.source_local_search,
                use_source_reranker=config.retrieval.qwen_rerank,
                use_attention_feedback=config.retrieval.qwen_feedback,
                feedback_slots=config.retrieval.qwen_feedback_slots,
                feedback_seed_slots=config.retrieval.qwen_feedback_seed_slots,
                feedback_evidence_tokens=(
                    config.retrieval.qwen_feedback_evidence_tokens
                ),
                feedback_query_tokens=config.retrieval.qwen_feedback_query_tokens,
                ef_search=config.retrieval.ef_search,
                candidates=config.retrieval.candidates,
                alpha=config.retrieval.alpha,
            )
            if config.retrieval.mode == "causal_graph"
            else None
        )
        packed = mc.build_context(
            query_text,
            # A haystack is not a live conversation: its "last 8 turns" are an
            # arbitrary slice of someone else's dialogue and would be noise in
            # the prompt. The QA protocol wants retrieved context only.
            recent_turns=0,
            k_memories=0 if causal else config.retrieval.k_memories,
            k_expansions=(0 if graph_results is not None else config.retrieval.k),
            hybrid=True,
            # Benchmark questions are independent probes.  Letting one probe
            # reheat its hits changes later rankings and makes accuracy depend
            # on question order rather than the stored conversation.
            reheat_memories=False,
            # Independent benchmark probes must not teach one another a live
            # co-activation graph either.
            use_consolidation=causal,
            learn_consolidation=False,
            consolidation_memory_slots=0 if causal else 1,
            consolidation_chunk_slots=(
                config.retrieval.consolidation_chunk_slots if causal else 1
            ),
            consolidation_min_count=config.retrieval.consolidation_min_count,
            consolidation_hops=config.retrieval.consolidation_hops,
            consolidation_candidates=config.retrieval.consolidation_candidates,
            consolidation_diffusion_width=(
                config.retrieval.consolidation_diffusion_width
            ),
            expansion_results=graph_results,
        )
        # `expansions` is already rendered text, unlike the chunk arms'
        # RetrievalResult objects.
        context_texts = [t for t in [packed.memory_header] if t]
        context_texts += list(packed.expansions)
    elif config.retrieval.mode == "span":
        context_texts = [
            r.chunk.text
            for r in mc.search_spans(
                query_text,
                levels=config.retrieval.span_levels,
                k_per_level=config.retrieval.k_per_level,
            )
        ]
    elif config.retrieval.mode == "source":
        context_texts = [
            result.chunk.text
            for result in mc.search_sources(
                query_text,
                k_sources=config.retrieval.k_sources,
            )
        ]
    elif config.retrieval.mode == "anchored_source":
        context_texts = [
            result.chunk.text
            for result in mc.search_anchored_sources(
                query_text,
                k=config.retrieval.k,
                ef_search=config.retrieval.ef_search,
                candidates=config.retrieval.candidates,
                alpha=config.retrieval.alpha,
            )
        ]
    elif config.retrieval.mode == "hybrid_source":
        context_texts = [
            result.chunk.text
            for result in mc.search_hybrid_sources(
                query_text,
                k=config.retrieval.k,
                source_slots=config.retrieval.source_slots,
                source_candidate_pool=config.retrieval.source_candidate_pool,
                source_activation_k=config.retrieval.source_activation_k,
                source_partition_routing=(
                    config.retrieval.source_partition_routing
                ),
                source_partition_slots=config.retrieval.source_partition_slots,
                source_partition_separator=(
                    config.retrieval.source_partition_separator
                ),
                source_local_search=config.retrieval.source_local_search,
                use_source_reranker=config.retrieval.qwen_rerank,
                ef_search=config.retrieval.ef_search,
                candidates=config.retrieval.candidates,
                alpha=config.retrieval.alpha,
            )
        ]
    elif config.retrieval.mode == "hybrid_graph":
        context_texts = [
            result.chunk.text
            for result in mc.search_hybrid_graph(
                query_text,
                k=config.retrieval.k,
                neighbor_radius=config.retrieval.neighbor_radius,
                neighbor_slots=config.retrieval.neighbor_slots,
                neighbor_direction=config.retrieval.neighbor_direction,
                source_slots=config.retrieval.source_slots,
                source_candidate_pool=config.retrieval.source_candidate_pool,
                source_activation_k=config.retrieval.source_activation_k,
                source_tfisf_activation=(
                    config.retrieval.source_tfisf_activation
                ),
                source_tfisf_slots=config.retrieval.source_tfisf_slots,
                source_hsc_activation=config.retrieval.source_hsc_activation,
                source_hsc_slots=config.retrieval.source_hsc_slots,
                source_hsc_hops=config.retrieval.source_hsc_hops,
                source_hsc_chunk_slots=(
                    config.retrieval.source_hsc_chunk_slots
                ),
                source_local_search=config.retrieval.source_local_search,
                use_source_reranker=config.retrieval.qwen_rerank,
                use_attention_feedback=config.retrieval.qwen_feedback,
                feedback_slots=config.retrieval.qwen_feedback_slots,
                feedback_seed_slots=config.retrieval.qwen_feedback_seed_slots,
                feedback_evidence_tokens=(
                    config.retrieval.qwen_feedback_evidence_tokens
                ),
                feedback_query_tokens=config.retrieval.qwen_feedback_query_tokens,
                ef_search=config.retrieval.ef_search,
                candidates=config.retrieval.candidates,
                alpha=config.retrieval.alpha,
            )
        ]
    elif config.retrieval.mode == "hybrid_neighbor":
        context_texts = [
            result.chunk.text
            for result in mc.search_hybrid_neighbors(
                query_text,
                k=config.retrieval.k,
                radius=config.retrieval.neighbor_radius,
                max_neighbors=config.retrieval.neighbor_slots,
                replacement_slots=config.retrieval.neighbor_replacement_slots,
                ef_search=config.retrieval.ef_search,
                candidates=config.retrieval.candidates,
                alpha=config.retrieval.alpha,
            )
        ]
    elif config.retrieval.effective_hybrid:
        retrieved = mc.search_hybrid(
            query_text,
            k=config.retrieval.k,
            ef_search=config.retrieval.ef_search,
            candidates=config.retrieval.candidates,
            alpha=config.retrieval.alpha,
        )
        context_texts = [r.chunk.text for r in retrieved]
    else:
        retrieved = mc.search(
            query_text,
            k=config.retrieval.k,
            ef_search=config.retrieval.ef_search,
        )
        context_texts = [r.chunk.text for r in retrieved]

    context_texts = cap_context_to_prompt_budget(
        query_text,
        context_texts,
        config.max_prompt_tokens,
    )
    messages = build_qa_prompt(query_text, context_texts)
    prompt_tokens = _message_content_tokens(messages)
    raw_answer = answer_fn(messages)
    if isinstance(raw_answer, tuple):
        answer, usage = raw_answer
    else:
        answer, usage = raw_answer, UsageStats()

    return (answer or "").strip(), context_texts, prompt_tokens, usage


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _context_fraction(context_tokens: int, transcript_tokens: int) -> float:
    """Fraction of completed-transcript content sent as retrieved context."""
    if transcript_tokens <= 0:
        return 0.0
    return context_tokens / transcript_tokens


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
    transcript_tokens = sum(count_tokens(text) for _role, text in sample.turns)

    mc = ingest_fn(sample, config, Path(data_dir))
    try:
        for question in sample.questions:
            prediction, chunk_texts, prompt_tokens, responder_usage = answer_question(
                mc, question, config, answer_fn
            )

            judge_correct: bool | None = None
            judge_reasoning: str | None = None
            judge_usage = UsageStats()
            if judge_fn is not None:
                judge_result = judge_fn(
                    question.question, question.answer, prediction
                )
                if len(judge_result) == 3:
                    judge_correct, judge_reasoning, judge_usage = judge_result
                else:
                    judge_correct, judge_reasoning = judge_result

            context_tokens = sum(count_tokens(text) for text in chunk_texts)
            context_fraction = _context_fraction(context_tokens, transcript_tokens)
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
                    context_tokens=context_tokens,
                    prompt_tokens=prompt_tokens,
                    transcript_tokens=transcript_tokens,
                    context_fraction=context_fraction,
                    transcript_token_savings=1.0 - context_fraction,
                    responder_usage=responder_usage,
                    judge_usage=judge_usage,
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
        mean_context_tokens=_mean(
            [float(qr.context_tokens) for qr in question_results]
        ),
        mean_prompt_tokens=_mean(
            [float(qr.prompt_tokens) for qr in question_results]
        ),
        transcript_tokens=transcript_tokens,
        mean_context_fraction=_mean(
            [qr.context_fraction for qr in question_results]
        ),
        mean_transcript_token_savings=_mean(
            [qr.transcript_token_savings for qr in question_results]
        ),
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
    dataset_sha256: str = "",
    split_manifest_sha256: str = "",
    benchmark_split: str = "",
    implementation_sha256: str = "",
    environment_lock_sha256: str = "",
    policy_manifest_sha256: str = "",
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
    effective_ingest_fn = (
        shared_embedding_ingest_fn(config.embedding_device)
        if ingest_fn is ingest_sample
        else ingest_fn
    )

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
                ingest_fn=effective_ingest_fn,
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
    judge_accuracy = _mean([1.0 if c else 0.0 for c in judged]) if judged else None
    prompt_budget_compliance = (
        config.max_prompt_tokens is None
        or all(
            qr.prompt_tokens <= config.max_prompt_tokens
            for qr in all_questions
        )
    )
    if judge_accuracy is None:
        target_status = "ungraded"
        target_met = None
    elif len(judged) < config.min_target_questions:
        target_status = "insufficient_questions"
        target_met = False
    elif not prompt_budget_compliance:
        target_status = "prompt_budget_exceeded"
        target_met = False
    elif judge_accuracy >= config.accuracy_target:
        target_status = "passed"
        target_met = True
    else:
        target_status = "failed"
        target_met = False
    prompt_counts = sorted(qr.prompt_tokens for qr in all_questions)
    p95_index = max(0, math.ceil(0.95 * len(prompt_counts)) - 1)

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
        judge_accuracy=judge_accuracy,
        mean_context_tokens=_mean(
            [float(qr.context_tokens) for qr in all_questions]
        ),
        mean_prompt_tokens=_mean([float(qr.prompt_tokens) for qr in all_questions]),
        p95_prompt_tokens=(prompt_counts[p95_index] if prompt_counts else 0),
        mean_transcript_tokens=_mean(
            [float(qr.transcript_tokens) for qr in all_questions]
        ),
        mean_context_fraction=_mean(
            [qr.context_fraction for qr in all_questions]
        ),
        mean_transcript_token_savings=_mean(
            [qr.transcript_token_savings for qr in all_questions]
        ),
        max_prompt_tokens_observed=max(prompt_counts, default=0),
        prompt_budget_compliance=prompt_budget_compliance,
        accuracy_target=config.accuracy_target,
        min_target_questions=config.min_target_questions,
        accuracy_target_met=target_met,
        target_status=target_status,
        responder_usage=sum(
            (question.responder_usage for question in all_questions),
            UsageStats(),
        ),
        judge_usage=sum(
            (question.judge_usage for question in all_questions),
            UsageStats(),
        ),
        dataset_sha256=dataset_sha256,
        split_manifest_sha256=split_manifest_sha256,
        benchmark_split=benchmark_split,
        implementation_sha256=implementation_sha256,
        environment_lock_sha256=environment_lock_sha256,
        policy_manifest_sha256=policy_manifest_sha256,
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
    (path.parent / f"{path.name}.sha256").write_text(
        f"{file_sha256(path)}  {path.name}\n",
        encoding="ascii",
    )
    return path


def print_benchmark_summary(result: BenchmarkRunResult) -> None:
    """Print an aggregate + per-category summary of a benchmark run."""
    c = result.config.chunker
    r = result.config.retrieval
    print(f"\n{'=' * 64}")
    print(f"Benchmark: {result.benchmark or '(unnamed)'}")
    print(f"Config: chunk({c.min_tokens}-{c.max_tokens}) k={r.k} ef={r.ef_search}")
    print(f"Samples: {result.num_samples}  Questions: {result.num_questions}")
    if result.dataset_sha256:
        print(
            f"Evidence: data {result.dataset_sha256[:12]}...  "
            f"code {result.implementation_sha256[:12]}...  "
            f"env {result.environment_lock_sha256[:12]}..."
        )
    print(f"Mean F1:      {result.mean_f1:.3f}")
    print(f"Exact match:  {result.exact_match_rate:.1%}")
    if result.judge_accuracy is not None:
        print(f"Operational answer accuracy: {result.judge_accuracy:.1%}")
    print(
        "Transcript -> retrieved context: "
        f"{result.mean_transcript_tokens:.1f} -> "
        f"{result.mean_context_tokens:.1f} mean tokens "
        f"({result.mean_transcript_token_savings:.1%} saved)"
    )
    print(
        f"Prompt tokens: mean {result.mean_prompt_tokens:.1f}, "
        f"p95 {result.p95_prompt_tokens}, "
        f"max {result.max_prompt_tokens_observed} "
        f"({'within budget' if result.prompt_budget_compliance else 'OVER BUDGET'})"
    )
    print(
        "Provider usage: "
        f"answer {result.responder_usage.calls} calls / "
        f"{result.responder_usage.input_tokens} in / "
        f"{result.responder_usage.output_tokens} out; "
        f"judge {result.judge_usage.calls} calls / "
        f"{result.judge_usage.input_tokens} in / "
        f"{result.judge_usage.output_tokens} out"
    )
    print(
        f"Target:       {result.accuracy_target:.1%} "
        f"({result.target_status}; minimum {result.min_target_questions} questions)"
    )
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
