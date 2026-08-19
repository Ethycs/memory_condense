"""Separate evidence sufficiency from retrieval and answer generation.

LongMemEval labels evidence at session/source granularity, not exact turn
granularity.  This audit therefore compares the prompt assembled by retrieval
with a *gold-source oracle* built from every turn in the labelled sources.  It
never describes that oracle as an exact evidence span.
"""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.eval.answer_value_coverage import contains_answer
from memory_condense.eval.benchmark import (
    IngestFn,
    cap_context_to_prompt_budget,
    ingest_sample,
    shared_embedding_ingest_fn,
)
from memory_condense.eval.recall_assembly import _assemble
from memory_condense.eval.schemas import EvalConfig, UsageStats
from memory_condense.ingest.loader import BenchmarkQuestion, BenchmarkSample


SUFFICIENCY_SYSTEM_PROMPT = (
    "You audit whether conversation excerpts contain enough evidence to derive "
    "a known answer. Judge the evidence, not the wording of a prediction. "
    "Arithmetic, date differences, comparison, and combining multiple excerpts "
    "are allowed; outside knowledge and unsupported assumptions are not."
)

SUFFICIENCY_USER_TEMPLATE = (
    "Question: {question}\n"
    "Known gold answer: {gold}\n\n"
    "Conversation excerpts:\n{context}\n\n"
    "Do these excerpts contain enough information to derive the gold answer? "
    "Reply SUFFICIENT or INSUFFICIENT, then one short reason."
)


SufficiencyFn = Callable[
    [str, str, list[str]],
    tuple[bool, str] | tuple[bool, str, UsageStats],
]


class SufficiencyQuestion(BaseModel):
    question_id: str
    category: str = ""
    evidence_granularity: str = "source"
    expected_source_ids: list[str] = Field(default_factory=list)
    retrieved_source_ids: list[str] = Field(default_factory=list)
    evidence_source_recall: float | None = None
    all_evidence_sources: bool | None = None
    haystack_literal: bool = False
    gold_source_literal: bool = False
    retrieved_literal: bool = False
    retrieved_gold_source_literal: bool = False
    #: Historical CSV name: true only means literal copying cannot produce the
    #: normalized gold answer. It may require inference, aggregation, or merely
    #: a semantic paraphrase; the report labels it accordingly.
    requires_inference: bool = False
    oracle_uncapped_tokens: int = 0
    oracle_context_tokens: int = 0
    retrieved_context_tokens: int = 0
    oracle_sufficient: bool | None = None
    retrieved_sufficient: bool | None = None
    gap: str = "unjudged"
    oracle_reason: str = ""
    retrieved_reason: str = ""
    judge_usage: UsageStats = Field(default_factory=UsageStats)


class SufficiencyReport(BaseModel):
    benchmark: str = ""
    mode: str = "dense"
    n_questions: int = 0
    n_evidence_labeled: int = 0
    haystack_literal_recall: float = 0.0
    gold_source_literal_recall: float = 0.0
    retrieved_literal_recall: float = 0.0
    retrieved_gold_source_literal_recall: float = 0.0
    inference_required_rate: float = 0.0
    mean_evidence_source_recall: float | None = None
    all_evidence_sources_rate: float | None = None
    oracle_sufficiency: float | None = None
    retrieved_sufficiency: float | None = None
    sufficiency_retention: float | None = None
    mean_oracle_context_tokens: float = 0.0
    mean_retrieved_context_tokens: float = 0.0
    judge_usage: UsageStats = Field(default_factory=UsageStats)
    questions: list[SufficiencyQuestion] = Field(default_factory=list)


def build_sufficiency_prompt(
    question: str,
    gold: str,
    excerpts: list[str],
) -> list[dict[str, str]]:
    context = "\n".join(
        f"[{index + 1}] {text}" for index, text in enumerate(excerpts)
    ) or "(no excerpts)"
    return [
        {"role": "system", "content": SUFFICIENCY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": SUFFICIENCY_USER_TEMPLATE.format(
                question=question,
                gold=gold,
                context=context,
            ),
        },
    ]


def gold_source_context(
    sample: BenchmarkSample,
    question: BenchmarkQuestion,
) -> list[str]:
    """All chronological turns in labelled sources (not exact evidence turns)."""

    expected = set(question.evidence_sources)
    if not expected or len(sample.turn_source_ids) != len(sample.turns):
        return []
    return [
        text
        for (_role, text), source_id in zip(
            sample.turns,
            sample.turn_source_ids,
            strict=True,
        )
        if source_id in expected
    ]


def _call_sufficiency(
    judge: SufficiencyFn,
    question: str,
    gold: str,
    context: list[str],
) -> tuple[bool, str, UsageStats]:
    raw = judge(question, gold, context)
    if len(raw) == 3:
        sufficient, reason, usage = raw
        return bool(sufficient), str(reason), usage
    sufficient, reason = raw
    return bool(sufficient), str(reason), UsageStats()


def _gap(
    *,
    has_labels: bool,
    oracle_sufficient: bool | None,
    retrieved_sufficient: bool | None,
) -> str:
    if not has_labels:
        return "no_evidence_labels"
    if oracle_sufficient is None or retrieved_sufficient is None:
        return "unjudged"
    if oracle_sufficient and retrieved_sufficient:
        return "sufficiency_retained"
    if oracle_sufficient:
        return "retrieval_or_packing_gap"
    if retrieved_sufficient:
        return "evidence_label_mismatch"
    return "gold_source_oracle_gap"


def audit_sample(
    sample: BenchmarkSample,
    config: EvalConfig,
    data_dir: Path,
    *,
    ingest_fn: IngestFn = ingest_sample,
    sufficiency_fn: SufficiencyFn | None = None,
) -> list[SufficiencyQuestion]:
    condenser = ingest_fn(sample, config, data_dir)
    try:
        rows: list[SufficiencyQuestion] = []
        haystack = [text for _role, text in sample.turns]
        for question in sample.questions:
            query = question.dated_question
            header, body, body_sources, _body_is_consolidation = _assemble(
                condenser,
                query,
                config,
            )
            header_count = len(header)
            retrieved = cap_context_to_prompt_budget(
                query,
                [*header, *body],
                config.max_prompt_tokens,
            )
            capped_body = retrieved[header_count:]
            capped_sources = body_sources[: len(capped_body)]
            expected = set(question.evidence_sources)
            retrieved_sources = {source for source in capped_sources if source}
            coverage = (
                len(expected & retrieved_sources) / len(expected)
                if expected
                else None
            )
            oracle_uncapped = gold_source_context(sample, question)
            oracle = cap_context_to_prompt_budget(
                query,
                oracle_uncapped,
                config.max_prompt_tokens,
            )
            retrieved_gold = [
                text
                for text, source in zip(capped_body, capped_sources, strict=True)
                if source in expected
            ]

            oracle_sufficient = None
            retrieved_sufficient = None
            oracle_reason = ""
            retrieved_reason = ""
            usage = UsageStats()
            if sufficiency_fn is not None and expected and oracle:
                oracle_sufficient, oracle_reason, oracle_usage = _call_sufficiency(
                    sufficiency_fn,
                    query,
                    question.answer,
                    oracle,
                )
                (
                    retrieved_sufficient,
                    retrieved_reason,
                    retrieved_usage,
                ) = _call_sufficiency(
                    sufficiency_fn,
                    query,
                    question.answer,
                    retrieved,
                )
                usage = oracle_usage + retrieved_usage

            gold_literal = contains_answer(oracle, question.answer)
            rows.append(
                SufficiencyQuestion(
                    question_id=question.question_id,
                    category=question.category or "",
                    expected_source_ids=list(question.evidence_sources),
                    retrieved_source_ids=list(
                        dict.fromkeys(source for source in capped_sources if source)
                    ),
                    evidence_source_recall=coverage,
                    all_evidence_sources=(
                        coverage == 1.0 if coverage is not None else None
                    ),
                    haystack_literal=contains_answer(haystack, question.answer),
                    gold_source_literal=gold_literal,
                    retrieved_literal=contains_answer(retrieved, question.answer),
                    retrieved_gold_source_literal=contains_answer(
                        retrieved_gold,
                        question.answer,
                    ),
                    requires_inference=not gold_literal,
                    oracle_uncapped_tokens=sum(
                        count_tokens(text) for text in oracle_uncapped
                    ),
                    oracle_context_tokens=sum(count_tokens(text) for text in oracle),
                    retrieved_context_tokens=sum(
                        count_tokens(text) for text in retrieved
                    ),
                    oracle_sufficient=oracle_sufficient,
                    retrieved_sufficient=retrieved_sufficient,
                    gap=_gap(
                        has_labels=bool(expected),
                        oracle_sufficient=oracle_sufficient,
                        retrieved_sufficient=retrieved_sufficient,
                    ),
                    oracle_reason=oracle_reason,
                    retrieved_reason=retrieved_reason,
                    judge_usage=usage,
                )
            )
        return rows
    finally:
        condenser.close()


def _fraction(values) -> float:
    values = list(values)
    return sum(bool(value) for value in values) / len(values) if values else 0.0


def run_sufficiency_audit(
    samples: list[BenchmarkSample],
    config: EvalConfig,
    *,
    benchmark: str = "",
    max_samples: int | None = None,
    ingest_fn: IngestFn = ingest_sample,
    sufficiency_fn: SufficiencyFn | None = None,
) -> SufficiencyReport:
    selected = samples[:max_samples] if max_samples is not None else samples
    effective_ingest = (
        shared_embedding_ingest_fn(config.embedding_device)
        if ingest_fn is ingest_sample
        else ingest_fn
    )
    rows: list[SufficiencyQuestion] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for index, sample in enumerate(selected):
            print(
                f"  [{index + 1}/{len(selected)}] {sample.sample_id} "
                f"({len(sample.turns)} turns, {len(sample.questions)} questions)..."
            )
            rows.extend(
                audit_sample(
                    sample,
                    config,
                    Path(tmpdir) / f"sample_{index}",
                    ingest_fn=effective_ingest,
                    sufficiency_fn=sufficiency_fn,
                )
            )
            if config.retrieval.mode in {"causal_consolidation", "causal_graph"}:
                gc.collect()

    labeled = [row for row in rows if row.expected_source_ids]
    coverage = [
        row.evidence_source_recall
        for row in labeled
        if row.evidence_source_recall is not None
    ]
    all_sources = [
        row.all_evidence_sources
        for row in labeled
        if row.all_evidence_sources is not None
    ]
    judged = [row for row in labeled if row.oracle_sufficient is not None]
    oracle_positive = [row for row in judged if row.oracle_sufficient]
    total_usage = UsageStats()
    for row in rows:
        total_usage = total_usage + row.judge_usage
    n = len(rows)
    return SufficiencyReport(
        benchmark=benchmark,
        mode=config.retrieval.mode,
        n_questions=n,
        n_evidence_labeled=len(labeled),
        haystack_literal_recall=_fraction(row.haystack_literal for row in rows),
        gold_source_literal_recall=_fraction(
            row.gold_source_literal for row in labeled
        ),
        retrieved_literal_recall=_fraction(
            row.retrieved_literal for row in rows
        ),
        retrieved_gold_source_literal_recall=_fraction(
            row.retrieved_gold_source_literal for row in labeled
        ),
        inference_required_rate=_fraction(row.requires_inference for row in labeled),
        mean_evidence_source_recall=(
            sum(coverage) / len(coverage) if coverage else None
        ),
        all_evidence_sources_rate=(
            _fraction(all_sources) if all_sources else None
        ),
        oracle_sufficiency=(
            _fraction(row.oracle_sufficient for row in judged) if judged else None
        ),
        retrieved_sufficiency=(
            _fraction(row.retrieved_sufficient for row in judged) if judged else None
        ),
        sufficiency_retention=(
            _fraction(row.retrieved_sufficient for row in oracle_positive)
            if oracle_positive
            else None
        ),
        mean_oracle_context_tokens=(
            sum(row.oracle_context_tokens for row in labeled) / len(labeled)
            if labeled
            else 0.0
        ),
        mean_retrieved_context_tokens=(
            sum(row.retrieved_context_tokens for row in rows) / n if n else 0.0
        ),
        judge_usage=total_usage,
        questions=rows,
    )


def print_sufficiency_report(report: SufficiencyReport) -> None:
    print()
    print("=" * 72)
    print("EVIDENCE SUFFICIENCY AUDIT")
    print(f"  benchmark: {report.benchmark or '(unnamed)'}")
    print(f"  mode     : {report.mode}")
    print(f"  questions: {report.n_questions}")
    print("  oracle   : gold source/session (not exact evidence turns)")
    print("=" * 72)
    print(f"{'literal answer in full haystack':<40}{report.haystack_literal_recall:>8.1%}")
    print(f"{'literal answer in gold sources':<40}{report.gold_source_literal_recall:>8.1%}")
    print(f"{'literal answer in retrieved context':<40}{report.retrieved_literal_recall:>8.1%}")
    print(
        f"{'literal answer in retrieved gold sources':<40}"
        f"{report.retrieved_gold_source_literal_recall:>8.1%}"
    )
    print(
        f"{'gold sources lacking a literal answer span':<40}"
        f"{report.inference_required_rate:>8.1%}"
    )
    if report.mean_evidence_source_recall is not None:
        print(f"{'mean evidence-source coverage':<40}{report.mean_evidence_source_recall:>8.1%}")
        print(f"{'questions with every evidence source':<40}{report.all_evidence_sources_rate:>8.1%}")
    print(f"{'mean gold-source oracle tokens':<40}{report.mean_oracle_context_tokens:>8.0f}")
    print(f"{'mean retrieved context tokens':<40}{report.mean_retrieved_context_tokens:>8.0f}")
    if report.oracle_sufficiency is not None:
        print(f"{'judged gold-source sufficiency':<40}{report.oracle_sufficiency:>8.1%}")
        print(f"{'judged retrieved sufficiency':<40}{report.retrieved_sufficiency:>8.1%}")
        print(f"{'retention when oracle is sufficient':<40}{report.sufficiency_retention:>8.1%}")
        print(f"{'judge calls':<40}{report.judge_usage.calls:>8}")
    else:
        print("  Semantic sufficiency unjudged; deterministic diagnostics only.")
    print()
