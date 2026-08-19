"""Portable candidate traces for fast, leakage-auditable transition sweeps."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from memory_condense._tokenizer import count_tokens
from memory_condense.embedding import EmbeddingService
from memory_condense.eval.benchmark import (
    QA_SYSTEM_PROMPT,
    QA_USER_TEMPLATE,
    IngestFn,
    cap_context_to_prompt_budget,
    f1_score,
)
from memory_condense.eval.recall import best_f1, contains_answer
from memory_condense.eval.schemas import EvalConfig
from memory_condense.loader import BenchmarkSample


TRACE_FORMAT = "memory-condense-transition-candidate-trace-v1"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TransitionTraceCandidate(BaseModel):
    chunk_id: str
    turn_id: str
    source_id: str | None = None
    text: str
    token_count: int = Field(ge=0)
    route: Literal["hybrid_anchor", "source_neighbor"]
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    anchor_chunk_id: str
    anchor_rank: int = Field(ge=1)
    transition_distance: int | None = Field(default=None, ge=1)
    transition_direction: Literal["previous", "next"] | None = None

    model_config = {"frozen": True}


class TransitionTraceQuestion(BaseModel):
    sample_id: str
    question_id: str
    category: str = ""
    question: str
    dated_question: str
    answer: str
    evidence_sources: list[str] = Field(default_factory=list)
    answer_in_haystack: bool
    candidates: list[TransitionTraceCandidate]

    model_config = {"frozen": True}


class TransitionTracePack(BaseModel):
    format: str = TRACE_FORMAT
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: str
    embedding_model: str
    embedding_dim: int
    chunker_min_tokens: int
    chunker_max_tokens: int
    hybrid_k: int
    hybrid_alpha: float
    hybrid_candidates: int
    max_radius: int
    questions: list[TransitionTraceQuestion]
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    model_config = {"frozen": True}

    def verified(self) -> bool:
        payload = self.model_dump(mode="json", exclude={"trace_sha256"})
        return _canonical_sha256(payload) == self.trace_sha256


class TransitionArm(BaseModel):
    """A deterministic action over a frozen candidate trace."""

    name: str
    retain_anchors: int = Field(ge=0)
    neighbor_slots: int = Field(ge=0)
    max_distance: int = Field(default=1, ge=1)
    direction: Literal["both", "previous", "next"] = "both"

    model_config = {"frozen": True}


class TransitionArmScore(BaseModel):
    arm: TransitionArm
    questions: int
    literal_recall: float
    mean_best_f1: float
    mean_context_tokens: float
    recall_per_1k_tokens: float
    evidence_source_recall: float | None = None
    evidence_all_source_recall: float | None = None
    gained_vs_stay: int = 0
    lost_vs_stay: int = 0

    model_config = {"frozen": True}


def build_transition_trace(
    samples: list[BenchmarkSample],
    config: EvalConfig,
    *,
    ingest_fn: IngestFn,
    embedder: EmbeddingService,
    dataset_sha256: str,
    split_manifest_sha256: str,
    split: str,
    max_radius: int = 6,
) -> TransitionTracePack:
    """Batch query encoding, then capture direct and source-local candidates."""
    if config.retrieval.mode == "memory":
        raise ValueError("transition traces require a non-extracting retrieval mode")
    if max_radius < 1:
        raise ValueError("max_radius must be positive")

    jobs = [
        (sample, question)
        for sample in samples
        for question in sample.questions
    ]
    embeddings = embedder.embed_queries(
        [question.dated_question for _sample, question in jobs]
    )
    embedding_by_question = {
        (sample.sample_id, question.question_id): embedding
        for (sample, question), embedding in zip(jobs, embeddings, strict=True)
    }

    rows: list[TransitionTraceQuestion] = []
    for sample_index, sample in enumerate(samples):
        mc = ingest_fn(sample, config, Path(f"unused-{sample_index}"))
        try:
            haystack_texts = [text for _role, text in sample.turns]
            for question in sample.questions:
                embedding = embedding_by_question[
                    (sample.sample_id, question.question_id)
                ]
                anchors = mc.search_hybrid_from_embedding(
                    question.dated_question,
                    embedding,
                    k=config.retrieval.k,
                    ef_search=config.retrieval.ef_search,
                    candidates=config.retrieval.candidates,
                    alpha=config.retrieval.alpha,
                )
                expanded = mc.expand_source_neighbors(
                    anchors,
                    radius=max_radius,
                )
                anchor_ranks = {
                    result.chunk.chunk_id: rank
                    for rank, result in enumerate(anchors, start=1)
                }
                candidates: list[TransitionTraceCandidate] = []
                for index, result in enumerate(expanded):
                    is_anchor = index < len(anchors)
                    anchor_id = (
                        result.chunk.chunk_id
                        if is_anchor
                        else result.anchor_chunk_id
                    )
                    if anchor_id is None or anchor_id not in anchor_ranks:
                        raise ValueError("neighbor candidate lost its anchor identity")
                    candidates.append(
                        TransitionTraceCandidate(
                            chunk_id=result.chunk.chunk_id,
                            turn_id=result.chunk.turn_id,
                            source_id=(
                                result.turn.source_id if result.turn else None
                            ),
                            text=result.chunk.text,
                            token_count=result.chunk.token_count,
                            route=(
                                "hybrid_anchor" if is_anchor else "source_neighbor"
                            ),
                            score=float(result.score),
                            dense_score=result.dense_score,
                            lexical_score=result.lexical_score,
                            anchor_chunk_id=anchor_id,
                            anchor_rank=anchor_ranks[anchor_id],
                            transition_distance=result.transition_distance,
                            transition_direction=result.transition_direction,
                        )
                    )
                rows.append(
                    TransitionTraceQuestion(
                        sample_id=sample.sample_id,
                        question_id=question.question_id,
                        category=question.category or "",
                        question=question.question,
                        dated_question=question.dated_question,
                        answer=question.answer,
                        evidence_sources=question.evidence_sources,
                        answer_in_haystack=contains_answer(
                            haystack_texts, question.answer
                        ),
                        candidates=candidates,
                    )
                )
        finally:
            mc.close()

    model_name = str(getattr(embedder, "model_name", type(embedder).__qualname__))
    payload = {
        "format": TRACE_FORMAT,
        "dataset_sha256": dataset_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "split": split,
        "embedding_model": model_name,
        "embedding_dim": int(embedder.dim),
        "chunker_min_tokens": config.chunker.min_tokens,
        "chunker_max_tokens": config.chunker.max_tokens,
        "hybrid_k": config.retrieval.k,
        "hybrid_alpha": config.retrieval.alpha,
        "hybrid_candidates": config.retrieval.candidates,
        "max_radius": max_radius,
        "questions": [row.model_dump(mode="json") for row in rows],
    }
    return TransitionTracePack(
        **payload,
        trace_sha256=_canonical_sha256(payload),
    )


def save_transition_trace(pack: TransitionTracePack, path: str | Path) -> Path:
    if not pack.verified():
        raise ValueError("refusing to save an invalid transition trace")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(pack.model_dump_json(indent=2), encoding="utf-8")
    return output


def load_transition_trace(path: str | Path) -> TransitionTracePack:
    pack = TransitionTracePack.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )
    if not pack.verified():
        raise ValueError("transition trace SHA-256 mismatch")
    return pack


def compose_transition_context(
    question: TransitionTraceQuestion,
    arm: TransitionArm,
    *,
    max_prompt_tokens: int | None,
) -> tuple[list[str], list[str | None]]:
    selected = _select_transition_candidates(question, arm)
    texts = cap_context_to_prompt_budget(
        question.dated_question,
        [candidate.text for candidate in selected],
        max_prompt_tokens,
    )
    return texts, [candidate.source_id for candidate in selected[: len(texts)]]


def _select_transition_candidates(
    question: TransitionTraceQuestion,
    arm: TransitionArm,
) -> list[TransitionTraceCandidate]:
    anchors = [
        candidate
        for candidate in question.candidates
        if candidate.route == "hybrid_anchor"
    ][: arm.retain_anchors]
    neighbors = [
        candidate
        for candidate in question.candidates
        if candidate.route == "source_neighbor"
        and (candidate.transition_distance or 0) <= arm.max_distance
        and (
            arm.direction == "both"
            or candidate.transition_direction == arm.direction
        )
    ][: arm.neighbor_slots]
    return anchors + neighbors


@lru_cache(maxsize=100_000)
def _candidate_contains(text: str, answer: str) -> bool:
    return contains_answer([text], answer)


@lru_cache(maxsize=100_000)
def _candidate_f1(text: str, answer: str) -> float:
    return f1_score(text, answer)


@lru_cache(maxsize=100_000)
def _rendered_token_count(text: str) -> int:
    # Chunk.token_count is a write-time segmentation estimate. Prompt
    # accounting measures the actual rendered text, whose BPE boundaries can
    # differ after sentence joins.
    return count_tokens(text)


@lru_cache(maxsize=10_000)
def _prompt_overhead_upper_bound(dated_question: str) -> int:
    # The actual context adds one numbered label and newline per candidate.
    # Twelve tokens per item is deliberately loose, so falling below this
    # bound proves the exact prompt cannot reach the hard cap.
    empty_user = QA_USER_TEMPLATE.format(context="", question=dated_question)
    return count_tokens(QA_SYSTEM_PROMPT) + count_tokens(empty_user)


def score_transition_arm(
    pack: TransitionTracePack,
    arm: TransitionArm,
    *,
    max_prompt_tokens: int | None = 8000,
    stay_hits: dict[str, bool] | None = None,
) -> tuple[TransitionArmScore, dict[str, bool]]:
    hits: dict[str, bool] = {}
    f1_values: list[float] = []
    token_counts: list[int] = []
    source_coverage: list[float] = []
    source_all: list[bool] = []
    for question in pack.questions:
        selected = _select_transition_candidates(question, arm)
        raw_tokens = sum(_rendered_token_count(candidate.text) for candidate in selected)
        safely_below_cap = (
            max_prompt_tokens is None
            or raw_tokens
            + _prompt_overhead_upper_bound(question.dated_question)
            + 12 * len(selected)
            <= max_prompt_tokens
        )
        if safely_below_cap:
            texts = [candidate.text for candidate in selected]
            sources = [candidate.source_id for candidate in selected]
            context_tokens = raw_tokens
            hit = any(
                _candidate_contains(candidate.text, question.answer)
                for candidate in selected
            )
            question_f1 = max(
                (
                    _candidate_f1(candidate.text, question.answer)
                    for candidate in selected
                ),
                default=0.0,
            )
        else:
            texts, sources = compose_transition_context(
                question,
                arm,
                max_prompt_tokens=max_prompt_tokens,
            )
            context_tokens = sum(count_tokens(text) for text in texts)
            hit = contains_answer(texts, question.answer)
            question_f1 = best_f1(texts, question.answer)
        hits[question.question_id] = hit
        f1_values.append(question_f1)
        token_counts.append(context_tokens)
        expected = set(question.evidence_sources)
        if expected:
            retrieved = {source for source in sources if source}
            coverage = len(expected & retrieved) / len(expected)
            source_coverage.append(coverage)
            source_all.append(coverage == 1.0)

    n = len(pack.questions)
    recall = sum(hits.values()) / n if n else 0.0
    mean_tokens = sum(token_counts) / n if n else 0.0
    gains = losses = 0
    if stay_hits is not None:
        gains = sum(hits[key] and not stay_hits.get(key, False) for key in hits)
        losses = sum(not hits[key] and stay_hits.get(key, False) for key in hits)
    return (
        TransitionArmScore(
            arm=arm,
            questions=n,
            literal_recall=recall,
            mean_best_f1=(sum(f1_values) / n if n else 0.0),
            mean_context_tokens=mean_tokens,
            recall_per_1k_tokens=(
                recall * 100.0 / (mean_tokens / 1000.0)
                if mean_tokens
                else 0.0
            ),
            evidence_source_recall=(
                sum(source_coverage) / len(source_coverage)
                if source_coverage
                else None
            ),
            evidence_all_source_recall=(
                sum(source_all) / len(source_all) if source_all else None
            ),
            gained_vs_stay=gains,
            lost_vs_stay=losses,
        ),
        hits,
    )
