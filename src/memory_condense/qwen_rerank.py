"""Bounded query-conditioned Qwen reranking for retrieved chunk candidates.

The Qwen prefix is a transient control plane: candidate text is truncated,
inspected in small groups, and discarded. Only ordinary ``RetrievalResult``
objects plus scalar diagnostics leave the pass; no residual, attention, Q/K/V,
or token sequence is persisted in the memory store.
"""

from __future__ import annotations

import gc
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from memory_condense._tokenizer import truncate_to_tokens
from memory_condense.head_memory import (
    AssociativeMemoryCandidate,
    NestedMemoryInspection,
)
from memory_condense.ranking import min_max_normalize
from memory_condense.schemas import RetrievalResult


@dataclass(frozen=True, slots=True)
class QwenRerankReport:
    """Text-free diagnostics for one bounded candidate tournament."""

    input_candidates: int
    protected_candidates: int
    inspected_candidates: int
    qwen_candidates_added: int
    output_candidates: int
    passes: int
    max_workspace_candidates: int
    max_workspace_tokens: int
    total_candidate_inspections: int
    retained_transformer_state_bytes: int = 0

    def model_dump(self) -> dict[str, int]:
        return asdict(self)


class QwenCandidateReranker:
    """Reserve a few source slots for winners of a recursive QK/OV tournament."""

    def __init__(
        self,
        linker: Any,
        *,
        candidate_pool: int = 64,
        qwen_slots: int = 6,
        group_size: int = 8,
        beam_per_group: int = 2,
        candidate_tokens: int = 64,
        query_tokens: int = 96,
        score_weight: float = 0.35,
    ) -> None:
        if candidate_pool < 1:
            raise ValueError("candidate_pool must be positive")
        if qwen_slots < 1 or qwen_slots > candidate_pool:
            raise ValueError("qwen_slots must lie in [1, candidate_pool]")
        if group_size < 2 or group_size > int(linker.max_candidates):
            raise ValueError("group_size must lie in [2, linker.max_candidates]")
        if beam_per_group < 1 or beam_per_group >= int(linker.max_candidates):
            raise ValueError("beam_per_group must be smaller than linker.max_candidates")
        if candidate_tokens < 1 or query_tokens < 1:
            raise ValueError("candidate and query token caps must be positive")
        if not 0.0 <= score_weight <= 1.0:
            raise ValueError("score_weight must lie in [0, 1]")
        self.linker = linker
        self.candidate_pool = int(candidate_pool)
        self.qwen_slots = int(qwen_slots)
        self.group_size = int(group_size)
        self.beam_per_group = int(beam_per_group)
        self.candidate_tokens = int(candidate_tokens)
        self.query_tokens = int(query_tokens)
        self.score_weight = float(score_weight)
        self.last_report: QwenRerankReport | None = None

    def select(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Choose attended evidence seeds without directly replacing context.

        The returned objects carry only chunk pointers and scalar QK/OV utility.
        They are intended to parameterize a bounded second retrieval round;
        candidate text and transformer state remain local to this call.
        """

        if top_k <= 0:
            self.last_report = QwenRerankReport(0, 0, 0, 0, 0, 0, 0, 0, 0)
            return []
        bounded: list[RetrievalResult] = []
        seen: set[str] = set()
        for result in candidates:
            chunk_id = result.chunk.chunk_id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            bounded.append(result)
            if len(bounded) >= self.candidate_pool:
                break
        if not bounded:
            self.last_report = QwenRerankReport(0, 0, 0, 0, 0, 0, 0, 0, 0)
            return []

        by_id = {result.chunk.chunk_id: result for result in bounded}
        inspectable = [
            AssociativeMemoryCandidate(
                episode_id=result.chunk.chunk_id,
                text=truncate_to_tokens(result.chunk.text, self.candidate_tokens),
                score=float(result.score),
                route=result.route or "hybrid",
                metadata={
                    "source_id": result.memory_source_id
                    or getattr(result.turn, "source_id", None),
                },
            )
            for result in bounded
        ]
        groups = [
            inspectable[start : start + self.group_size]
            for start in range(0, len(inspectable), self.group_size)
        ]
        inspection: NestedMemoryInspection = self.linker.inspect_nested(
            truncate_to_tokens(query, self.query_tokens),
            groups,
            beam_per_group=self.beam_per_group,
            top_k=min(top_k, int(self.linker.max_candidates)),
            score_mode="qk_ov",
        )
        utilities = [
            max(0.0, float(hit.qk_score))
            + math.log1p(max(0.0, float(hit.ov_transport)))
            for hit in inspection.hits
        ]
        normalized = min_max_normalize(utilities)
        selected: list[RetrievalResult] = []
        for hit, score in zip(inspection.hits, normalized, strict=True):
            original = by_id.get(hit.episode_id)
            if original is None:
                continue
            selected.append(
                original.model_copy(
                    update={
                        "route": "qwen_attention_seed",
                        "association_score": score,
                    }
                )
            )
            if len(selected) >= top_k:
                break
        self.last_report = QwenRerankReport(
            input_candidates=len(bounded),
            protected_candidates=0,
            inspected_candidates=len(bounded),
            qwen_candidates_added=len(selected),
            output_candidates=len(selected),
            passes=inspection.passes,
            max_workspace_candidates=inspection.max_workspace_candidates,
            max_workspace_tokens=inspection.max_workspace_tokens,
            total_candidate_inspections=inspection.total_candidate_inspections,
        )
        return selected

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Keep strong scalar candidates and fill reserved slots with Qwen winners."""

        if top_k <= 0:
            self.last_report = QwenRerankReport(0, 0, 0, 0, 0, 0, 0, 0, 0)
            return []
        bounded: list[RetrievalResult] = []
        seen: set[str] = set()
        for result in candidates:
            chunk_id = result.chunk.chunk_id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            bounded.append(result)
            if len(bounded) >= self.candidate_pool:
                break
        if not bounded:
            self.last_report = QwenRerankReport(0, 0, 0, 0, 0, 0, 0, 0, 0)
            return []

        output_limit = min(top_k, len(bounded))
        reserved = min(self.qwen_slots, output_limit)
        protected_count = max(0, output_limit - reserved)
        protected = bounded[:protected_count]
        protected_ids = {result.chunk.chunk_id for result in protected}
        exploration = [
            result
            for result in bounded
            if result.chunk.chunk_id not in protected_ids
        ]

        if not exploration:
            selected = bounded[:output_limit]
            self.last_report = QwenRerankReport(
                input_candidates=len(bounded),
                protected_candidates=len(protected),
                inspected_candidates=0,
                qwen_candidates_added=0,
                output_candidates=len(selected),
                passes=0,
                max_workspace_candidates=0,
                max_workspace_tokens=0,
                total_candidate_inspections=0,
            )
            return selected

        by_id = {result.chunk.chunk_id: result for result in exploration}
        inspectable = [
            AssociativeMemoryCandidate(
                episode_id=result.chunk.chunk_id,
                text=truncate_to_tokens(result.chunk.text, self.candidate_tokens),
                score=float(result.score),
                route=result.route or "hybrid_source_local",
                metadata={
                    "source_id": result.memory_source_id
                    or getattr(result.turn, "source_id", None),
                },
            )
            for result in exploration
        ]
        groups = [
            inspectable[start : start + self.group_size]
            for start in range(0, len(inspectable), self.group_size)
        ]
        inspection: NestedMemoryInspection = self.linker.inspect_nested(
            truncate_to_tokens(query, self.query_tokens),
            groups,
            beam_per_group=self.beam_per_group,
            top_k=min(reserved, int(self.linker.max_candidates)),
            score_mode="qk_ov",
        )
        utilities = [
            max(0.0, float(hit.qk_score))
            + math.log1p(max(0.0, float(hit.ov_transport)))
            for hit in inspection.hits
        ]
        normalized = min_max_normalize(utilities)
        floor = min((float(result.score) for result in protected), default=0.0)

        promoted: list[RetrievalResult] = []
        for hit, qwen_score in zip(inspection.hits, normalized, strict=True):
            original = by_id.get(hit.episode_id)
            if original is None:
                continue
            blended = (
                (1.0 - self.score_weight) * float(original.score)
                + self.score_weight * qwen_score
            )
            promoted.append(
                original.model_copy(
                    update={
                        "score": max(floor, blended),
                        "route": "qwen_rerank",
                        "association_score": qwen_score,
                    }
                )
            )
            if len(promoted) >= reserved:
                break

        selected = [*protected, *promoted]
        selected_ids = {result.chunk.chunk_id for result in selected}
        for result in bounded:
            if result.chunk.chunk_id in selected_ids:
                continue
            selected.append(result)
            selected_ids.add(result.chunk.chunk_id)
            if len(selected) >= output_limit:
                break
        selected = selected[:output_limit]
        self.last_report = QwenRerankReport(
            input_candidates=len(bounded),
            protected_candidates=len(protected),
            inspected_candidates=len(exploration),
            qwen_candidates_added=len(promoted),
            output_candidates=len(selected),
            passes=inspection.passes,
            max_workspace_candidates=inspection.max_workspace_candidates,
            max_workspace_tokens=inspection.max_workspace_tokens,
            total_candidate_inspections=inspection.total_candidate_inspections,
        )
        return selected

    def close(self) -> None:
        """Drop the prefix model and release cached CUDA allocations."""

        linker = getattr(self, "linker", None)
        torch = getattr(getattr(linker, "encoder", None), "_torch", None)
        if linker is not None:
            self.linker = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
