"""Provider-free semantic reranking over a bounded retrieval frontier.

The MS MARCO cross-encoder is a standard query/passage relevance baseline.
It is deliberately a read-time control plane: raw ``RetrievalResult`` rows
remain the payload, scores and model activations are transient, and no score
is written to the memory store.  An optional Qwen prefix selector may consume
the semantic order to demote duplicate event support after relevance ranking.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from memory_condense.domain._tokenizer import truncate_to_tokens
from memory_condense.search.selectors.coverage_models import ReportDumpMixin
from memory_condense.search.selectors.coverage_selector import compile_set_program
from memory_condense.domain.schemas import RetrievalResult


MS_MARCO_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
MS_MARCO_MODEL_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
MS_MARCO_WEIGHTS_SHA256 = (
    "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
)


def verify_ms_marco_checkpoint(model_dir: str | Path) -> str:
    """Verify the pinned safe-tensors weights and return their SHA-256."""

    root = Path(model_dir)
    required = (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete {MS_MARCO_MODEL_ID} checkpoint under {root}: "
            f"missing {', '.join(missing)}"
        )
    digest = hashlib.sha256()
    with (root / "model.safetensors").open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != MS_MARCO_WEIGHTS_SHA256:
        raise ValueError(
            f"unexpected {MS_MARCO_MODEL_ID} weights sha256: {actual}; "
            f"expected {MS_MARCO_WEIGHTS_SHA256} from revision "
            f"{MS_MARCO_MODEL_REVISION}"
        )
    return actual


@dataclass(frozen=True, slots=True)
class CrossEncoderSelectionReport(ReportDumpMixin):
    """Text-free diagnostics for one semantic and optional grouping pass."""

    operator: str
    cardinality: int | None
    requires_completeness: bool
    input_candidates: int
    inspected_candidates: int
    classified_candidates: int
    event_clusters: int
    new_assignments: int
    existing_assignments: int
    null_assignments: int
    uncertain_assignments: int
    output_candidates: int
    representatives: int
    supporting_candidates: int
    workspace_tokens: int
    elapsed_s: float
    semantic_model_id: str
    semantic_model_revision: str
    semantic_checkpoint_sha256: str
    semantic_rerank_enabled: bool
    semantic_score_only_enabled: bool
    semantic_inspected_candidates: int
    semantic_workspace_tokens: int
    semantic_elapsed_s: float
    prefix_model_id: str = ""
    prefix_model_revision: str = ""
    prefix_checkpoint_sha256: str = ""
    prefix_device: str = ""
    prefix_dtype: str = ""
    prefix_layers: int = 0
    prefix_attention_layer: int = -1
    frontier_candidates: int = 0
    frontier_attempted: int = 0
    frontier_uninspected: int = 0
    frontier_exhaustive: bool = False
    frontier_batches: int = 0
    routed_frontier_exhaustive: bool | None = None
    active_partition_total: int | None = None
    active_partition_inspected: int | None = None
    active_partition_exhaustive: bool | None = None
    active_partition_sources_total: int | None = None
    active_partition_structural_rows: int = 0
    active_partition_structural_hypotheses: int = 0
    active_partition_candidates_admitted: int = 0
    active_partition_candidates_already_present: int = 0
    active_partition_candidates_replaced: int = 0
    active_partition_candidates_truncated: int = 0
    active_partition_structural_overflow: int = 0
    active_partition_scan_contract: str = ""
    active_partition_semantically_complete: bool | None = None
    partition_scope_kind: str = "approximate_top_k"
    partition_inventory_total: int | None = None
    selected_partition_count: int | None = None
    partition_scope_exhaustive: bool | None = None
    selected_scope_structurally_complete: bool | None = None
    global_semantic_complete: bool | None = None
    allow_selected_scope_fixed_k_closure: bool = False
    cardinality_deficit: int = 0
    duplicate_grouping: Mapping[str, Any] | None = None
    retained_transformer_state_bytes: int = 0
    fallback_reason: str = ""


# The grouper's ``model_dump`` fields mirrored into the combined report, keyed
# by the coercion each mirrored value receives.  Fields whose default or
# coercion is unique (``operator``, ``uncertain_assignments``, ...) stay
# explicit at the construction site.
_GROUPING_INT_FIELDS = (
    "event_clusters",
    "new_assignments",
    "existing_assignments",
    "null_assignments",
    "representatives",
    "supporting_candidates",
    "frontier_candidates",
    "frontier_attempted",
    "frontier_uninspected",
    "frontier_batches",
)
_GROUPING_BLANKABLE_INT_FIELDS = (
    "prefix_layers",
    "active_partition_structural_rows",
    "active_partition_structural_hypotheses",
    "active_partition_candidates_admitted",
    "active_partition_candidates_already_present",
    "active_partition_candidates_replaced",
    "active_partition_candidates_truncated",
    "active_partition_structural_overflow",
    "cardinality_deficit",
)
_GROUPING_STR_FIELDS = (
    "prefix_model_id",
    "prefix_model_revision",
    "prefix_checkpoint_sha256",
    "prefix_device",
    "prefix_dtype",
)
_GROUPING_BOOL_FIELDS = (
    "frontier_exhaustive",
    "allow_selected_scope_fixed_k_closure",
)
_GROUPING_PASSTHROUGH_FIELDS = (
    "routed_frontier_exhaustive",
    "active_partition_total",
    "active_partition_inspected",
    "active_partition_exhaustive",
    "active_partition_sources_total",
    "active_partition_semantically_complete",
    "partition_inventory_total",
    "selected_partition_count",
    "partition_scope_exhaustive",
    "selected_scope_structurally_complete",
    "global_semantic_complete",
)


def _mirrored_grouping_fields(grouping: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the shared diagnostics out of one grouper report dict."""

    mirrored: dict[str, Any] = {
        name: int(grouping.get(name, 0)) for name in _GROUPING_INT_FIELDS
    }
    mirrored.update(
        (name, int(grouping.get(name, 0) or 0))
        for name in _GROUPING_BLANKABLE_INT_FIELDS
    )
    mirrored.update(
        (name, str(grouping.get(name, ""))) for name in _GROUPING_STR_FIELDS
    )
    mirrored.update(
        (name, bool(grouping.get(name, False)))
        for name in _GROUPING_BOOL_FIELDS
    )
    mirrored.update(
        (name, grouping.get(name)) for name in _GROUPING_PASSTHROUGH_FIELDS
    )
    mirrored["prefix_attention_layer"] = int(
        grouping.get("prefix_attention_layer", -1)
    )
    mirrored["partition_scope_kind"] = str(
        grouping.get("partition_scope_kind", "approximate_top_k")
    )
    mirrored["active_partition_scan_contract"] = str(
        grouping.get("active_partition_scan_contract", "") or ""
    )
    return mirrored


@dataclass(frozen=True, slots=True)
class CrossEncoderCompanionReport(ReportDumpMixin):
    """Text-free diagnostics for one bounded source-companion pass."""

    input_sources: int
    input_candidates: int
    inspected_candidates: int
    selected_sources: int
    candidate_pool: int
    workspace_tokens: int
    elapsed_s: float
    selected_chunk_ids: Mapping[str, str]
    retained_transformer_state_bytes: int = 0
    fallback_reason: str = ""


def _source_id(result: RetrievalResult) -> str:
    return result.durable_source_id


class MSMarcoCrossEncoderSelector:
    """Deterministically rerank a bounded set without dropping evidence.

    Raw MS MARCO logits are used only for ordering; they are not calibrated
    membership probabilities.  Equal scores retain the incoming order.  Rows
    outside ``candidate_pool`` remain after the scored prefix in their original
    order, so this selector never deletes a candidate unless ``max_results`` is
    explicitly supplied by its caller.
    """

    requires_baseline_ranking = False

    def __init__(
        self,
        encoder: Any,
        *,
        candidate_pool: int = 64,
        candidate_tokens: int = 96,
        query_tokens: int = 192,
        batch_size: int = 32,
        max_length: int = 256,
        max_workspace_tokens: int = 8192,
        duplicate_grouper: Any | None = None,
        model_id: str = MS_MARCO_MODEL_ID,
        model_revision: str = MS_MARCO_MODEL_REVISION,
        checkpoint_sha256: str = MS_MARCO_WEIGHTS_SHA256,
        semantic_rerank: bool = True,
        semantic_score_only: bool = False,
        strict: bool = False,
    ) -> None:
        if min(
            candidate_pool,
            candidate_tokens,
            query_tokens,
            batch_size,
            max_length,
            max_workspace_tokens,
        ) < 1:
            raise ValueError("cross-encoder bounds must be positive")
        if max_length > max_workspace_tokens:
            raise ValueError("cross-encoder max_length exceeds its workspace")
        if semantic_rerank and semantic_score_only:
            raise ValueError(
                "semantic rerank and score-only modes are mutually exclusive"
            )
        self.encoder = encoder
        self.candidate_pool = int(candidate_pool)
        self.candidate_tokens = int(candidate_tokens)
        self.query_tokens = int(query_tokens)
        self.max_length = int(max_length)
        self.max_workspace_tokens = int(max_workspace_tokens)
        self.batch_size = min(
            int(batch_size),
            max(1, self.max_workspace_tokens // self.max_length),
        )
        self.duplicate_grouper = duplicate_grouper
        self.allow_selected_scope_fixed_k_closure = bool(
            getattr(
                duplicate_grouper,
                "allow_selected_scope_fixed_k_closure",
                False,
            )
        )
        self.model_id = str(model_id)
        self.model_revision = str(model_revision)
        self.checkpoint_sha256 = str(checkpoint_sha256)
        self.semantic_rerank = bool(semantic_rerank)
        self.semantic_score_only = bool(semantic_score_only)
        self.strict = bool(strict)
        self.last_report: CrossEncoderSelectionReport | None = None
        self.last_source_companion_report: CrossEncoderCompanionReport | None = None
        self.last_candidate_trace: list[dict[str, Any]] = []

    def close(self) -> None:
        """Release both transient models and the CUDA allocator cache."""

        grouper = self.duplicate_grouper
        self.duplicate_grouper = None
        if grouper is not None:
            close = getattr(grouper, "close", None)
            if callable(close):
                close()
        encoder = self.encoder
        self.encoder = None
        if encoder is not None:
            del encoder
        self.last_report = None
        self.last_source_companion_report = None
        self.last_candidate_trace = []
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - torch is optional at import time
            pass

    @staticmethod
    def _unique(
        candidates: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]:
        unique: list[RetrievalResult] = []
        seen: set[str] = set()
        for result in candidates:
            chunk_id = result.chunk.chunk_id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            unique.append(result)
        return unique

    def _semantic_order(
        self,
        query: str,
        unique: Sequence[RetrievalResult],
    ) -> tuple[list[RetrievalResult], dict[str, float | None], float, str]:
        bounded = list(unique[: self.candidate_pool])
        if not bounded:
            return list(unique), {}, 0.0, ""
        encoder = self.encoder
        if encoder is None:
            raise RuntimeError("cross-encoder selector is closed")
        query_text = truncate_to_tokens(query, self.query_tokens)
        pairs = [
            (
                query_text,
                truncate_to_tokens(result.chunk.text, self.candidate_tokens),
            )
            for result in bounded
        ]
        started = time.perf_counter()
        try:
            predicted = encoder.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            scores = np.asarray(predicted, dtype=np.float64).reshape(-1)
            if scores.size != len(bounded):
                raise ValueError(
                    "cross-encoder returned "
                    f"{scores.size} scores for {len(bounded)} candidates"
                )
            if not np.isfinite(scores).all():
                raise ValueError("cross-encoder returned a non-finite score")
            order = sorted(
                range(len(bounded)),
                key=lambda index: (-float(scores[index]), index),
            )
            ranked = [bounded[index] for index in order]
            ranked.extend(unique[len(bounded) :])
            score_by_id = {
                result.chunk.chunk_id: float(score)
                for result, score in zip(bounded, scores, strict=True)
            }
            return ranked, score_by_id, time.perf_counter() - started, ""
        except Exception as exc:
            if self.strict:
                raise
            return (
                list(unique),
                {result.chunk.chunk_id: None for result in bounded},
                time.perf_counter() - started,
                f"{type(exc).__name__}: {exc}",
            )

    def select_source_companions(
        self,
        query: str,
        candidates_by_source: Mapping[str, Sequence[RetrievalResult]],
    ) -> Mapping[str, RetrievalResult]:
        """Choose one query-relevant raw payload per source in one CE call.

        Candidate admission is round-robin by source so a large first source
        cannot consume the bounded semantic workspace.  Ties preserve each
        source's incoming order.  A backend failure returns the first supplied
        row for every non-empty source, so metadata hydration cannot lose an
        otherwise available companion.
        """

        started = time.perf_counter()
        source_rows: list[tuple[str, list[RetrievalResult]]] = []
        for source_id, candidates in candidates_by_source.items():
            rows: list[RetrievalResult] = []
            seen: set[str] = set()
            for result in candidates:
                chunk_id = result.chunk.chunk_id
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                rows.append(result)
            if rows:
                source_rows.append((str(source_id), rows))

        fallback = {source_id: rows[0] for source_id, rows in source_rows}
        flattened: list[tuple[str, RetrievalResult]] = []
        depth = 0
        while len(flattened) < self.candidate_pool:
            added = False
            for source_id, rows in source_rows:
                if depth >= len(rows):
                    continue
                flattened.append((source_id, rows[depth]))
                added = True
                if len(flattened) >= self.candidate_pool:
                    break
            if not added:
                break
            depth += 1

        fallback_reason = ""
        winners = dict(fallback)
        encoder = self.encoder
        if flattened and encoder is None:
            raise RuntimeError("cross-encoder selector is closed")
        if flattened:
            query_text = truncate_to_tokens(query, self.query_tokens)
            pairs = [
                (
                    query_text,
                    truncate_to_tokens(result.chunk.text, self.candidate_tokens),
                )
                for _source_id, result in flattened
            ]
            try:
                predicted = encoder.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                scores = np.asarray(predicted, dtype=np.float64).reshape(-1)
                if scores.size != len(flattened):
                    raise ValueError(
                        "cross-encoder returned "
                        f"{scores.size} scores for {len(flattened)} candidates"
                    )
                if not np.isfinite(scores).all():
                    raise ValueError("cross-encoder returned a non-finite score")
                best_scores: dict[str, float] = {}
                for (source_id, result), score in zip(
                    flattened,
                    scores,
                    strict=True,
                ):
                    value = float(score)
                    if source_id not in best_scores or value > best_scores[source_id]:
                        best_scores[source_id] = value
                        winners[source_id] = result
            except Exception as exc:
                if self.strict:
                    raise
                winners = fallback
                fallback_reason = f"{type(exc).__name__}: {exc}"

        elapsed = time.perf_counter() - started
        workspace_tokens = (
            min(self.batch_size, len(flattened)) * self.max_length
            if flattened
            else 0
        )
        self.last_source_companion_report = CrossEncoderCompanionReport(
            input_sources=len(source_rows),
            input_candidates=sum(len(rows) for _source_id, rows in source_rows),
            inspected_candidates=len(flattened),
            selected_sources=len(winners),
            candidate_pool=self.candidate_pool,
            workspace_tokens=workspace_tokens,
            elapsed_s=elapsed,
            selected_chunk_ids={
                source_id: result.chunk.chunk_id
                for source_id, result in winners.items()
            },
            fallback_reason=fallback_reason,
        )
        return winners

    def select(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        max_results: int | None = None,
        source_timestamps: Mapping[str, str] | None = None,
        active_partition_total: int | None = None,
        active_partition_inspected: int | None = None,
        active_partition_scan: Mapping[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if max_results is not None and max_results < 1:
            raise ValueError("max_results must be positive when supplied")
        started = time.perf_counter()
        unique = self._unique(candidates)
        semantic_scoring = self.semantic_rerank or self.semantic_score_only
        if semantic_scoring:
            scored_order, scores, semantic_elapsed, semantic_fallback = (
                self._semantic_order(query, unique)
            )
            semantic = scored_order if self.semantic_rerank else list(unique)
        else:
            semantic = list(unique)
            scored_order = list(unique)
            scores = {}
            semantic_elapsed = 0.0
            semantic_fallback = ""
        cross_encoder_rank = {
            result.chunk.chunk_id: rank
            for rank, result in enumerate(scored_order, start=1)
        }
        trace_by_id = {
            result.chunk.chunk_id: {
                "chunk_id": result.chunk.chunk_id,
                "source_id": _source_id(result),
                "cross_encoder_input_rank": rank,
                "cross_encoder_score": scores.get(result.chunk.chunk_id),
                "cross_encoder_rank": cross_encoder_rank[result.chunk.chunk_id],
            }
            for rank, result in enumerate(unique, start=1)
        }

        grouping_report: dict[str, Any] | None = None
        grouper = self.duplicate_grouper
        if grouper is not None:
            group_kwargs: dict[str, Any] = {
                "max_results": max_results,
                "source_timestamps": source_timestamps,
            }
            group_select = grouper.select
            try:
                parameters = tuple(
                    inspect.signature(group_select).parameters.values()
                )
            except (TypeError, ValueError):
                parameters = ()
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            if accepts_kwargs or any(
                parameter.name == "semantic_scores"
                for parameter in parameters
            ):
                group_kwargs["semantic_scores"] = {
                    result.chunk.chunk_id: scores.get(result.chunk.chunk_id)
                    for result in semantic
                }
            for name, value in (
                ("active_partition_total", active_partition_total),
                ("active_partition_inspected", active_partition_inspected),
                ("active_partition_scan", active_partition_scan),
            ):
                if accepts_kwargs or any(
                    parameter.name == name for parameter in parameters
                ):
                    group_kwargs[name] = value
            selected = group_select(query, semantic, **group_kwargs)
            report = getattr(grouper, "last_report", None)
            grouping_report = report.model_dump() if report is not None else None
            if grouping_report is not None and int(
                grouping_report.get("retained_transformer_state_bytes", 0)
            ) != 0:
                raise RuntimeError("duplicate grouper retained transformer state")
            for row in getattr(grouper, "last_candidate_trace", ()):
                if not isinstance(row, Mapping):
                    continue
                chunk_id = row.get("chunk_id")
                if not isinstance(chunk_id, str) or chunk_id not in trace_by_id:
                    continue
                for key, value in row.items():
                    if key not in {"chunk_id", "source_id", "selector_input_rank"}:
                        trace_by_id[chunk_id][key] = value
        else:
            selected = list(semantic)
            if max_results is not None:
                selected = selected[:max_results]

        self.last_candidate_trace = [
            trace_by_id[result.chunk.chunk_id] for result in unique
        ]
        program = compile_set_program(query)
        grouping = grouping_report or {}
        grouping_fallback = str(grouping.get("fallback_reason", ""))
        fallback = "; ".join(
            reason for reason in (semantic_fallback, grouping_fallback) if reason
        )
        semantic_workspace_tokens = (
            self.batch_size * self.max_length
            if unique and semantic_scoring
            else 0
        )
        group_workspace = int(grouping.get("workspace_tokens", 0))
        peak_workspace = max(semantic_workspace_tokens, group_workspace)
        semantic_inspected = (
            min(len(unique), self.candidate_pool)
            if semantic_scoring
            else 0
        )
        group_inspected = int(grouping.get("inspected_candidates", 0))
        group_classified = int(grouping.get("classified_candidates", 0))
        self.last_report = CrossEncoderSelectionReport(
            operator=str(grouping.get("operator", program.operator.value)),
            cardinality=grouping.get("cardinality", program.cardinality),
            requires_completeness=bool(
                grouping.get(
                    "requires_completeness",
                    program.requires_completeness,
                )
            ),
            input_candidates=len(unique),
            inspected_candidates=max(semantic_inspected, group_inspected),
            classified_candidates=max(semantic_inspected, group_classified),
            uncertain_assignments=int(
                grouping.get(
                    "uncertain_assignments",
                    max(0, len(unique) - semantic_inspected),
                )
            ),
            output_candidates=len(selected),
            workspace_tokens=peak_workspace,
            elapsed_s=time.perf_counter() - started,
            semantic_model_id=self.model_id,
            semantic_model_revision=self.model_revision,
            semantic_checkpoint_sha256=self.checkpoint_sha256,
            semantic_rerank_enabled=self.semantic_rerank,
            semantic_score_only_enabled=self.semantic_score_only,
            semantic_inspected_candidates=semantic_inspected,
            semantic_workspace_tokens=semantic_workspace_tokens,
            semantic_elapsed_s=semantic_elapsed,
            duplicate_grouping=grouping_report,
            retained_transformer_state_bytes=0,
            fallback_reason=fallback,
            **_mirrored_grouping_fields(grouping),
        )
        return selected
