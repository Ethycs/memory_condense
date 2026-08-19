"""Concrete local BGE/Qwen execution binding for gold-blind LongMemEval.

The receipt distinguishes simultaneous residency from staged BGE release.
This module accepts no benchmark answer, evidence label, provider, or loader.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

from memory_condense.application.condenser import MemoryCondenser
from memory_condense.application.discourse_sources import (
    build_episode_source_candidate_scope,
)
from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
from memory_condense.domain.discourse import canonical_json, identity_sha256, quote_sha256
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.eval.diffuse_longmemeval_analysis import (
    DiffuseLongMemEvalArm,
    DiffuseLongMemEvalRetrievalPhase,
    GoldBlindLongMemEvalQuestion,
    GoldBlindLongMemEvalSample,
    LegacyDiffuseCandidates,
    ingest_gold_blind_sample_deterministically,
    retrieve_diffuse_longmemeval_sample,
)
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    EmbeddingService,
)
from memory_condense.modeling.qwen_prefix import (
    DEFAULT_MODEL_ID as QWEN_MODEL_ID,
    DEFAULT_MODEL_REVISION as QWEN_MODEL_REVISION,
    Qwen3PrefixEncoder,
    expected_prefix_checkpoint_sha256,
)
from memory_condense.search.episodes import (
    EpisodeRepresentativeRetrievalPolicy,
    QwenAttentionHeadSurpriseScorer,
)
from memory_condense.search.episodes.qwen_episode_signal import (
    _canonical_callable_code,
)
from memory_condense.search.selectors.qwen_rerank import QwenCandidateReranker


DIFFUSE_RUNTIME_FORMAT = "memory-condense-longmemeval-diffuse-runtime-v1"
DIFFUSE_RUNTIME_RESULT_FORMAT = (
    "memory-condense-longmemeval-diffuse-runtime-result-v1"
)
FROZEN_LEGACY_QUERY_FORMAT = (
    "memory-condense-longmemeval-frozen-legacy-query-v1"
)
ResidencyMode = Literal[
    "resident_bge_qwen",
    "staged_bge_then_qwen",
]
_PACKED_CONTEXT_MODES = {
    "memory",
    "causal_consolidation",
    "causal_graph",
}


class TreatmentQuestionLike(Protocol):
    """Structural view of the evaluator firebreak's treatment question."""

    question_id: str
    question: str
    question_date: str | None


class TreatmentSampleLike(Protocol):
    """Structural, scorer-free view accepted by the concrete runtime."""

    sample_id: str
    turns: Sequence[tuple[str, str]]
    turn_source_ids: Sequence[str | None]
    turn_created_at: Sequence[datetime | None]
    questions: Sequence[TreatmentQuestionLike]


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        normalized = int(value)  # type: ignore[arg-type]
        exact = math.isfinite(float(value)) and float(value) == normalized  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if not exact or normalized < 1:
        raise ValueError(f"{label} must be a positive integer")
    return normalized


def _digest(value: object, label: str) -> str:
    normalized = str(value).casefold()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _callable_identity(value: object) -> dict[str, str | None]:
    """Identify the callable itself, not merely its metaclass."""

    if inspect.isclass(value) or inspect.isfunction(value):
        owner = value
    else:
        owner = type(value)
    target = (
        getattr(value, "__init__", None)
        if inspect.isclass(value)
        else value
        if inspect.isfunction(value)
        else getattr(type(value), "__call__", None)
    )
    code = getattr(target, "__code__", None)
    callable_name = (
        f"{getattr(owner, '__module__', '')}."
        f"{getattr(owner, '__qualname__', '')}"
    )
    code_sha256 = None
    if code is not None:
        canonical = _canonical_callable_code(
            code,
            stable_filename=callable_name,
        )
        code_sha256 = hashlib.sha256(canonical).hexdigest()
    return {
        "callable": callable_name,
        "python_code_sha256": code_sha256,
    }


def _treatment_corpus_sha256(
    sample_id: str,
    turns: Sequence[tuple[str, str]],
    source_ids: Sequence[str | None],
    created_at: Sequence[datetime | None],
) -> str:
    return identity_sha256(
        {
            "sample_id": sample_id,
            "turns": [
                {
                    "ordinal": index,
                    "role": role,
                    "text_sha256": quote_sha256(text),
                    "source_id": source_id,
                    "created_at": (
                        None if timestamp is None else timestamp.isoformat()
                    ),
                }
                for index, ((role, text), source_id, timestamp) in enumerate(
                    zip(turns, source_ids, created_at, strict=True)
                )
            ],
        }
    )


def gold_blind_from_treatment_sample(
    sample: TreatmentSampleLike,
) -> GoldBlindLongMemEvalSample:
    """Adapt one closed-schema treatment sample without importing ``tools``.

    The raw question is the retrieval query.  Its optional date is confined to
    the eventual prompt question, matching the firebreak projection.
    """

    sample_id = str(sample.sample_id).strip()
    turns = tuple((str(role), str(text)) for role, text in sample.turns)
    sources = tuple(sample.turn_source_ids)
    timestamps = tuple(sample.turn_created_at)
    if len(sources) != len(turns) or len(timestamps) != len(turns):
        raise ValueError("treatment turn coordinates must be parallel")
    questions: list[GoldBlindLongMemEvalQuestion] = []
    for raw in sample.questions:
        question = str(raw.question).strip()
        question_date = raw.question_date
        prompt = (
            question
            if question_date is None
            else f"[Question asked at {str(question_date).strip()}]\n{question}"
        )
        questions.append(
            GoldBlindLongMemEvalQuestion(
                question_id=str(raw.question_id),
                retrieval_query=question,
                prompt_question=prompt,
            )
        )
    return GoldBlindLongMemEvalSample(
        sample_id=sample_id,
        turns=turns,
        turn_source_ids=sources,
        turn_created_at=timestamps,
        questions=tuple(questions),
        corpus_sha256=_treatment_corpus_sha256(
            sample_id,
            turns,
            sources,
            timestamps,
        ),
    )


def _require_supported_direct_mode(retrieval: RetrievalConfig) -> None:
    if retrieval.mode in _PACKED_CONTEXT_MODES:
        raise ValueError(
            f"retrieval mode {retrieval.mode!r} produces a packed context, not "
            "an exact RetrievalResult anchor sequence"
        )
    if retrieval.mode == "hybrid_source" and any(
        (
            retrieval.query_facet_retrieval,
            retrieval.role_aware_retrieval,
            retrieval.multi_fact_source_diversity,
            retrieval.source_partition_routing,
        )
    ):
        raise ValueError(
            "the live hybrid_source API does not implement facet, role, "
            "diversity, or partition controls; use hybrid_graph"
        )


def retrieve_exact_legacy_anchors(
    condenser: Any,
    query: str,
    retrieval: RetrievalConfig,
) -> tuple[RetrievalResult, ...]:
    """Execute one direct-result legacy arm with every applicable flag.

    The mappings follow the live ``MemoryCondenser`` method signatures.  The
    three context-packing modes fail closed because their benchmark path does
    not expose an authoritative ``RetrievalResult`` sequence.
    """

    if not isinstance(retrieval, RetrievalConfig):
        raise TypeError("retrieval must be a RetrievalConfig")
    normalized_query = str(query).strip()
    if not normalized_query:
        raise ValueError("query must be non-empty")
    _require_supported_direct_mode(retrieval)

    if retrieval.mode == "span":
        results = condenser.search_spans(
            normalized_query,
            levels=retrieval.span_levels,
            k_per_level=retrieval.k_per_level,
        )
    elif retrieval.mode == "source":
        results = condenser.search_sources(
            normalized_query,
            k_sources=retrieval.k_sources,
        )
    elif retrieval.mode == "anchored_source":
        results = condenser.search_anchored_sources(
            normalized_query,
            k=retrieval.k,
            ef_search=retrieval.ef_search,
            candidates=retrieval.candidates,
            alpha=retrieval.alpha,
        )
    elif retrieval.mode == "hybrid_source":
        results = condenser.search_hybrid_sources(
            normalized_query,
            k=retrieval.k,
            source_slots=retrieval.source_slots,
            source_candidate_pool=retrieval.source_candidate_pool,
            source_activation_k=retrieval.source_activation_k,
            source_local_search=retrieval.source_local_search,
            use_source_reranker=retrieval.qwen_rerank,
            ef_search=retrieval.ef_search,
            candidates=retrieval.candidates,
            alpha=retrieval.alpha,
        )
    elif retrieval.mode == "hybrid_graph":
        results = condenser.search_hybrid_graph(
            normalized_query,
            k=retrieval.k,
            neighbor_radius=retrieval.neighbor_radius,
            neighbor_slots=retrieval.neighbor_slots,
            neighbor_direction=retrieval.neighbor_direction,
            source_slots=retrieval.source_slots,
            source_candidate_pool=retrieval.source_candidate_pool,
            source_activation_k=retrieval.source_activation_k,
            query_facet_retrieval=retrieval.query_facet_retrieval,
            query_facet_slots=retrieval.query_facet_slots,
            query_facet_max=retrieval.query_facet_max,
            role_aware_retrieval=retrieval.role_aware_retrieval,
            role_user_weight=retrieval.role_user_weight,
            role_assistant_weight=retrieval.role_assistant_weight,
            role_system_weight=retrieval.role_system_weight,
            multi_fact_source_diversity=(
                retrieval.multi_fact_source_diversity
            ),
            source_tfisf_activation=retrieval.source_tfisf_activation,
            source_tfisf_slots=retrieval.source_tfisf_slots,
            source_hsc_activation=retrieval.source_hsc_activation,
            source_hsc_slots=retrieval.source_hsc_slots,
            source_hsc_hops=retrieval.source_hsc_hops,
            source_hsc_chunk_slots=retrieval.source_hsc_chunk_slots,
            source_partition_routing=retrieval.source_partition_routing,
            source_partition_slots=retrieval.source_partition_slots,
            source_partition_separator=retrieval.source_partition_separator,
            source_local_search=retrieval.source_local_search,
            use_source_reranker=retrieval.qwen_rerank,
            use_attention_feedback=retrieval.qwen_feedback,
            feedback_slots=retrieval.qwen_feedback_slots,
            feedback_seed_slots=retrieval.qwen_feedback_seed_slots,
            feedback_evidence_tokens=retrieval.qwen_feedback_evidence_tokens,
            feedback_query_tokens=retrieval.qwen_feedback_query_tokens,
            ef_search=retrieval.ef_search,
            candidates=retrieval.candidates,
            alpha=retrieval.alpha,
        )
    elif retrieval.mode == "hybrid_neighbor":
        results = condenser.search_hybrid_neighbors(
            normalized_query,
            k=retrieval.k,
            radius=retrieval.neighbor_radius,
            max_neighbors=retrieval.neighbor_slots,
            replacement_slots=retrieval.neighbor_replacement_slots,
            ef_search=retrieval.ef_search,
            candidates=retrieval.candidates,
            alpha=retrieval.alpha,
        )
    elif retrieval.effective_hybrid:
        results = condenser.search_hybrid(
            normalized_query,
            k=retrieval.k,
            ef_search=retrieval.ef_search,
            candidates=retrieval.candidates,
            alpha=retrieval.alpha,
        )
    else:
        results = condenser.search(
            normalized_query,
            k=retrieval.k,
            ef_search=retrieval.ef_search,
        )
    frozen = tuple(results)
    if any(not isinstance(item, RetrievalResult) for item in frozen):
        raise TypeError("legacy retrieval must return RetrievalResult values")
    return frozen


def _anchor_sequence_sha256(
    anchors: Sequence[RetrievalResult],
) -> str:
    return identity_sha256(
        [item.model_dump(mode="json") for item in anchors]
    )


def _source_streams_identity(condenser: Any) -> tuple[tuple[str, ...], str]:
    streams = tuple(condenser.discourse_source_streams())
    universe = tuple(str(stream.source_id) for stream in streams)
    return universe, identity_sha256(
        [
            {
                "source_id": stream.source_id,
                "content_chunk_ids": list(stream.content_chunk_ids),
                "metadata_chunk_ids": list(stream.metadata_chunk_ids),
                "first_ordinal": stream.first_ordinal,
                "last_ordinal": stream.last_ordinal,
                "stream_sha256": stream.stream_sha256,
            }
            for stream in streams
        ]
    )


@dataclass(frozen=True, slots=True)
class FrozenLegacyQueryInputs:
    """Artifact-agnostic legacy rows frozen before a staged Qwen load."""

    query: str
    retrieval_policy_sha256: str
    anchors: tuple[RetrievalResult, ...]
    lexical_sources: tuple[tuple[str, float], ...]
    universe_source_ids: tuple[str, ...]
    source_streams_sha256: str
    format: str = FROZEN_LEGACY_QUERY_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        query = str(self.query).strip()
        if not query:
            raise ValueError("query must be non-empty")
        object.__setattr__(self, "query", query)
        _digest(self.retrieval_policy_sha256, "retrieval_policy_sha256")
        _digest(self.source_streams_sha256, "source_streams_sha256")
        anchors = tuple(self.anchors)
        if any(not isinstance(item, RetrievalResult) for item in anchors):
            raise TypeError("anchors must contain RetrievalResult values")
        object.__setattr__(self, "anchors", anchors)
        lexical: list[tuple[str, float]] = []
        for source_id, score in self.lexical_sources:
            normalized_id = str(source_id).strip()
            normalized_score = float(score)
            if not normalized_id or not math.isfinite(normalized_score):
                raise ValueError("source lexical rows require finite scores and IDs")
            lexical.append((normalized_id, normalized_score))
        object.__setattr__(self, "lexical_sources", tuple(lexical))
        universe = tuple(str(item).strip() for item in self.universe_source_ids)
        if any(not item for item in universe) or len(set(universe)) != len(universe):
            raise ValueError("source universe must contain unique non-empty IDs")
        object.__setattr__(self, "universe_source_ids", universe)
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("frozen legacy query receipt does not match")
        object.__setattr__(self, "receipt_sha256", expected)

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "format": self.format,
            "query_sha256": identity_sha256({"query": self.query}),
            "retrieval_policy_sha256": self.retrieval_policy_sha256,
            "anchor_sequence_sha256": _anchor_sequence_sha256(self.anchors),
            "lexical_sources": [list(item) for item in self.lexical_sources],
            "universe_source_ids": list(self.universe_source_ids),
            "source_streams_sha256": self.source_streams_sha256,
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


def freeze_legacy_query_inputs(
    condenser: Any,
    queries: Sequence[str],
    retrieval: RetrievalConfig,
) -> tuple[FrozenLegacyQueryInputs, ...]:
    """Freeze dense anchors and independent all-source lexical rows."""

    universe, streams_sha256 = _source_streams_identity(condenser)
    policy_sha256 = identity_sha256(retrieval.model_dump(mode="json"))
    frozen: list[FrozenLegacyQueryInputs] = []
    by_query: dict[str, FrozenLegacyQueryInputs] = {}
    for raw_query in queries:
        query = str(raw_query).strip()
        existing = by_query.get(query)
        if existing is not None:
            frozen.append(existing)
            continue
        anchors = retrieve_exact_legacy_anchors(condenser, query, retrieval)
        lexical = tuple(
            condenser.retriever.source_tfisf_query(
                query,
                k_sources=max(len(universe), 1),
            )
        )
        row = FrozenLegacyQueryInputs(
            query=query,
            retrieval_policy_sha256=policy_sha256,
            anchors=anchors,
            lexical_sources=lexical,
            universe_source_ids=universe,
            source_streams_sha256=streams_sha256,
        )
        by_query[query] = row
        frozen.append(row)
    return tuple(frozen)


@dataclass(frozen=True, slots=True)
class ResidentLegacyDiffuseInputProvider:
    """Retrieve anchors, then call the authoritative all-source router."""

    max_sources: int = 64
    rrf_constant: int = 60

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_sources", _positive_int(self.max_sources, "max_sources"))
        object.__setattr__(self, "rrf_constant", _positive_int(self.rrf_constant, "rrf_constant"))

    def analysis_identity_payload(self) -> Mapping[str, object]:
        return {
            "provider": "resident-exact-anchor-all-source-router-v1",
            "residency_mode": "resident_bge_qwen",
            "max_sources": self.max_sources,
            "rrf_constant": self.rrf_constant,
        }

    def __call__(
        self,
        condenser: MemoryCondenser,
        *,
        query: str,
        retrieval: RetrievalConfig,
        artifact_id: str,
    ) -> LegacyDiffuseCandidates:
        anchors = retrieve_exact_legacy_anchors(condenser, query, retrieval)
        scope = condenser.route_discourse_episode_sources(
            query,
            anchors,
            artifact_id=artifact_id,
            max_sources=self.max_sources,
            rrf_constant=self.rrf_constant,
        )
        return LegacyDiffuseCandidates(
            anchors=anchors,
            source_candidate_scope=scope,
        )


@dataclass(frozen=True, slots=True)
class _FrozenLegacyQuerySnapshot:
    query: str
    retrieval_policy_sha256: str
    anchor_json: tuple[str, ...]
    lexical_sources: tuple[tuple[str, float], ...]
    universe_source_ids: tuple[str, ...]
    source_streams_sha256: str
    receipt_sha256: str

    def materialize_anchors(self) -> tuple[RetrievalResult, ...]:
        return tuple(
            RetrievalResult.model_validate_json(payload)
            for payload in self.anchor_json
        )


@dataclass(frozen=True, slots=True)
class FrozenLegacyDiffuseInputProvider:
    """Bind pre-Qwen rows to the compiled artifact without another search."""

    inputs: tuple[FrozenLegacyQueryInputs, ...]
    max_sources: int = 64
    rrf_constant: int = 60
    _snapshots: tuple[_FrozenLegacyQuerySnapshot, ...] = field(init=False, repr=False)
    _by_query: Mapping[str, _FrozenLegacyQuerySnapshot] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_sources", _positive_int(self.max_sources, "max_sources"))
        object.__setattr__(self, "rrf_constant", _positive_int(self.rrf_constant, "rrf_constant"))
        rows = tuple(self.inputs)
        snapshots: list[_FrozenLegacyQuerySnapshot] = []
        by_query: dict[str, _FrozenLegacyQuerySnapshot] = {}
        for row in rows:
            if not isinstance(row, FrozenLegacyQueryInputs):
                raise TypeError("inputs must contain FrozenLegacyQueryInputs")
            receipt = identity_sha256(row.identity_payload(include_receipt=False))
            if receipt != row.receipt_sha256:
                raise ValueError("frozen legacy query receipt does not match")
            snapshot = _FrozenLegacyQuerySnapshot(
                query=row.query,
                retrieval_policy_sha256=row.retrieval_policy_sha256,
                anchor_json=tuple(
                    canonical_json(item.model_dump(mode="json")) for item in row.anchors
                ),
                lexical_sources=row.lexical_sources,
                universe_source_ids=row.universe_source_ids,
                source_streams_sha256=row.source_streams_sha256,
                receipt_sha256=receipt,
            )
            snapshots.append(snapshot)
            previous = by_query.setdefault(row.query, snapshot)
            if previous.receipt_sha256 != receipt:
                raise ValueError("duplicate staged query has different frozen inputs")
        object.__setattr__(self, "inputs", rows)
        object.__setattr__(self, "_snapshots", tuple(snapshots))
        object.__setattr__(self, "_by_query", MappingProxyType(by_query))

    def analysis_identity_payload(self) -> Mapping[str, object]:
        return {
            "provider": "staged-frozen-anchor-all-source-router-v1",
            "residency_mode": "staged_bge_then_qwen",
            "max_sources": self.max_sources,
            "rrf_constant": self.rrf_constant,
            "frozen_query_receipts_sha256": identity_sha256(
                [item.receipt_sha256 for item in self._snapshots]
            ),
        }

    def __call__(
        self,
        condenser: MemoryCondenser,
        *,
        query: str,
        retrieval: RetrievalConfig,
        artifact_id: str,
    ) -> LegacyDiffuseCandidates:
        normalized_query = str(query).strip()
        row = self._by_query.get(normalized_query)
        if row is None:
            raise KeyError("query was not frozen before the staged Qwen load")
        if row.retrieval_policy_sha256 != identity_sha256(
            retrieval.model_dump(mode="json")
        ):
            raise ValueError("retrieval policy changed after staged acquisition")
        universe, streams_sha256 = _source_streams_identity(condenser)
        if universe != row.universe_source_ids or streams_sha256 != row.source_streams_sha256:
            raise RuntimeError("source corpus changed after staged acquisition")
        anchors = row.materialize_anchors()
        scope = build_episode_source_candidate_scope(
            artifact_id=artifact_id,
            snapshot=condenser.discourse.snapshot(),
            query=normalized_query,
            anchors=anchors,
            lexical_sources=row.lexical_sources,
            universe_source_ids=row.universe_source_ids,
            max_sources=self.max_sources,
            rrf_constant=self.rrf_constant,
        )
        return LegacyDiffuseCandidates(
            anchors=anchors,
            source_candidate_scope=scope,
        )


@dataclass(frozen=True, slots=True)
class DiffuseLongMemEvalRuntimeConfig:
    """Pinned local model and hard-cap controls for a sanitized canary."""

    qwen_model_dir: Path
    residency_mode: ResidencyMode = "resident_bge_qwen"
    embedding_batch_size: int = 32
    qwen_device: str = "cuda"
    qwen_dtype: str = "float16"
    qwen_max_candidates: int = 8
    qwen_max_workspace_tokens: int = 2048
    # Measured Qwen-prefix plus worst bounded forward residency was ~2402 MiB;
    # keep more than a 512 MiB margin instead of treating that peak as a cap.
    resident_min_free_mib: int = 3072
    source_router_max_sources: int = 64
    source_router_rrf_constant: int = 60
    surprise_max_spans: int = 256
    surprise_span_tokens: int = 64
    surprise_probe_tokens: int = 96
    surprise_max_transport_dimension: int = 8192
    representative_max_input_sources: int = 64
    representative_max_source_groups: int = 64
    representative_max_episodes_per_source: int = 64
    representative_max_total_episodes: int = 256
    representative_max_per_episode: int = 2
    representative_group_size: int = 8
    representative_beam_per_group: int = 2
    representative_top_k: int = 8
    representative_tokens: int = 96
    representative_query_tokens: int = 96
    representative_score_mode: Literal["qk", "qk_ov"] = "qk_ov"

    def __post_init__(self) -> None:
        model_dir = Path(self.qwen_model_dir)
        object.__setattr__(self, "qwen_model_dir", model_dir)
        if self.residency_mode not in {
            "resident_bge_qwen",
            "staged_bge_then_qwen",
        }:
            raise ValueError("unsupported model residency mode")
        for name in (
            "embedding_batch_size",
            "qwen_max_candidates",
            "qwen_max_workspace_tokens",
            "resident_min_free_mib",
            "source_router_max_sources",
            "source_router_rrf_constant",
            "surprise_max_spans",
            "surprise_span_tokens",
            "surprise_probe_tokens",
            "surprise_max_transport_dimension",
            "representative_max_input_sources",
            "representative_max_source_groups",
            "representative_max_episodes_per_source",
            "representative_max_total_episodes",
            "representative_max_per_episode",
            "representative_group_size",
            "representative_beam_per_group",
            "representative_top_k",
            "representative_tokens",
            "representative_query_tokens",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.representative_max_input_sources < self.source_router_max_sources:
            raise ValueError("representative input cap must cover the source-router cap")
        if self.representative_max_source_groups < self.source_router_max_sources:
            raise ValueError("representative source-group cap must cover the source-router cap")
        if self.representative_max_source_groups > self.representative_max_input_sources:
            raise ValueError("representative source groups exceed input sources")
        if self.representative_group_size > self.qwen_max_candidates:
            raise ValueError("representative group size exceeds Qwen candidate cap")
        if self.representative_beam_per_group >= self.qwen_max_candidates:
            raise ValueError("representative beam must be smaller than Qwen candidate cap")
        if self.representative_top_k > self.representative_max_total_episodes:
            raise ValueError("representative top_k exceeds the total episode cap")


@dataclass(frozen=True, slots=True)
class ResidencyPreflightObservation:
    policy: str
    device: str
    required_free_bytes: int
    observed_free_bytes: int | None
    observed_total_bytes: int | None
    embedding_released_before_qwen_load: bool

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
            }
        )


def _resident_cuda_preflight(
    device: str,
    required_free_bytes: int,
) -> ResidencyPreflightObservation:
    normalized_device = str(device).strip().casefold()
    if not normalized_device.startswith("cuda"):
        return ResidencyPreflightObservation(
            policy="cuda-mem-get-info-min-free-v1:not-required",
            device=normalized_device,
            required_free_bytes=required_free_bytes,
            observed_free_bytes=None,
            observed_total_bytes=None,
            embedding_released_before_qwen_load=False,
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("resident Qwen CUDA preflight found no CUDA device")
    free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(device))
    free = int(free_bytes)
    total = int(total_bytes)
    if free < required_free_bytes:
        raise MemoryError(
            f"resident Qwen load requires at least {required_free_bytes} free "
            f"CUDA bytes after BGE ingest; observed {free}"
        )
    return ResidencyPreflightObservation(
        policy="cuda-mem-get-info-min-free-v1",
        device=normalized_device,
        required_free_bytes=required_free_bytes,
        observed_free_bytes=free,
        observed_total_bytes=total,
        embedding_released_before_qwen_load=False,
    )


@dataclass(frozen=True, slots=True)
class DiffuseLongMemEvalRuntimeFactories:
    """Dependency seam used only to keep construction tests model-free."""

    embedding: Callable[..., Any] = EmbeddingService
    condenser: Callable[..., Any] = MemoryCondenser
    qwen_encoder: Callable[..., Any] = Qwen3PrefixEncoder
    qwen_linker: Callable[..., Any] = QwenMemoryLinker
    qwen_scorer: Callable[..., Any] = QwenAttentionHeadSurpriseScorer
    qwen_reranker: Callable[..., Any] = QwenCandidateReranker
    resident_preflight: Callable[[str, int], ResidencyPreflightObservation] = (
        _resident_cuda_preflight
    )


_OWNED_FACTORY_CALLABLES = (
    EmbeddingService,
    MemoryCondenser,
    Qwen3PrefixEncoder,
    QwenMemoryLinker,
    QwenAttentionHeadSurpriseScorer,
    QwenCandidateReranker,
    _resident_cuda_preflight,
)
_OWNED_CRITICAL_NAMES = (
    (EmbeddingService, ("__init__", "_load_model", "embed_chunks", "embed_query", "close")),
    (MemoryCondenser, (
        "__init__", "ingest_many", "search", "search_hybrid", "search_spans",
        "search_sources", "search_anchored_sources", "search_hybrid_sources",
        "search_hybrid_graph", "search_hybrid_neighbors",
        "route_discourse_episode_sources",
    )),
    (Qwen3PrefixEncoder, ("__init__",)),
    (QwenMemoryLinker, ("__init__", "inspect_coverage", "inspect_nested")),
    (QwenAttentionHeadSurpriseScorer, ("__init__", "score_sequence")),
    (QwenCandidateReranker, ("__init__", "select")),
)
_OWNED_CRITICAL_METHODS = tuple(
    (owner, name, getattr(owner, name))
    for owner, names in _OWNED_CRITICAL_NAMES
    for name in names
)


def _runtime_factories_certified(
    factories: DiffuseLongMemEvalRuntimeFactories,
) -> bool:
    configured = (
        factories.embedding,
        factories.condenser,
        factories.qwen_encoder,
        factories.qwen_linker,
        factories.qwen_scorer,
        factories.qwen_reranker,
        factories.resident_preflight,
    )
    return configured == _OWNED_FACTORY_CALLABLES and all(
        getattr(owner, method_name, None) is expected
        for owner, method_name, expected in _OWNED_CRITICAL_METHODS
    )


@dataclass(frozen=True, slots=True)
class _OwnedQwenRuntime:
    encoder: Any
    linker: Any
    scorer: Any
    reranker: Any | None


class RepresentativePolicyFactory:
    """Callable object whose immutable controls are visible to receipts."""

    def __init__(self, runtime: DiffuseLongMemEvalRuntimeConfig) -> None:
        self._runtime = runtime

    def analysis_identity_payload(self) -> Mapping[str, object]:
        return {
            "factory": "diffuse-representative-policy-v1",
            **self.controls_payload(),
        }

    def controls_payload(self) -> dict[str, object]:
        runtime = self._runtime
        return {
            "max_input_sources": runtime.representative_max_input_sources,
            "max_source_groups": runtime.representative_max_source_groups,
            "max_episodes_per_source": runtime.representative_max_episodes_per_source,
            "max_total_episodes": runtime.representative_max_total_episodes,
            "max_representatives_per_episode": runtime.representative_max_per_episode,
            "group_size": runtime.representative_group_size,
            "beam_per_group": runtime.representative_beam_per_group,
            "top_k": runtime.representative_top_k,
            "representative_tokens": runtime.representative_tokens,
            "query_tokens": runtime.representative_query_tokens,
            "score_mode": runtime.representative_score_mode,
        }

    def __call__(self, artifact_id: str) -> EpisodeRepresentativeRetrievalPolicy:
        return EpisodeRepresentativeRetrievalPolicy(
            artifact_id=artifact_id,
            **self.controls_payload(),
        )


class DiffuseLongMemEvalExecutionBinding:
    """Reusable resident binding, or deliberately single-use staged binding."""

    def __init__(
        self,
        *,
        config: EvalConfig,
        runtime: DiffuseLongMemEvalRuntimeConfig,
        factories: DiffuseLongMemEvalRuntimeFactories,
    ) -> None:
        if config.embedding_device is None or not str(config.embedding_device).strip():
            raise ValueError("diffuse runtime requires an explicit embedding_device")
        if config.max_prompt_tokens is None:
            raise ValueError("diffuse runtime requires an explicit prompt cap")
        _require_supported_direct_mode(config.retrieval)
        if runtime.residency_mode == "staged_bge_then_qwen" and (
            config.retrieval.qwen_rerank or config.retrieval.qwen_feedback
        ):
            raise ValueError(
                "staged mode cannot run Qwen-controlled legacy retrieval before "
                "the shared Qwen runtime is loaded"
            )
        if config.retrieval.qwen_rerank_use_cav and (
            config.retrieval.qwen_rerank or config.retrieval.qwen_feedback
        ):
            raise ValueError(
                "the shared owned Qwen surprise/representative linker cannot carry CAV"
            )
        if (
            config.retrieval.qwen_rerank or config.retrieval.qwen_feedback
        ) and config.retrieval.qwen_rerank_max_workspace_tokens != (
            runtime.qwen_max_workspace_tokens
        ):
            raise ValueError(
                "legacy Qwen retrieval and the shared runtime must use the same "
                "workspace-token cap"
            )
        self.config = config
        self.runtime = runtime
        self.factories = factories
        self.embedder = factories.embedding(
            model_name=DEFAULT_MODEL_NAME,
            model_revision=DEFAULT_MODEL_REVISION,
            device=config.embedding_device,
            batch_size=runtime.embedding_batch_size,
            verify_checkpoint=True,
        )
        self.embedding_identity = {
            "backend": "sentence-transformers.encode-v1",
            "model_id": DEFAULT_MODEL_NAME,
            "model_revision": DEFAULT_MODEL_REVISION,
            "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
            "dimension": DEFAULT_MODEL_DIM,
            "device": str(config.embedding_device).casefold(),
            "batch_size": runtime.embedding_batch_size,
            "normalize_embeddings": False,
            "output_dtype": "float32",
        }
        self.representative_policy_factory = RepresentativePolicyFactory(runtime)
        self._qwen: _OwnedQwenRuntime | None = None
        self._preflight: ResidencyPreflightObservation | None = None
        self._staged_consumed = False

    @property
    def runtime_binding_certified(self) -> bool:
        """Recheck owned callables so post-construction shadowing fails shut."""

        return _runtime_factories_certified(self.factories)

    def analysis_identity_payload(self) -> Mapping[str, object]:
        retrieval = self.config.retrieval
        expected_qwen = expected_prefix_checkpoint_sha256(
            retrieval.qwen_rerank_prefix_layers,
            model_id=QWEN_MODEL_ID,
            model_revision=QWEN_MODEL_REVISION,
        )
        return {
            "format": DIFFUSE_RUNTIME_FORMAT,
            "runtime_binding_certified": self.runtime_binding_certified,
            "residency_mode": self.runtime.residency_mode,
            "resident_preflight": {
                "policy": (
                    "cuda-mem-get-info-min-free-v1"
                    if self.runtime.residency_mode == "resident_bge_qwen"
                    else "bge-close-before-qwen-load-v1"
                ),
                "required_free_bytes": (
                    self.runtime.resident_min_free_mib * 1024 * 1024
                    if self.runtime.residency_mode == "resident_bge_qwen"
                    else 0
                ),
            },
            "embedding": dict(self.embedding_identity),
            "qwen": {
                # Filesystem placement is not behavioral identity.  The
                # verified checkpoint digest below binds the exact bytes.
                "model_locator": "local-verified-checkpoint",
                "model_id": QWEN_MODEL_ID,
                "model_revision": QWEN_MODEL_REVISION,
                "checkpoint_sha256": expected_qwen,
                "prefix_layers": retrieval.qwen_rerank_prefix_layers,
                "attention_layer": retrieval.qwen_rerank_attention_layer,
                "device": self.runtime.qwen_device,
                "dtype": self.runtime.qwen_dtype,
                "max_candidates": self.runtime.qwen_max_candidates,
                "max_workspace_tokens": self.runtime.qwen_max_workspace_tokens,
                "surprise": {
                    "max_spans": self.runtime.surprise_max_spans,
                    "span_token_cap": self.runtime.surprise_span_tokens,
                    "probe_token_cap": self.runtime.surprise_probe_tokens,
                    "max_transport_dimension": self.runtime.surprise_max_transport_dimension,
                },
            },
            "source_router": {
                "max_sources": self.runtime.source_router_max_sources,
                "rrf_constant": self.runtime.source_router_rrf_constant,
            },
            "representative": self.representative_policy_factory.controls_payload(),
            "retrieval_policy_sha256": identity_sha256(
                retrieval.model_dump(mode="json")
            ),
            "factories": {
                name: _callable_identity(getattr(self.factories, name))
                for name in (
                    "embedding",
                    "condenser",
                    "qwen_encoder",
                    "qwen_linker",
                    "qwen_scorer",
                    "qwen_reranker",
                    "resident_preflight",
                )
            },
        }

    @property
    def binding_sha256(self) -> str:
        return identity_sha256(dict(self.analysis_identity_payload()))

    def new_condenser(self, data_dir: Path) -> Any:
        config = self.config
        return self.factories.condenser(
            data_dir=data_dir,
            model_name=DEFAULT_MODEL_NAME,
            chunker_min_tokens=config.chunker.min_tokens,
            chunker_max_tokens=config.chunker.max_tokens,
            device=config.embedding_device,
            auto_extract=False,
            embedder=self.embedder,
            persist_index_on_close=True,
        )

    def prepare_resident_replay_runtime(self) -> tuple[ResidencyPreflightObservation, _OwnedQwenRuntime]:
        """Establish certified simultaneous BGE/Qwen residency for replay."""
        if self.runtime.residency_mode != "resident_bge_qwen":
            raise ValueError("shared-base replay requires resident_bge_qwen")
        if not self.runtime_binding_certified:
            raise RuntimeError("embedding residency requires certified runtime")
        self.embedder._load_model()  # noqa: SLF001 - owned runtime boundary
        from memory_condense.eval._diffuse_base_store import (
            owned_build_runtime_identity,
            validate_embedder_certification,
        )
        validate_embedder_certification(
            self.embedder, owned_build_runtime_identity(self.new_condenser)
        )
        observation = self._resident_preflight()
        qwen = self.ensure_qwen_runtime()
        if not self.runtime_binding_certified:
            raise RuntimeError("runtime certification changed during model load")
        return observation, qwen

    def _resident_preflight(self) -> ResidencyPreflightObservation:
        if self._preflight is None:
            self._preflight = self.factories.resident_preflight(
                self.runtime.qwen_device,
                self.runtime.resident_min_free_mib * 1024 * 1024,
            )
        return self._preflight

    def _staged_release(self) -> ResidencyPreflightObservation:
        close = getattr(self.embedder, "close", None)
        if not callable(close):
            raise TypeError("staged embedding runtime must expose close()")
        close()
        observation = ResidencyPreflightObservation(
            policy="bge-close-before-qwen-load-v1",
            device=str(self.runtime.qwen_device).casefold(),
            required_free_bytes=0,
            observed_free_bytes=None,
            observed_total_bytes=None,
            embedding_released_before_qwen_load=True,
        )
        self._preflight = observation
        return observation

    def ensure_qwen_runtime(self) -> _OwnedQwenRuntime:
        if self._qwen is not None:
            return self._qwen
        if self._preflight is None:
            raise RuntimeError("Qwen load requires an observed residency preflight")
        retrieval = self.config.retrieval
        checkpoint = expected_prefix_checkpoint_sha256(
            retrieval.qwen_rerank_prefix_layers,
            model_id=QWEN_MODEL_ID,
            model_revision=QWEN_MODEL_REVISION,
        )
        encoder = self.factories.qwen_encoder(
            self.runtime.qwen_model_dir,
            layers=retrieval.qwen_rerank_prefix_layers,
            device=self.runtime.qwen_device,
            dtype=self.runtime.qwen_dtype,
            model_id=QWEN_MODEL_ID,
            model_revision=QWEN_MODEL_REVISION,
            expected_checkpoint_sha256=checkpoint,
        )
        linker = self.factories.qwen_linker(
            encoder,
            layer=retrieval.qwen_rerank_attention_layer,
            cav_bank=None,
            max_candidates=self.runtime.qwen_max_candidates,
            max_workspace_tokens=self.runtime.qwen_max_workspace_tokens,
        )
        scorer = self.factories.qwen_scorer(
            linker,
            max_spans=self.runtime.surprise_max_spans,
            span_token_cap=self.runtime.surprise_span_tokens,
            probe_token_cap=self.runtime.surprise_probe_tokens,
            max_transport_dimension=self.runtime.surprise_max_transport_dimension,
        )
        reranker = None
        if retrieval.qwen_rerank or retrieval.qwen_feedback:
            reranker = self.factories.qwen_reranker(
                linker,
                candidate_pool=(
                    retrieval.qwen_feedback_candidate_pool
                    if retrieval.qwen_feedback
                    else retrieval.qwen_rerank_candidate_pool
                ),
                qwen_slots=(
                    retrieval.qwen_feedback_seed_slots
                    if retrieval.qwen_feedback
                    else retrieval.qwen_rerank_slots
                ),
                group_size=retrieval.qwen_rerank_group_size,
                beam_per_group=retrieval.qwen_rerank_beam_per_group,
                candidate_tokens=retrieval.qwen_rerank_candidate_tokens,
                query_tokens=(
                    retrieval.qwen_feedback_query_tokens
                    if retrieval.qwen_feedback
                    else retrieval.qwen_rerank_query_tokens
                ),
                score_weight=retrieval.qwen_rerank_score_weight,
                association_artifact=None,
            )
        self._qwen = _OwnedQwenRuntime(
            encoder=encoder,
            linker=linker,
            scorer=scorer,
            reranker=reranker,
        )
        return self._qwen


def build_diffuse_longmemeval_execution_binding(
    *,
    config: EvalConfig,
    runtime: DiffuseLongMemEvalRuntimeConfig,
    factories: DiffuseLongMemEvalRuntimeFactories | None = None,
) -> DiffuseLongMemEvalExecutionBinding:
    """Construct the pinned local binding without loading either model."""

    return DiffuseLongMemEvalExecutionBinding(
        config=config,
        runtime=runtime,
        factories=factories or DiffuseLongMemEvalRuntimeFactories(),
    )


@dataclass(frozen=True, slots=True)
class DiffuseLongMemEvalRuntimeResult:
    phase: DiffuseLongMemEvalRetrievalPhase
    runtime_binding_sha256: str
    runtime_binding_certified: bool
    residency_preflight: ResidencyPreflightObservation
    format: str = DIFFUSE_RUNTIME_RESULT_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _digest(self.runtime_binding_sha256, "runtime_binding_sha256")
        if type(self.runtime_binding_certified) is not bool:
            raise ValueError("runtime_binding_certified must be boolean")
        if not isinstance(self.residency_preflight, ResidencyPreflightObservation):
            raise TypeError(
                "residency_preflight must be a ResidencyPreflightObservation"
            )
        expected = identity_sha256(self.identity_payload(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("runtime result receipt does not match")
        object.__setattr__(self, "receipt_sha256", expected)

    def identity_payload(self, *, include_receipt: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "format": self.format,
            "retrieval_phase_receipt_sha256": self.phase.receipt_sha256,
            "runtime_binding_sha256": self.runtime_binding_sha256,
            "runtime_binding_certified": self.runtime_binding_certified,
            "residency_preflight": {
                name: getattr(self.residency_preflight, name)
                for name in self.residency_preflight.__dataclass_fields__
            },
            "residency_preflight_receipt_sha256": (
                self.residency_preflight.receipt_sha256
            ),
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


def run_diffuse_treatment_sample(
    sample: TreatmentSampleLike,
    *,
    binding: DiffuseLongMemEvalExecutionBinding,
    arm: DiffuseLongMemEvalArm,
    data_dir: str | Path,
) -> DiffuseLongMemEvalRuntimeResult:
    """Run one sanitized treatment sample; benchmark gold cannot enter."""

    if not isinstance(binding, DiffuseLongMemEvalExecutionBinding):
        raise TypeError("binding must be a DiffuseLongMemEvalExecutionBinding")
    if not binding.runtime_binding_certified:
        raise RuntimeError(
            "sanitized canary execution requires the exact certified local runtime"
        )
    store_path = Path(data_dir)
    if store_path.exists():
        raise FileExistsError("diffuse runtime requires a fresh store path")
    if (
        binding.runtime.residency_mode == "staged_bge_then_qwen"
        and binding._staged_consumed
    ):
        raise RuntimeError("a staged binding is single-sample; construct a fresh binding")
    blind = gold_blind_from_treatment_sample(sample)
    condenser = binding.new_condenser(store_path)
    if type(condenser) is not MemoryCondenser:
        close = getattr(condenser, "close", None)
        if callable(close):
            close()
        raise TypeError(
            "certified condenser factory must return the exact MemoryCondenser"
        )
    shadowed = {
        method_name
        for owner, method_name, _expected in _OWNED_CRITICAL_METHODS
        if owner is MemoryCondenser and method_name in vars(condenser)
    }
    if shadowed:
        condenser.close()
        raise RuntimeError(
            f"certified condenser has shadowed methods: {sorted(shadowed)}"
        )
    try:
        ingest_gold_blind_sample_deterministically(condenser, blind)
        if binding.runtime.residency_mode == "staged_bge_then_qwen":
            frozen = freeze_legacy_query_inputs(
                condenser,
                [item.retrieval_query for item in blind.questions],
                binding.config.retrieval,
            )
            observation = binding._staged_release()
            provider: Any = FrozenLegacyDiffuseInputProvider(
                frozen,
                max_sources=binding.runtime.source_router_max_sources,
                rrf_constant=binding.runtime.source_router_rrf_constant,
            )
            binding._staged_consumed = True
        else:
            observation = binding._resident_preflight()
            provider = ResidentLegacyDiffuseInputProvider(
                max_sources=binding.runtime.source_router_max_sources,
                rrf_constant=binding.runtime.source_router_rrf_constant,
            )
        qwen = binding.ensure_qwen_runtime()
        if qwen.reranker is not None:
            condenser.set_source_candidate_reranker(qwen.reranker)
        phase = retrieve_diffuse_longmemeval_sample(
            condenser,
            blind,
            config=binding.config,
            arm=arm,
            legacy_input_provider=provider,
            qwen_scorer=(
                qwen.scorer
                if arm.compilation.boundary_mode == "qwen_head"
                else None
            ),
            embedding_identity=binding.embedding_identity,
            representative_linker=qwen.linker,
            representative_policy_factory=(
                binding.representative_policy_factory
            ),
        )
        return DiffuseLongMemEvalRuntimeResult(
            phase=phase,
            runtime_binding_sha256=binding.binding_sha256,
            runtime_binding_certified=binding.runtime_binding_certified,
            residency_preflight=observation,
        )
    finally:
        condenser.close()


__all__ = [
    "DIFFUSE_RUNTIME_FORMAT",
    "DIFFUSE_RUNTIME_RESULT_FORMAT",
    "DiffuseLongMemEvalExecutionBinding",
    "DiffuseLongMemEvalRuntimeConfig",
    "DiffuseLongMemEvalRuntimeFactories",
    "DiffuseLongMemEvalRuntimeResult",
    "FrozenLegacyDiffuseInputProvider",
    "FrozenLegacyQueryInputs",
    "RepresentativePolicyFactory",
    "ResidencyMode",
    "ResidencyPreflightObservation",
    "ResidentLegacyDiffuseInputProvider",
    "TreatmentQuestionLike",
    "TreatmentSampleLike",
    "build_diffuse_longmemeval_execution_binding",
    "freeze_legacy_query_inputs",
    "gold_blind_from_treatment_sample",
    "retrieve_exact_legacy_anchors",
    "run_diffuse_treatment_sample",
]
