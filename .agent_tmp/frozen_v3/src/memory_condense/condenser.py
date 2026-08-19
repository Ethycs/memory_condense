from __future__ import annotations

import hashlib
import inspect
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence

import numpy as np

from memory_condense._tokenizer import count_tokens, truncate_to_tokens
from memory_condense.association_store import (
    AssociationArtifact,
    AssociationStore,
    HebbianUpdate,
)
from memory_condense.associative_retrieval import expand_associative_results
from memory_condense.heat_diffusion import expand_heat_diffusion_results
from memory_condense.hebbian_retrieval import (
    expand_hebbian_results,
    retrieval_concept_activations,
)
from memory_condense.chunker import Chunker
from memory_condense.context_packer import (
    ContextBudget,
    ContextPacker,
    ExpansionSelector,
)
from memory_condense.lexical import tokenize
from memory_condense.consolidation import (
    ConsolidationNode,
    ConsolidationUpdate,
    LiveConsolidationStore,
    context_activations,
    expand_context_associations,
    inspect_qwen_context_hyperplane,
)
from memory_condense.db import Database
from memory_condense.embedding import EmbeddingService
from memory_condense.extractor import Extractor, RuleBasedExtractor
from memory_condense.memory_store import MemoryStore
from memory_condense.performance_events import (
    is_direct_past_performance,
    is_performance_query,
    performance_event_key,
)
from memory_condense.ranking import DEFAULT_WEIGHTS, RankWeights
from memory_condense.retrieval import DEFAULT_SPAN_TOKENS, SimilarityRetriever
from memory_condense.schemas import (
    Chunk,
    MemoryResult,
    PackedContext,
    RetrievalResult,
    Turn,
)
from memory_condense.transcript_store import TranscriptStore, parse_source_metadata
from memory_condense.validator import Validator


# Public-facade defaults validated by the locked v3 source-family split. The
# low-level expansion primitive keeps ``None`` controls for explicit ablations.
SAFE_ASSOCIATION_LEXICAL_THRESHOLD = 0.9
SAFE_ASSOCIATION_MAX_TOKEN_INCREASE = 0

_DATED_QUESTION_RE = re.compile(r"^\[Question asked at .+?\]\s*", re.DOTALL)
_FACET_SPLIT_RE = re.compile(r"\s*,\s*(?:and\s+)?|\s+and\s+", re.IGNORECASE)
_DIRECT_DATE_QUERY_RE = re.compile(
    r"\bwhen\b|\b(?:what|which)\s+(?:(?:was|is|were|are)\s+)?"
    r"(?:the\s+)?(?:date|day)\b",
    re.IGNORECASE,
)


def _concept_term(term: str) -> str:
    """Light singular normalization for concept-object set queries."""

    return term[:-1] if len(term) > 4 and term.endswith("s") else term


def rank_concept_members(
    query: str,
    results: Sequence[RetrievalResult],
) -> list[RetrievalResult]:
    """Fuse CAV membership with TF-ISF query-object overlap."""

    if not results:
        return []
    documents = [
        {_concept_term(term) for term in tokenize(result.chunk.text)}
        for result in results
    ]
    query_terms = {_concept_term(term) for term in tokenize(query)}
    frequencies = Counter(term for document in documents for term in document)
    count = len(documents)
    idf = {
        term: math.log2((count + 1.0) / (frequency + 1.0)) + 1.0
        for term, frequency in frequencies.items()
    }
    query_weight = sum(idf.get(term, math.log2(count + 1.0)) for term in query_terms)
    margins = [max(0.0, float(result.score)) for result in results]
    low = min(margins, default=0.0)
    high = max(margins, default=0.0)
    ranked: list[tuple[float, int, RetrievalResult]] = []
    for index, (result, terms, margin) in enumerate(
        zip(results, documents, margins, strict=True)
    ):
        lexical = (
            sum(idf.get(term, 1.0) for term in terms & query_terms) / query_weight
            if query_weight > 0.0
            else 0.0
        )
        normalized_margin = (
            (margin - low) / (high - low) if high > low else 0.0
        )
        score = 0.8 * lexical + 0.2 * normalized_margin
        ranked.append(
            (
                score,
                -index,
                result.model_copy(update={"score": score}),
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [result for _score, _index, result in ranked]


def query_facets(query: str, *, max_facets: int = 4) -> list[str]:
    """Extract explicit list facets for bounded multi-query retrieval.

    This deliberately handles only questions that spell out their facets
    after a colon. It does not guess latent subquestions or call a model, so a
    singleton query keeps exactly the historical retrieval path.
    """

    if max_facets < 1:
        raise ValueError("max_facets must be positive")
    body = _DATED_QUESTION_RE.sub("", query.strip())
    if ":" not in body:
        return []
    tail = body.split(":", 1)[1].strip().rstrip("?.!")
    pieces = _FACET_SPLIT_RE.split(tail)
    facets: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        facet = re.sub(r"^\s*(?:and\s+)?", "", piece, flags=re.IGNORECASE)
        facet = re.sub(r"\s+", " ", facet).strip(" ,;:-")
        key = facet.casefold()
        if len(facet.split()) < 3 or key in seen:
            continue
        facets.append(facet)
        seen.add(key)
        if len(facets) >= max_facets:
            break
    return facets if len(facets) >= 2 else []


def role_aware_results(
    query: str,
    candidates: Sequence[RetrievalResult],
    *,
    user_weight: float = 1.25,
    assistant_weight: float = 0.75,
    system_weight: float = 0.50,
) -> list[RetrievalResult]:
    """Prefer user evidence for explicitly autobiographical questions.

    The prior is deliberately inactive unless the query contains a first-
    person pronoun. It changes only transient scalar scores; durable chunks,
    embeddings, and graph state remain untouched.
    """

    if min(user_weight, assistant_weight, system_weight) < 0.0:
        raise ValueError("role weights must be non-negative")
    if re.search(r"\b(?:i|me|my|mine|myself)\b", query, re.IGNORECASE) is None:
        return list(candidates)
    weights = {
        "user": user_weight,
        "assistant": assistant_weight,
        "system": system_weight,
    }
    ranked: list[tuple[float, int, RetrievalResult]] = []
    for index, result in enumerate(candidates):
        role = result.turn.role.lower() if result.turn is not None else ""
        adjusted = float(result.score) * weights.get(role, 1.0)
        ranked.append(
            (
                adjusted,
                index,
                result.model_copy(update={"score": adjusted}),
            )
        )
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [result for _score, _index, result in ranked]


def is_multi_fact_query(query: str) -> bool:
    """Whether the wording explicitly asks for an ordered or complete set."""

    return re.search(
        r"\b(?:order|ordered|earliest|latest|chronological|sequence|all|each)\b",
        query,
        re.IGNORECASE,
    ) is not None


def _retrieval_source_id(result: RetrievalResult) -> str:
    return str(
        result.memory_source_id
        or (result.turn.source_id if result.turn is not None else None)
        or result.chunk.turn_id
    )


def source_diverse_results(
    candidates: Sequence[RetrievalResult],
) -> list[RetrievalResult]:
    """Round-robin ranked chunks by durable source, preserving local order."""

    source_order: list[str] = []
    groups: dict[str, list[RetrievalResult]] = {}
    for result in candidates:
        source_id = (
            str(result.turn.source_id or result.turn.turn_id)
            if result.turn is not None
            else result.memory_source_id or result.chunk.turn_id
        )
        if source_id not in groups:
            source_order.append(source_id)
            groups[source_id] = []
        groups[source_id].append(result)
    output: list[RetrievalResult] = []
    depth = 0
    while True:
        added = False
        for source_id in source_order:
            group = groups[source_id]
            if depth >= len(group):
                continue
            output.append(group[depth])
            added = True
        if not added:
            return output
        depth += 1


def _source_partition(source_id: str, separator: str) -> str:
    """Top-level durable partition encoded by ``partition::source``."""

    return source_id.split(separator, 1)[0]


def select_source_partitions(
    candidates: Sequence[RetrievalResult],
    *,
    slots: int,
    separator: str = "::",
    max_hits_per_partition: int = 8,
) -> list[str]:
    """Rank coarse partitions by reciprocal-rank heat over chunk hits.

    A single lucky chunk should not route an entire million-token memory.
    Accumulating bounded reciprocal-rank mass rewards partitions that produce
    several independently relevant candidates while remaining insensitive to
    their raw source or transcript length.
    """

    if slots < 1:
        raise ValueError("partition slots must be positive")
    if not separator:
        raise ValueError("partition separator must be non-empty")
    if max_hits_per_partition < 1:
        raise ValueError("max_hits_per_partition must be positive")
    return [
        str(item["partition"])
        for item in source_partition_ranking(
            candidates,
            separator=separator,
            max_hits_per_partition=max_hits_per_partition,
        )[:slots]
    ]


def source_partition_ranking(
    candidates: Sequence[RetrievalResult],
    *,
    separator: str = "::",
    max_hits_per_partition: int = 8,
) -> list[dict[str, str | int | float]]:
    """Expose the bounded coarse-cue evidence used for partition routing.

    Keeping this diagnostic beside the selector prevents benchmark analysis
    from reconstructing route decisions from final packed chunks, which may
    also contain cross-partition consolidation results.
    """

    if not separator:
        raise ValueError("partition separator must be non-empty")
    if max_hits_per_partition < 1:
        raise ValueError("max_hits_per_partition must be positive")
    scores: dict[str, float] = {}
    best_scores: dict[str, float] = {}
    first_rank: dict[str, int] = {}
    hit_counts: dict[str, int] = {}
    for rank, result in enumerate(candidates, start=1):
        if result.turn is None:
            continue
        source_id = str(result.turn.source_id or result.turn.turn_id)
        partition = _source_partition(source_id, separator)
        count = hit_counts.get(partition, 0)
        if count >= max_hits_per_partition:
            continue
        hit_counts[partition] = count + 1
        scores[partition] = scores.get(partition, 0.0) + 1.0 / (60.0 + rank)
        best_scores[partition] = max(
            best_scores.get(partition, float("-inf")),
            float(result.score),
        )
        first_rank.setdefault(partition, rank)
    ordered = sorted(
        scores,
        key=lambda partition: (
            -scores[partition],
            first_rank[partition],
            partition,
        ),
    )
    return [
        {
            "partition": partition,
            "rrf_heat": scores[partition],
            "best_score": best_scores[partition],
            "first_rank": first_rank[partition],
            "hits": hit_counts[partition],
        }
        for partition in ordered
    ]


class SourceCandidateReranker(Protocol):
    candidate_pool: int
    last_report: Any
    association_artifact: AssociationArtifact | None

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
        unique_sources: bool = False,
    ) -> list[RetrievalResult]: ...

    def select(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]: ...


class SourceCompanionSelector(Protocol):
    """Optional transient chooser for ambiguous metadata-only sources."""

    def select_source_companions(
        self,
        query: str,
        candidates_by_source: Mapping[str, Sequence[RetrievalResult]],
    ) -> Mapping[str, RetrievalResult]: ...


_SOURCE_COMPANION_MAX_PER_SOURCE = 4
_SOURCE_CANONICAL_COMPANIONS_PER_SOURCE = 4
_SOURCE_PERFORMANCE_COMPANIONS_PER_SOURCE = 1
_ACTIVE_PARTITION_HYPOTHESIS_CAP = 128


@dataclass(frozen=True, slots=True)
class _ActivePartitionHypothesis:
    """One bounded typed occurrence retained from a complete row scan."""

    chunk_id: str
    source_id: str
    timestamp: str | None
    ordinal: int
    surface_score: float
    identity_key: str | None


@dataclass(frozen=True, slots=True)
class _ActivePartitionRoutingSnapshot:
    """Immutable audit bound to one selected-partition scan and DB turn."""

    routing_identity: str
    query_sha256: str
    transcript_turn: int
    content_high_watermark: int
    selected_partitions: tuple[str, ...]
    routed_source_ids: tuple[str, ...]
    frontier_chunk_ids: tuple[str, ...]
    frontier_routes: tuple[str, ...]
    active_frontier_rows: tuple[tuple[str, str], ...]
    active_partition_total: int
    active_partition_inspected: int
    active_partition_exhaustive: bool
    active_partition_sources_total: int
    active_partition_structural_rows: int
    active_partition_structural_hypotheses: int
    active_partition_candidates_admitted: int
    active_partition_candidates_already_present: int
    active_partition_candidates_replaced: int
    active_partition_candidates_truncated: int
    active_partition_structural_overflow: int
    active_partition_scan_contract: str
    active_partition_semantically_complete: bool
    partition_scope_kind: str
    partition_inventory_total: int
    selected_partition_count: int
    partition_scope_exhaustive: bool
    selected_scope_structurally_complete: bool
    global_semantic_complete: bool

    def matches(
        self,
        query: str,
        transcript_turn: int,
        content_high_watermark: int,
        candidates: Sequence[RetrievalResult],
    ) -> bool:
        return bool(
            transcript_turn == self.transcript_turn
            and content_high_watermark == self.content_high_watermark
            and hashlib.sha256(query.encode("utf-8")).hexdigest()
            == self.query_sha256
            and tuple(
                sorted(
                    (
                        result.chunk.chunk_id,
                        str(result.route or ""),
                    )
                    for result in candidates
                    if str(result.route or "").casefold().startswith(
                        "active_partition_"
                    )
                )
            )
            == self.active_frontier_rows
        )

    def pack_fields(self) -> dict[str, Any]:
        scan = {
            field: getattr(self, field)
            for field in (
                "active_partition_total",
                "active_partition_inspected",
                "active_partition_exhaustive",
                "active_partition_sources_total",
                "active_partition_structural_rows",
                "active_partition_structural_hypotheses",
                "active_partition_candidates_admitted",
                "active_partition_candidates_already_present",
                "active_partition_candidates_replaced",
                "active_partition_candidates_truncated",
                "active_partition_structural_overflow",
                "active_partition_scan_contract",
                "active_partition_semantically_complete",
                "partition_scope_kind",
                "partition_inventory_total",
                "selected_partition_count",
                "partition_scope_exhaustive",
                "selected_scope_structurally_complete",
                "global_semantic_complete",
            )
        }
        # Keep the two legacy scalar arguments during the API transition, but
        # carry the complete validated audit as one indivisible mapping.
        return {
            "active_partition_total": self.active_partition_total,
            "active_partition_inspected": self.active_partition_inspected,
            "active_partition_scan": scan,
        }


class MemoryCondenser:
    """High-level facade wiring the full memory pipeline.

    Everything stateful runs locally: chunking, embedding (bge-m3), the ANN and
    lexical indexes, the memory state machine, and context packing. No LLM call
    is made from this class — memory extraction defaults to the offline
    rule-based extractor. To use LLM-proposed memory, inject an
    ``extractor=LLMExtractor(complete=...)`` with your own provider binding.

    Usage::

        mc = MemoryCondenser(data_dir="./data")
        mc.ingest("user", "I prefer dark mode in all my apps.")
        ctx = mc.build_context("What are my UI preferences?")
        print(ctx.messages)
        mc.close()
    """

    def __init__(
        self,
        data_dir: str | Path = "./data",
        model_name: str = "BAAI/bge-m3",
        chunker_min_tokens: int = 120,
        chunker_max_tokens: int = 250,
        device: str | None = None,
        extractor: Extractor | None = None,
        budget: ContextBudget | None = None,
        auto_extract: bool = True,
        embedder: EmbeddingService | None = None,
        persist_index_on_close: bool = True,
        retriever_max_elements: int = 100_000,
        read_only: bool = False,
    ) -> None:
        data_dir = Path(data_dir)
        if not read_only:
            data_dir.mkdir(parents=True, exist_ok=True)

        self._db = Database(data_dir / "memory.db", read_only=read_only)
        self._transcript = TranscriptStore(self._db)
        self._chunker = Chunker(
            min_tokens=chunker_min_tokens,
            max_tokens=chunker_max_tokens,
        )
        # Injectable so tests (and alternate backends) can avoid loading bge-m3.
        self._embedder = embedder or EmbeddingService(
            model_name=model_name,
            device=device,
        )
        self._associations = AssociationStore(self._db)
        self._retriever = SimilarityRetriever(
            db=self._db,
            dim=self._embedder.dim,
            index_path=data_dir / "hnsw_index.bin",
            association_store=self._associations,
            max_elements=retriever_max_elements,
        )
        self._memory = MemoryStore(self._db, embedder=self._embedder)
        self._consolidation = LiveConsolidationStore(self._db)
        self._validator = Validator(self._db)
        self._extractor = extractor if extractor is not None else RuleBasedExtractor()
        self._packer = ContextPacker(budget)
        self._auto_extract = auto_extract
        # A read-only facade must not rewrite the adjacent ANN artifact on
        # close even when a caller leaves the writable default unchanged.
        self._persist_index_on_close = bool(
            persist_index_on_close and not read_only
        )
        self._source_candidate_reranker: SourceCandidateReranker | None = None
        self._context_candidate_selector: ExpansionSelector | None = None
        self._source_concept_artifact_id: str | None = None
        self.last_source_rerank_report: dict[str, Any] = {}
        self.last_coverage_selection_report: dict[str, Any] = {}
        self.last_coverage_candidate_trace: list[dict[str, Any]] = []
        self.last_partition_routing_report: dict[str, Any] = {}
        self._active_partition_routing_snapshot: (
            _ActivePartitionRoutingSnapshot | None
        ) = None
        self.last_query_facet_report: dict[str, Any] = {}
        self.last_source_diversity_report: dict[str, Any] = {}
        self.last_source_companion_report: dict[str, Any] = {}

    def set_source_candidate_reranker(
        self,
        reranker: SourceCandidateReranker | None,
    ) -> None:
        """Attach one shared transient reranker; no model state enters SQLite."""

        self._source_candidate_reranker = reranker
        artifact = getattr(reranker, "association_artifact", None)
        self._source_concept_artifact_id = (
            artifact.artifact_id if artifact is not None else None
        )
        self.last_source_rerank_report = {}

    def set_context_candidate_selector(
        self,
        selector: ExpansionSelector | None,
    ) -> None:
        """Attach a transient listwise selector immediately before packing.

        The selector receives only the bounded expansion subset.  It may
        reorder or reject candidates, but raw chunks remain the evidence and
        no classifier text or activation is written to the durable store.
        """

        self._context_candidate_selector = selector
        self._packer.expansion_selector = selector
        self.last_coverage_selection_report = {}
        self.last_coverage_candidate_trace = []

    # -- ingestion ----------------------------------------------------------

    def ingest(
        self,
        role: str,
        text: str,
        *,
        source_id: str | None = None,
    ) -> tuple[Turn, list[Chunk]]:
        """Ingest a single conversation turn.

        Stores the turn, chunks and embeds the text, indexes the chunks for
        both dense and lexical retrieval, and — when ``auto_extract`` is on —
        proposes memory items, validates their provenance, and applies the
        surviving ops.
        """
        turn = self._transcript.append(role, text, source_id=source_id)
        chunks = self._chunker.chunk_turn(turn.turn_id, text)

        if chunks:
            chunks = self._embedder.embed_chunks(chunks)
            self._retriever.add_chunks(chunks)

        if self._auto_extract:
            self.extract_memory([turn], chunks)

        return turn, chunks

    def ingest_many(
        self,
        turns: Sequence[tuple[str, str, str | None]],
    ) -> list[tuple[Turn, list[Chunk]]]:
        """Ingest a turn batch with one embedding/index update.

        This is the fast path for document and benchmark loading. Transcript
        order and source provenance remain exact, but all chunks are embedded
        together so a 30-turn session does not launch 30 tiny model forwards.

        Automatic memory extraction remains strictly turn-causal and therefore
        uses :meth:`ingest` sequentially. The batched path is used when
        ``auto_extract=False``, which is already the retrieval-evaluation and
        corpus-indexing configuration.
        """
        records = list(turns)
        if self._auto_extract:
            return [
                self.ingest(role, text, source_id=source_id)
                for role, text, source_id in records
            ]

        staged: list[tuple[Turn, list[Chunk]]] = []
        flat_chunks: list[Chunk] = []
        for role, text, source_id in records:
            turn = self._transcript.append(role, text, source_id=source_id)
            chunks = self._chunker.chunk_turn(turn.turn_id, text)
            staged.append((turn, chunks))
            flat_chunks.extend(chunks)

        if not flat_chunks:
            return staged

        embedded = self._embedder.embed_chunks(flat_chunks)
        self._retriever.add_chunks(embedded)
        by_turn: dict[str, list[Chunk]] = {}
        for chunk in embedded:
            by_turn.setdefault(chunk.turn_id, []).append(chunk)
        return [(turn, by_turn.get(turn.turn_id, [])) for turn, _ in staged]

    def compile_cav_signatures(
        self,
        linker: object,
        artifact: AssociationArtifact,
        chunks: Sequence[Chunk | RetrievalResult],
        *,
        batch_size: int = 8,
        overwrite: bool = False,
        conceptual_spans: bool = True,
    ) -> dict[str, int]:
        """Compile event/concept memberships into bounded durable scalars.

        The Qwen prefix acts only as a write-time teacher.  No residual,
        attention matrix, token sequence, or K/V cache is stored; each chunk
        contributes exactly one float32 value per named concept.
        """

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        bank = getattr(linker, "cav_bank", None)
        compile_many = getattr(linker, "signatures", None)
        if bank is None or compile_many is None:
            raise ValueError("linker must expose a CAV bank and batched signatures")
        if tuple(bank.names) != artifact.concept_names:
            raise ValueError("linker and artifact concept names do not match")
        if int(bank.layer) != artifact.cav_layer:
            raise ValueError("linker and artifact CAV layers do not match")
        self._associations.register_artifact(artifact)

        unique: dict[str, Chunk] = {}
        for value in chunks:
            chunk = value.chunk if isinstance(value, RetrievalResult) else value
            unique.setdefault(chunk.chunk_id, chunk)
        pending = [
            chunk
            for chunk in unique.values()
            if overwrite
            or self._associations.get_signature(
                chunk.chunk_id, artifact.artifact_id
            )
            is None
        ]
        span_texts: list[str] = []
        span_owners: list[str] = []
        for chunk in pending:
            spans = (
                self._chunker.conceptual_spans(chunk.text)
                if conceptual_spans
                else [chunk.text]
            )
            for span in spans or [chunk.text]:
                span_texts.append(span)
                span_owners.append(chunk.chunk_id)
        span_signatures = compile_many(
            span_texts,
            batch_size=batch_size,
        )
        if len(span_signatures) != len(span_texts):
            raise ValueError("linker returned a misaligned signature batch")
        pooled: dict[str, tuple[float, ...]] = {}
        for chunk_id, signature in zip(
            span_owners, span_signatures, strict=True
        ):
            values = tuple(float(value) for value in signature)
            previous = pooled.get(chunk_id)
            pooled[chunk_id] = (
                values
                if previous is None
                else tuple(
                    max(left, right)
                    for left, right in zip(previous, values, strict=True)
                )
            )
        written = self._associations.put_signatures(
            artifact.artifact_id,
            [
                (chunk.chunk_id, pooled[chunk.chunk_id])
                for chunk in pending
            ],
        )
        return {
            "requested": len(unique),
            "compiled": written,
            "reused": len(unique) - written,
            "compiled_spans": len(span_texts),
            "signature_width": len(artifact.concept_names),
            # Canonical invariant: no request-derived token IDs, Q/K/V,
            # attention maps, residuals, or generation K/V survive the pass.
            # Reusable checkpoint weights/tokenizer assets are not request
            # state and are deliberately outside this metric.
            "retained_request_token_state_bytes": 0,
            # Compatibility alias retained for old reports.
            "retained_token_state_bytes": 0,
        }

    def compile_indexed_cav_signatures(
        self,
        linker: object,
        artifact: AssociationArtifact,
        *,
        batch_size: int = 8,
        overwrite: bool = False,
        roles: Sequence[str] = ("user", "assistant", "system"),
    ) -> dict[str, int]:
        """Compile every active indexed chunk without hydrating embeddings."""

        selected_roles = tuple(dict.fromkeys(str(role) for role in roles))
        invalid_roles = set(selected_roles) - {"user", "assistant", "system"}
        if not selected_roles or invalid_roles:
            raise ValueError("roles must contain valid transcript roles")
        placeholders = ",".join("?" for _ in selected_roles)
        rows = self._db.execute(
            "SELECT c.chunk_id, c.turn_id, c.text, c.start_char, c.end_char, "
            "c.token_count FROM chunks AS c "
            "JOIN turns AS t ON t.turn_id = c.turn_id "
            "WHERE c.embedding IS NOT NULL AND c.hnsw_label IS NOT NULL "
            f"AND t.role IN ({placeholders}) ORDER BY c.hnsw_label",
            selected_roles,
        ).fetchall()
        chunks = [
            Chunk(
                chunk_id=str(row[0]),
                turn_id=str(row[1]),
                text=str(row[2]),
                start_char=int(row[3]),
                end_char=int(row[4]),
                token_count=int(row[5]),
            )
            for row in rows
        ]
        return self.compile_cav_signatures(
            linker,
            artifact,
            chunks,
            batch_size=batch_size,
            overwrite=overwrite,
        )

    def extract_memory(
        self, turns: list[Turn], chunks: list[Chunk] | None = None
    ) -> dict[str, int]:
        """Propose, validate, and apply memory ops for the given turns.

        Ops whose provenance cannot be verified against the transcript are
        rejected — an LLM cannot write a memory it did not quote.
        """
        ops = self._extractor.extract(turns, chunks)
        if ops.is_empty():
            return {}
        report = self._validator.validate(ops)
        return self._memory.apply(report)

    # -- retrieval ----------------------------------------------------------

    def search(
        self, query: str, k: int = 10, ef_search: int = 50
    ) -> list[RetrievalResult]:
        """Dense-only chunk search. This is the baseline retrieval path."""
        query_embedding = self._embedder.embed_query(query)
        return self._retriever.query(query_embedding, k=k, ef_search=ef_search)

    def search_hybrid(
        self,
        query: str,
        k: int = 10,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[RetrievalResult]:
        """Hybrid dense + BM25 chunk search, blended and reranked."""
        query_embedding = self._embedder.embed_query(query)
        return self.search_hybrid_from_embedding(
            query,
            query_embedding,
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )

    def search_hybrid_from_embedding(
        self,
        query: str,
        query_embedding: np.ndarray,
        *,
        k: int = 10,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[RetrievalResult]:
        """Run hybrid retrieval from a precomputed query embedding.

        Evaluation can batch query encoding across isolated sample stores,
        then preserve exactly the same hybrid ranking in each store.
        """
        return self._retriever.hybrid_query(
            query_text=query,
            query_embedding=np.asarray(query_embedding, dtype=np.float32),
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )

    def search_hybrid_many(
        self,
        queries: Sequence[str],
        *,
        k: int = 10,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[list[RetrievalResult]]:
        """Batch query embedding once, then run deterministic hybrid retrieval."""
        if not queries:
            return []
        embed_many = getattr(self._embedder, "embed_queries", None)
        if embed_many is None:
            embeddings = np.stack(
                [self._embedder.embed_query(query) for query in queries]
            )
        else:
            embeddings = embed_many(queries)
        return [
            self._retriever.hybrid_query(
                query_text=query,
                query_embedding=np.asarray(embedding, dtype=np.float32),
                k=k,
                ef_search=ef_search,
                candidates=candidates,
                alpha=alpha,
            )
            for query, embedding in zip(queries, embeddings, strict=True)
        ]

    def search_associative(
        self,
        query: str,
        artifact_id: str,
        *,
        k: int = 10,
        association_slots: int = 0,
        qk_reserve: int = 1,
        neighbors_per_anchor: int = 4,
        association_hops: int = 1,
        max_association_candidates: int = 64,
        cav_candidates: int = 8,
        lexical_protection_threshold: float | None = (
            SAFE_ASSOCIATION_LEXICAL_THRESHOLD
        ),
        max_prompt_token_increase: int | None = (
            SAFE_ASSOCIATION_MAX_TOKEN_INCREASE
        ),
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
        touch: bool = True,
    ) -> list[RetrievalResult]:
        """Expand hybrid anchors through external links under the same item cap.

        This method does not run Qwen and cannot grow transformer context.  It
        reads compact links previously emitted by bounded head-inspection
        passes, hydrates only the chosen chunk IDs, and returns at most ``k``
        ordinary retrieval results.  ``association_slots=0`` is conservative:
        association routes may fill only slots freed by duplicate anchors.
        A positive value explicitly trades that many direct slots for graph or
        CAV exploration while keeping the total retrieval budget fixed. By
        default, a near-maximum lexical tail anchor cannot be displaced and a
        composition that increases prompt tokens is rolled back before touch.
        """
        if k <= 0:
            return []
        anchors = self.search_hybrid(
            query,
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )
        return self.expand_associative(
            anchors,
            artifact_id,
            k=k,
            association_slots=association_slots,
            qk_reserve=qk_reserve,
            neighbors_per_anchor=neighbors_per_anchor,
            association_hops=association_hops,
            max_association_candidates=max_association_candidates,
            cav_candidates=cav_candidates,
            lexical_protection_threshold=lexical_protection_threshold,
            max_prompt_token_increase=max_prompt_token_increase,
            touch=touch,
        )

    def expand_associative(
        self,
        anchors: Sequence[RetrievalResult],
        artifact_id: str,
        *,
        k: int | None = None,
        association_slots: int = 0,
        qk_reserve: int = 1,
        neighbors_per_anchor: int = 4,
        association_hops: int = 1,
        max_association_candidates: int = 64,
        cav_candidates: int = 8,
        lexical_protection_threshold: float | None = (
            SAFE_ASSOCIATION_LEXICAL_THRESHOLD
        ),
        max_prompt_token_increase: int | None = (
            SAFE_ASSOCIATION_MAX_TOKEN_INCREASE
        ),
        touch: bool = True,
    ) -> list[RetrievalResult]:
        """Fan out cached hybrid anchors without embedding the query again."""
        return expand_associative_results(
            anchors,
            artifact_id,
            store=self._associations,
            hydrate=self._retriever.hydrate_chunk,
            now_turn=self._db.current_turn(),
            k=k,
            association_slots=association_slots,
            qk_reserve=qk_reserve,
            neighbors_per_anchor=neighbors_per_anchor,
            association_hops=association_hops,
            max_association_candidates=max_association_candidates,
            cav_candidates=cav_candidates,
            lexical_protection_threshold=lexical_protection_threshold,
            max_prompt_token_increase=max_prompt_token_increase,
            touch=touch,
        )

    def search_heat_associative(
        self,
        query: str,
        artifact_id: str,
        *,
        k: int = 10,
        association_slots: int = 1,
        qk_reserve: int = 1,
        ranked_qk_reserve: int = 0,
        neighbors_per_node: int = 3,
        diffusion_hops: int = 2,
        max_diffusion_nodes: int = 8,
        restart_probability: float = 0.35,
        seed_temperature: float = 1.0,
        edge_temperature: float = 1.0,
        lexical_protection_threshold: float | None = (
            SAFE_ASSOCIATION_LEXICAL_THRESHOLD
        ),
        max_prompt_token_increase: int | None = (
            SAFE_ASSOCIATION_MAX_TOKEN_INCREASE
        ),
        max_source_token_fraction: float = 1.0,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
        touch: bool = True,
    ) -> list[RetrievalResult]:
        """Diffuse compact head evidence, then expose memory by source heat.

        Qwen is not loaded here. The method carries only IDs, normalized scalar
        heat, and one explanatory path through a bounded external graph.
        """
        if k <= 0:
            return []
        anchors = self.search_hybrid(
            query,
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )
        return self.expand_heat_associative(
            anchors,
            artifact_id,
            k=k,
            association_slots=association_slots,
            qk_reserve=qk_reserve,
            ranked_qk_reserve=ranked_qk_reserve,
            neighbors_per_node=neighbors_per_node,
            diffusion_hops=diffusion_hops,
            max_diffusion_nodes=max_diffusion_nodes,
            restart_probability=restart_probability,
            seed_temperature=seed_temperature,
            edge_temperature=edge_temperature,
            lexical_protection_threshold=lexical_protection_threshold,
            max_prompt_token_increase=max_prompt_token_increase,
            max_source_token_fraction=max_source_token_fraction,
            touch=touch,
        )

    def expand_heat_associative(
        self,
        anchors: Sequence[RetrievalResult],
        artifact_id: str,
        *,
        k: int | None = None,
        association_slots: int = 1,
        qk_reserve: int = 1,
        ranked_qk_reserve: int = 0,
        neighbors_per_node: int = 3,
        diffusion_hops: int = 2,
        max_diffusion_nodes: int = 8,
        restart_probability: float = 0.35,
        seed_temperature: float = 1.0,
        edge_temperature: float = 1.0,
        lexical_protection_threshold: float | None = (
            SAFE_ASSOCIATION_LEXICAL_THRESHOLD
        ),
        max_prompt_token_increase: int | None = (
            SAFE_ASSOCIATION_MAX_TOKEN_INCREASE
        ),
        max_source_token_fraction: float = 1.0,
        touch: bool = True,
    ) -> list[RetrievalResult]:
        """Diffuse cached anchors without embedding the query or loading Qwen."""
        return expand_heat_diffusion_results(
            anchors,
            artifact_id,
            store=self._associations,
            hydrate=self._retriever.hydrate_chunk,
            now_turn=self._db.current_turn(),
            k=k,
            association_slots=association_slots,
            qk_reserve=qk_reserve,
            ranked_qk_reserve=ranked_qk_reserve,
            neighbors_per_node=neighbors_per_node,
            diffusion_hops=diffusion_hops,
            max_diffusion_nodes=max_diffusion_nodes,
            restart_probability=restart_probability,
            seed_temperature=seed_temperature,
            edge_temperature=edge_temperature,
            lexical_protection_threshold=lexical_protection_threshold,
            max_prompt_token_increase=max_prompt_token_increase,
            max_source_token_fraction=max_source_token_fraction,
            touch=touch,
        )

    def search_hebbian(
        self,
        query: str,
        artifact_id: str,
        *,
        k: int = 10,
        hebbian_slots: int = 1,
        max_candidates: int = 32,
        half_life_turns: float = 200.0,
        min_score: float = 0.05,
        lexical_protection_threshold: float | None = (
            SAFE_ASSOCIATION_LEXICAL_THRESHOLD
        ),
        max_prompt_token_increase: int | None = (
            SAFE_ASSOCIATION_MAX_TOKEN_INCREASE
        ),
        access_event_id: str | None = None,
        max_concepts_per_event: int = 12,
        max_degree: int = 32,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[RetrievalResult]:
        """Use learned same-turn co-access links inside a fixed result budget.

        Supplying ``access_event_id`` records the final returned set as one
        idempotent live access event. Omit it for evaluation reads that must not
        train the graph. The ID should identify the user turn or generation,
        not the query text itself.
        """
        if k <= 0:
            return []
        anchors = self.search_hybrid(
            query,
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )
        results = self.expand_hebbian(
            anchors,
            artifact_id,
            k=k,
            hebbian_slots=hebbian_slots,
            max_candidates=max_candidates,
            half_life_turns=half_life_turns,
            min_score=min_score,
            lexical_protection_threshold=lexical_protection_threshold,
            max_prompt_token_increase=max_prompt_token_increase,
        )
        if access_event_id is not None:
            self.observe_retrieval_access(
                results,
                artifact_id,
                access_event_id=access_event_id,
                half_life_turns=half_life_turns,
                max_concepts_per_event=max_concepts_per_event,
                max_degree=max_degree,
            )
        return results

    def expand_hebbian(
        self,
        anchors: Sequence[RetrievalResult],
        artifact_id: str,
        *,
        k: int | None = None,
        hebbian_slots: int = 1,
        max_candidates: int = 32,
        half_life_turns: float = 200.0,
        min_score: float = 0.05,
        lexical_protection_threshold: float | None = (
            SAFE_ASSOCIATION_LEXICAL_THRESHOLD
        ),
        max_prompt_token_increase: int | None = (
            SAFE_ASSOCIATION_MAX_TOKEN_INCREASE
        ),
    ) -> list[RetrievalResult]:
        """Expand cached anchors through live co-access links without an LLM."""
        return expand_hebbian_results(
            anchors,
            artifact_id,
            store=self._associations,
            hydrate=self._retriever.hydrate_chunk,
            now_turn=self._db.current_turn(),
            k=k,
            hebbian_slots=hebbian_slots,
            max_candidates=max_candidates,
            half_life_turns=half_life_turns,
            min_score=min_score,
            lexical_protection_threshold=lexical_protection_threshold,
            max_prompt_token_increase=max_prompt_token_increase,
        )

    def observe_retrieval_access(
        self,
        results: Sequence[RetrievalResult],
        artifact_id: str,
        *,
        access_event_id: str,
        now_turn: int | None = None,
        learning_rate: float = 1.0,
        half_life_turns: float = 200.0,
        max_concepts_per_event: int = 12,
        max_degree: int = 32,
        min_edge_score: float = 0.0,
        max_event_history: int = 4096,
    ) -> HebbianUpdate:
        """Reinforce conceptual chunks actually retrieved in the same turn."""
        activations = retrieval_concept_activations(
            results,
            max_concepts=max_concepts_per_event,
        )
        return self._associations.reinforce_retrieval_coaccess(
            artifact_id,
            access_event_id,
            activations,
            now_turn=now_turn,
            learning_rate=learning_rate,
            half_life_turns=half_life_turns,
            max_concepts_per_event=max_concepts_per_event,
            max_degree=max_degree,
            min_edge_score=min_edge_score,
            max_event_history=max_event_history,
        )

    def search_spans(
        self,
        query: str,
        levels: Sequence[int] = DEFAULT_SPAN_TOKENS,
        k_per_level: int = 2,
    ) -> list[RetrievalResult]:
        """Search pooled spans of contiguous chunks rather than single chunks.

        For short conversational turns this recovers most of what per-chunk
        search loses: a 27-token turn carries too little topical signal to be
        matched, while a ~110-220 token span carries enough. ``levels`` are
        token targets, so the same setting works on corpora whose turns differ
        by an order of magnitude in length. Returns the member chunks of the
        winning spans, so callers handle ordinary ``RetrievalResult``s.

        Not the default. It replicates on four LoCoMo samples (10.3% -> 23.4%
        answer containment, better on every sample), but ``search`` remains the
        baseline the eval ablations compare against, and nothing has yet
        measured this on the long-form corpus the original ablation used.
        """
        query_embedding = self._embedder.embed_query(query)
        return self._retriever.span_query(
            query_embedding, levels=levels, k_per_level=k_per_level
        )

    def search_sources(
        self,
        query: str,
        *,
        k_sources: int = 4,
    ) -> list[RetrievalResult]:
        """Retrieve complete source/session groups by pooled dense similarity."""

        query_embedding = self._embedder.embed_query(query)
        return self._retriever.source_query(query_embedding, k_sources=k_sources)

    def search_anchored_sources(
        self,
        query: str,
        *,
        k: int = 10,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[RetrievalResult]:
        """Select sources with hybrid anchors, then fairly expand each source."""
        anchors = self.search_hybrid(
            query,
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )
        source_ids: list[str] = []
        source_scores: dict[str, float] = {}
        for result in anchors:
            if result.turn is None:
                continue
            source_id = result.turn.source_id or result.turn.turn_id
            if source_id not in source_scores:
                source_ids.append(source_id)
                source_scores[source_id] = float(result.score)
            else:
                source_scores[source_id] = max(
                    source_scores[source_id], float(result.score)
                )
        return self._retriever.hydrate_sources(
            source_ids,
            source_scores=source_scores,
            interleave=True,
        )

    def search_hybrid_sources(
        self,
        query: str,
        *,
        k: int = 10,
        source_slots: int = 24,
        source_candidate_pool: int = 200,
        source_activation_k: int | None = None,
        source_local_search: bool = False,
        use_source_reranker: bool = False,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[RetrievalResult]:
        """Rerank a bounded pool inside sources activated by a ranked prefix.

        The top-k hybrid results remain the anchors.  Remaining slots can only
        be filled by lower-ranked candidates from a source represented in the
        independently bounded activation prefix. This reaches evidence
        elsewhere in a relevant conversation without hydrating whole sources
        or retaining request-derived model/token state.
        """
        self.last_source_rerank_report = {}
        if k <= 0:
            return []
        if source_slots < 0:
            raise ValueError("source_slots must be non-negative")
        if source_candidate_pool < k:
            raise ValueError("source_candidate_pool must be at least k")
        activation_k = k if source_activation_k is None else source_activation_k
        if activation_k < k or activation_k > source_candidate_pool:
            raise ValueError("source_activation_k must be between k and the pool")
        if use_source_reranker and not source_local_search:
            raise ValueError("source reranking requires source_local_search")

        query_embedding = self._embedder.embed_query(query)
        anchors = self.search_hybrid_from_embedding(
            query,
            query_embedding,
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )
        pool_size = max(k, source_candidate_pool)
        pool = self.search_hybrid_from_embedding(
            query,
            query_embedding,
            k=pool_size,
            ef_search=ef_search,
            candidates=max(candidates, pool_size),
            alpha=alpha,
        )
        if source_slots == 0 or not pool:
            return anchors

        anchor_by_source: dict[str, str] = {}
        source_scores: dict[str, float] = {}
        for result in pool[:activation_k]:
            if result.turn is None:
                continue
            source_id = result.turn.source_id or result.turn.turn_id
            anchor_by_source.setdefault(source_id, result.chunk.chunk_id)
            source_scores[source_id] = max(
                source_scores.get(source_id, 0.0), float(result.score)
            )

        anchor_ids = {result.chunk.chunk_id for result in anchors}
        if source_local_search:
            if use_source_reranker and self._source_candidate_reranker is None:
                raise RuntimeError("source reranking requested but no reranker is attached")
            result_limit = source_slots
            if use_source_reranker:
                result_limit = max(
                    result_limit,
                    self._source_candidate_reranker.candidate_pool,
                )
            local = self._retriever.hybrid_query_sources(
                query,
                query_embedding,
                list(anchor_by_source),
                k=result_limit,
                candidates_per_source=source_candidate_pool,
                alpha=alpha,
                source_scores=source_scores,
                anchor_chunk_ids=anchor_by_source,
                exclude_chunk_ids=tuple(anchor_ids),
            )
            if use_source_reranker:
                local = self._source_candidate_reranker.rerank(
                    query,
                    local,
                    top_k=source_slots,
                )
                report = self._source_candidate_reranker.last_report
                self.last_source_rerank_report = (
                    report.model_dump() if report is not None else {}
                )
            return [*anchors, *local]

        extras: list[RetrievalResult] = []
        for result in pool:
            if result.turn is None or result.chunk.chunk_id in anchor_ids:
                continue
            source_id = result.turn.source_id or result.turn.turn_id
            anchor_id = anchor_by_source.get(source_id)
            if anchor_id is None:
                continue
            extras.append(
                result.model_copy(
                    update={
                        "route": "hybrid_source",
                        "anchor_chunk_id": anchor_id,
                    }
                )
            )
            if len(extras) >= source_slots:
                break
        return list(anchors) + extras

    @staticmethod
    def _active_partition_surface_score(query: str, text: str) -> float:
        query_terms = set(tokenize(query))
        text_terms = set(tokenize(text))
        if not query_terms:
            return 0.0
        return len(query_terms & text_terms) / len(query_terms)

    @staticmethod
    def _first_person_completed_venue_occurrence(text: str) -> bool:
        """Conservative direct-event gate for canonical venue mentions."""

        first_person = re.search(r"\b(?:I|we|my|our)\b", text, re.IGNORECASE)
        completed = re.search(
            r"\b(?:visited|attended|participated|went|saw|took|returned|"
            r"came\s+back|got\s+back)\b",
            text,
            re.IGNORECASE,
        )
        return first_person is not None and completed is not None

    @staticmethod
    def _venue_episode_alignment(
        text: str,
        source_timestamp: str | None,
    ) -> bool | None:
        """Whether a stated venue occurrence belongs to its source episode.

        Immediate relative language is aligned by construction.  An explicit
        month/day is compared with the durable source date; a disagreement is
        a proved retrospective recap.  Rows without either signal remain
        ambiguous and are kept as alternatives rather than promoted to a
        structural primary.
        """

        relative_alignment = re.search(
            r"\b(?:today|tonight|yesterday|this\s+(?:morning|afternoon|evening)|"
            r"just\s+(?:came|got|returned)(?:\s+back)?)\b",
            text,
            re.IGNORECASE,
        ) is not None
        month_numbers = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        explicit = re.search(
            r"\b(?P<month>" + "|".join(month_numbers) + r")\s+"
            r"(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(?P<year>\d{4}))?\b",
            text,
            re.IGNORECASE,
        )
        if explicit is None:
            return True if relative_alignment else None
        from memory_condense.coverage_selector import _timestamp_key

        source_value = _timestamp_key(source_timestamp)
        if source_value is None:
            return None
        source_date = datetime.fromtimestamp(source_value)
        month = month_numbers[explicit.group("month").casefold()]
        day = int(explicit.group("day"))
        year = int(explicit.group("year") or source_date.year)
        try:
            event_date = datetime(year, month, day)
        except ValueError:
            return None
        return event_date.date() == source_date.date()

    def _active_partition_timestamps(
        self,
        source_ids: Sequence[str],
    ) -> dict[str, str]:
        timestamps: dict[str, str] = {}
        for start in range(0, len(source_ids), 400):
            metadata = self._transcript.source_metadata(
                list(source_ids[start : start + 400])
            )
            for source_id, text in metadata.items():
                parsed = parse_source_metadata(text)
                if parsed is not None:
                    timestamps[source_id] = parsed[1]
        return timestamps

    def _content_high_watermark(self) -> int:
        """Return the committed chunk generation used by scan snapshots."""

        row = self._db.execute(
            "SELECT COALESCE(MAX(rowid), 0) FROM chunks"
        ).fetchone()
        return int(row[0] if row is not None else 0)

    def _scan_active_partition_frontier(
        self,
        query: str,
        query_embedding: np.ndarray,
        partition_ids: Sequence[str],
        routed_source_ids: Sequence[str],
        *,
        separator: str,
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        """Reduce a complete selected-partition row scan to bounded IDs."""

        from memory_condense.coverage_selector import (
            SetQuantifier,
            QwenPrefixCoverageSelector,
            _canonical_answer_object_key,
            _timestamp_key,
            compile_set_program,
        )

        program = compile_set_program(query)
        venue_program = bool(
            program.requires_completeness
            and re.search(
                r"\b(?:museum|museums|gallery|galleries)\b",
                query,
                re.IGNORECASE,
            )
        )
        performance_program = bool(
            program.requires_completeness and is_performance_query(query)
        )
        partition_inventory = self._retriever.source_partition_ids(
            separator=separator,
        )
        selected_partition_ids = list(
            dict.fromkeys(str(value) for value in partition_ids if str(value))
        )
        partition_scope_exhaustive = bool(
            partition_inventory
            and set(selected_partition_ids) == set(partition_inventory)
        )
        base_report: dict[str, Any] = {
            "active_partition_scan_status": "bypassed",
            "active_partition_total": 0,
            "active_partition_inspected": 0,
            "active_partition_exhaustive": None,
            "active_partition_sources_total": len(routed_source_ids),
            "active_partition_sources_inspected": 0,
            "active_partition_structural_rows": 0,
            "active_partition_structural_hypotheses": 0,
            "active_partition_alternative_hypotheses": 0,
            "active_partition_ambiguous_structural_rows": 0,
            "active_partition_recap_conflict_rows": 0,
            "active_partition_performance_multirow_sources": 0,
            "active_partition_role_rejected_rows": 0,
            "active_partition_time_rejected_rows": 0,
            "active_partition_unknown_timestamp_rows": 0,
            "active_partition_candidates_admitted": 0,
            "active_partition_candidates_already_present": 0,
            "active_partition_candidates_replaced": 0,
            "active_partition_candidates_truncated": 0,
            "active_partition_structural_overflow": 0,
            "active_partition_scan_contract": "",
            "active_partition_semantically_complete": False,
            "partition_scope_kind": (
                "global" if partition_scope_exhaustive else "approximate_top_k"
            ),
            "partition_inventory_total": len(partition_inventory),
            "selected_partition_count": len(selected_partition_ids),
            "partition_scope_exhaustive": partition_scope_exhaustive,
            "selected_scope_structurally_complete": False,
            "global_semantic_complete": False,
            "active_partition_scan_elapsed_s": 0.0,
        }
        if not partition_ids or not (venue_program or performance_program):
            return [], base_report

        contract = (
            "canonical_venue_episode_aligned_v1"
            if venue_program
            else "direct_performance_source_occurrence_v1"
        )
        started = time.perf_counter()
        timestamps = self._active_partition_timestamps(routed_source_ids)
        # A source may contain more than one completed occurrence.  Transient
        # venue/performance keys separate those occurrences without becoming
        # stored state, and globally contract exact keyed recaps even when a
        # recap leaked into another routed source.
        primary_by_occurrence: dict[
            tuple[str, str], _ActivePartitionHypothesis
        ] = {}
        alternative_by_occurrence: dict[
            tuple[str, str], _ActivePartitionHypothesis
        ] = {}
        ambiguous_occurrences: set[tuple[str, str]] = set()
        inspected_sources: set[str] = set()
        total_rows = 0
        structural_rows = 0
        role_rejected = 0
        time_rejected = 0
        unknown_timestamp = 0
        ambiguous_rows = 0
        recap_conflicts = 0
        performance_occurrence_counts: Counter[str] = Counter()
        try:
            for row in self._retriever.iter_source_content_rows(
                routed_source_ids,
            ):
                total_rows += 1
                inspected_sources.add(row.source_id)
                canonical_key = (
                    _canonical_answer_object_key(query, row.text)
                    if venue_program
                    else performance_event_key(query, row.text)
                )
                typed_match = (
                    canonical_key is not None
                    and self._first_person_completed_venue_occurrence(row.text)
                    if venue_program
                    else is_direct_past_performance(query, row.text)
                )
                if not typed_match:
                    continue
                if (
                    program.preferred_evidence_role is not None
                    and row.role.casefold() != program.preferred_evidence_role
                ):
                    role_rejected += 1
                    continue
                timestamp = timestamps.get(row.source_id)
                temporal_in_scope = QwenPrefixCoverageSelector._timestamp_in_scope(
                    program,
                    timestamp,
                )
                if temporal_in_scope is False:
                    time_rejected += 1
                    continue
                if temporal_in_scope is None and (
                    program.query_timestamp is not None
                    or program.temporal_window_days is not None
                ):
                    unknown_timestamp += 1
                hypothesis = _ActivePartitionHypothesis(
                    chunk_id=row.chunk_id,
                    source_id=row.source_id,
                    timestamp=timestamp,
                    ordinal=row.ordinal,
                    surface_score=self._active_partition_surface_score(
                        query,
                        row.text,
                    ),
                    identity_key=(
                        str(canonical_key) if canonical_key is not None else None
                    ),
                )
                occurrence_key = (
                    (row.source_id, str(canonical_key))
                    if venue_program
                    else (
                        ("performance", str(canonical_key))
                        if canonical_key is not None
                        else (row.source_id, row.chunk_id)
                    )
                )
                if venue_program:
                    alignment = self._venue_episode_alignment(row.text, timestamp)
                    if alignment is False:
                        recap_conflicts += 1
                        alternative_by_occurrence.setdefault(
                            occurrence_key,
                            hypothesis,
                        )
                        continue
                    if alignment is None:
                        ambiguous_rows += 1
                        ambiguous_occurrences.add(occurrence_key)
                        alternative_by_occurrence.setdefault(
                            occurrence_key,
                            hypothesis,
                        )
                        continue
                else:
                    performance_occurrence_counts[row.source_id] += 1
                    if canonical_key is None:
                        ambiguous_rows += 1
                        ambiguous_occurrences.add(occurrence_key)
                        alternative_by_occurrence.setdefault(
                            occurrence_key,
                            hypothesis,
                        )
                        continue
                structural_rows += 1
                existing = primary_by_occurrence.get(occurrence_key)
                if existing is None:
                    primary_by_occurrence[occurrence_key] = hypothesis
                else:
                    existing_timestamp = _timestamp_key(existing.timestamp)
                    hypothesis_timestamp = _timestamp_key(hypothesis.timestamp)
                    existing_order = (
                        existing_timestamp is None,
                        existing_timestamp or 0.0,
                        existing.ordinal,
                        existing.chunk_id,
                    )
                    hypothesis_order = (
                        hypothesis_timestamp is None,
                        hypothesis_timestamp or 0.0,
                        hypothesis.ordinal,
                        hypothesis.chunk_id,
                    )
                    if hypothesis_order < existing_order:
                        primary_by_occurrence[occurrence_key] = hypothesis
        except Exception as exc:
            base_report.update(
                {
                    "active_partition_scan_status": "failed",
                    "active_partition_total": total_rows,
                    "active_partition_inspected": total_rows,
                    "active_partition_exhaustive": False,
                    "active_partition_sources_inspected": len(inspected_sources),
                    "active_partition_scan_contract": contract,
                    "active_partition_scan_error": type(exc).__name__,
                    "active_partition_scan_elapsed_s": (
                        time.perf_counter() - started
                    ),
                }
            )
            return [], base_report

        primary_occurrences = list(primary_by_occurrence.values())
        alternative_occurrences = [
            hypothesis
            for occurrence_key, hypothesis in alternative_by_occurrence.items()
            if occurrence_key not in primary_by_occurrence
        ]
        if venue_program:
            def venue_occurrence_order(
                hypothesis: _ActivePartitionHypothesis,
            ) -> tuple[bool, float, int, str]:
                timestamp_key = _timestamp_key(hypothesis.timestamp)
                return (
                    timestamp_key is None,
                    timestamp_key or 0.0,
                    hypothesis.ordinal,
                    hypothesis.chunk_id,
                )

            primary_by_identity: dict[str, _ActivePartitionHypothesis] = {}
            for hypothesis in sorted(
                primary_occurrences,
                key=venue_occurrence_order,
            ):
                if hypothesis.identity_key is not None:
                    primary_by_identity.setdefault(
                        hypothesis.identity_key,
                        hypothesis,
                    )
            primary_hypotheses = list(primary_by_identity.values())
            primary_identity_keys = set(primary_by_identity)
            alternative_by_identity: dict[
                str, _ActivePartitionHypothesis
            ] = {}
            for hypothesis in sorted(
                alternative_occurrences,
                key=venue_occurrence_order,
            ):
                identity_key = hypothesis.identity_key
                if (
                    identity_key is not None
                    and identity_key not in primary_identity_keys
                ):
                    alternative_by_identity.setdefault(identity_key, hypothesis)
            alternatives = list(alternative_by_identity.values())
            ambiguous_identity_keys = {
                identity_key
                for _source_id, identity_key in ambiguous_occurrences
                if identity_key not in primary_identity_keys
            }
            ambiguous_hypotheses = len(ambiguous_identity_keys)
        else:
            primary_hypotheses = primary_occurrences
            alternatives = alternative_occurrences
            ambiguous_hypotheses = len(ambiguous_occurrences)
        hypothesis_count = len(primary_hypotheses)
        overflow = (
            max(0, hypothesis_count - int(program.cardinality or 0))
            if program.quantifier is SetQuantifier.FIXED
            else 0
        )
        def hypothesis_rank(
            item: _ActivePartitionHypothesis,
        ) -> tuple[float, int, str]:
            return (
                -item.surface_score,
                item.ordinal,
                item.chunk_id,
            )

        pre_ranked = [
            *sorted(primary_hypotheses, key=hypothesis_rank),
            *sorted(alternatives, key=hypothesis_rank),
        ]
        cap_truncated = max(
            0,
            len(pre_ranked) - _ACTIVE_PARTITION_HYPOTHESIS_CAP,
        )
        retained = pre_ranked[:_ACTIVE_PARTITION_HYPOTHESIS_CAP]
        dense_scores = self._retriever.cosine_scores(
            query_embedding,
            [item.chunk_id for item in retained],
        )
        primary_ids = {item.chunk_id for item in primary_hypotheses}
        retained.sort(
            key=lambda item: (
                item.chunk_id not in primary_ids,
                -dense_scores.get(item.chunk_id, float("-inf")),
                -item.surface_score,
                item.ordinal,
                item.chunk_id,
            )
        )
        candidates: list[RetrievalResult] = []
        for hypothesis in retained:
            score = dense_scores.get(hypothesis.chunk_id, 0.0)
            hydrated = self._retriever.hydrate_chunk(
                hypothesis.chunk_id,
                score=score,
                route=(
                    "active_partition_structural"
                    if hypothesis.chunk_id in primary_ids
                    else "active_partition_alternative"
                ),
                anchor_chunk_id=hypothesis.chunk_id,
            )
            if hydrated is None:
                cap_truncated += 1
                continue
            candidates.append(
                hydrated.model_copy(
                    update={
                        "memory_source_id": hypothesis.source_id,
                        "source_heat": max(0.0, score),
                    }
                )
            )

        fixed_complete = bool(
            program.quantifier is SetQuantifier.FIXED
            and program.cardinality is not None
            and hypothesis_count == program.cardinality
        )
        exhaustive_set_complete = bool(
            performance_program
            and program.quantifier in {SetQuantifier.ALL, SetQuantifier.COUNT}
        )
        performance_multirow_sources = sum(
            count > 1 for count in performance_occurrence_counts.values()
        )
        semantic_complete = bool(
            cap_truncated == 0
            and unknown_timestamp == 0
            and ambiguous_hypotheses == 0
            and (fixed_complete or exhaustive_set_complete)
        )
        base_report.update(
            {
                "active_partition_scan_status": "applied",
                "active_partition_total": total_rows,
                "active_partition_inspected": total_rows,
                "active_partition_exhaustive": True,
                "active_partition_sources_inspected": len(routed_source_ids),
                "active_partition_structural_rows": structural_rows,
                "active_partition_structural_hypotheses": hypothesis_count,
                "active_partition_alternative_hypotheses": len(alternatives),
                "active_partition_ambiguous_structural_rows": ambiguous_rows,
                "active_partition_recap_conflict_rows": recap_conflicts,
                "active_partition_performance_multirow_sources": (
                    performance_multirow_sources
                ),
                "active_partition_role_rejected_rows": role_rejected,
                "active_partition_time_rejected_rows": time_rejected,
                "active_partition_unknown_timestamp_rows": unknown_timestamp,
                "active_partition_candidates_truncated": cap_truncated,
                "active_partition_structural_overflow": overflow,
                "active_partition_scan_contract": contract,
                "active_partition_semantically_complete": semantic_complete,
                "selected_scope_structurally_complete": semantic_complete,
                "global_semantic_complete": bool(
                    partition_scope_exhaustive and semantic_complete
                ),
                "active_partition_scan_elapsed_s": time.perf_counter() - started,
            }
        )
        return candidates, base_report

    @staticmethod
    def _admit_active_partition_candidates(
        baseline: Sequence[RetrievalResult],
        candidates: Sequence[RetrievalResult],
        *,
        anchor_chunk_ids: set[str],
        semantic_complete: bool = True,
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        """Force typed hypotheses into a fixed-count frontier.

        A proved-complete typed scan may consume the full frontier.  An
        ambiguous, overflowing, or truncated scan remains fail-open: direct
        anchors are immutable and at least one quarter of the baseline is
        reserved before bounded typed additions are considered.
        """

        capacity = len(baseline)
        baseline_ids = {result.chunk.chunk_id for result in baseline}
        unique_candidates: list[RetrievalResult] = []
        candidate_ids: set[str] = set()
        for result in candidates:
            chunk_id = result.chunk.chunk_id
            if chunk_id in candidate_ids:
                continue
            candidate_ids.add(chunk_id)
            unique_candidates.append(result)

        protected_ids: set[str] = set()
        if not semantic_complete and capacity:
            protected_ids.update(baseline_ids & anchor_chunk_ids)
            reserve = max(1, math.ceil(capacity / 4))
            for result in baseline:
                if len(protected_ids) >= reserve:
                    break
                protected_ids.add(result.chunk.chunk_id)
        evictable_ids = baseline_ids - protected_ids
        existing_candidate_ids = baseline_ids & candidate_ids
        new_candidate_budget = (
            capacity
            if semantic_complete
            else len(evictable_ids - existing_candidate_ids)
        )
        retained: list[RetrievalResult] = []
        admitted_count = 0
        for result in unique_candidates:
            chunk_id = result.chunk.chunk_id
            if chunk_id in baseline_ids:
                retained.append(result)
            elif admitted_count < new_candidate_budget:
                retained.append(result)
                admitted_count += 1
            if len(retained) >= capacity:
                break
        capacity_truncated = max(0, len(unique_candidates) - len(retained))
        retained_ids = {result.chunk.chunk_id for result in retained}
        already_present = len(retained_ids & baseline_ids)
        admitted = len(retained_ids - baseline_ids)
        ordinary = [
            (index, result)
            for index, result in enumerate(baseline)
            if result.chunk.chunk_id not in retained_ids
        ]

        def eviction_key(
            item: tuple[int, RetrievalResult],
        ) -> tuple[int, float, int, str]:
            index, result = item
            route = (result.route or "").casefold()
            if route == "hsc_contraction":
                tier = 0
            elif route == "live_consolidation":
                tier = 1
            elif "neighbor" in route:
                tier = 2
            elif result.chunk.chunk_id not in anchor_chunk_ids:
                tier = 3
            else:
                tier = 4
            return tier, float(result.score), -index, result.chunk.chunk_id

        evict_count = min(admitted, len(ordinary))
        evicted_ids = {
            result.chunk.chunk_id
            for _index, result in sorted(
                (
                    item
                    for item in ordinary
                    if item[1].chunk.chunk_id not in protected_ids
                ),
                key=eviction_key,
            )[:evict_count]
        }
        survivors = [
            result
            for _index, result in ordinary
            if result.chunk.chunk_id not in evicted_ids
        ]
        output = [*retained, *survivors]
        if len(output) > capacity:
            output = output[:capacity]
        return output, {
            "active_partition_candidates_admitted": admitted,
            "active_partition_candidates_already_present": already_present,
            "active_partition_candidates_replaced": len(evicted_ids),
            "active_partition_candidates_truncated": capacity_truncated,
            "active_partition_baseline_protected": len(protected_ids),
            "active_partition_candidate_count_before": capacity,
            "active_partition_candidate_count_after": len(output),
        }

    def search_hybrid_graph(
        self,
        query: str,
        *,
        k: int = 10,
        neighbor_radius: int = 5,
        neighbor_slots: int = 24,
        neighbor_direction: Literal["both", "previous", "next"] = "next",
        source_slots: int = 24,
        source_candidate_pool: int = 200,
        source_activation_k: int | None = None,
        query_facet_retrieval: bool = False,
        query_facet_slots: int = 6,
        query_facet_max: int = 4,
        role_aware_retrieval: bool = False,
        role_user_weight: float = 1.25,
        role_assistant_weight: float = 0.75,
        role_system_weight: float = 0.50,
        multi_fact_source_diversity: bool = False,
        source_tfisf_activation: bool = False,
        source_tfisf_slots: int = 8,
        source_hsc_activation: bool = False,
        source_hsc_slots: int = 8,
        source_hsc_hops: int = 2,
        source_hsc_chunk_slots: int = 8,
        source_partition_routing: bool = False,
        source_partition_slots: int = 3,
        source_partition_separator: str = "::",
        source_local_search: bool = False,
        use_source_reranker: bool = False,
        use_attention_feedback: bool = False,
        feedback_slots: int = 12,
        feedback_seed_slots: int = 6,
        feedback_evidence_tokens: int = 48,
        feedback_query_tokens: int = 384,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[RetrievalResult]:
        """Union transition and source links behind immutable hybrid anchors.

        Results are ordered as anchors, bounded source-local transitions, then
        lower-ranked candidates from activated sources.  The caller's prompt
        cap remains the final hard byte/token boundary.
        """
        self.last_source_rerank_report = {}
        self._active_partition_routing_snapshot = None
        if k <= 0:
            return []
        if neighbor_radius < 0 or neighbor_slots < 0 or source_slots < 0:
            raise ValueError("graph retrieval bounds must be non-negative")
        if use_attention_feedback and (
            feedback_slots < 0 or feedback_slots > source_slots
        ):
            raise ValueError("feedback_slots must lie in [0, source_slots]")
        if feedback_seed_slots < 1:
            raise ValueError("feedback_seed_slots must be positive")
        if feedback_evidence_tokens < 1 or feedback_query_tokens < 1:
            raise ValueError("feedback token caps must be positive")
        if neighbor_direction not in {"both", "previous", "next"}:
            raise ValueError("invalid neighbor_direction")
        if source_candidate_pool < k:
            raise ValueError("source_candidate_pool must be at least k")
        if query_facet_slots < 0 or (
            query_facet_retrieval and query_facet_slots > source_slots
        ):
            raise ValueError("query_facet_slots must lie in [0, source_slots]")
        if query_facet_max < 1:
            raise ValueError("query_facet_max must be positive")
        if min(role_user_weight, role_assistant_weight, role_system_weight) < 0.0:
            raise ValueError("role weights must be non-negative")
        if source_tfisf_slots < 1:
            raise ValueError("source_tfisf_slots must be positive")
        if source_hsc_slots < 1 or source_hsc_hops < 1:
            raise ValueError("source HSC slots and hops must be positive")
        if source_hsc_activation and (
            source_hsc_chunk_slots < 1 or source_hsc_chunk_slots > source_slots
        ):
            raise ValueError("source_hsc_chunk_slots must lie in [1, source_slots]")
        if (
            query_facet_retrieval
            and source_hsc_activation
            and query_facet_slots + source_hsc_chunk_slots > source_slots
        ):
            raise ValueError("facet and HSC reserves cannot exceed source_slots")
        activation_k = k if source_activation_k is None else source_activation_k
        if activation_k < k or activation_k > source_candidate_pool:
            raise ValueError("source_activation_k must be between k and the pool")
        if use_source_reranker and not source_local_search:
            raise ValueError("source reranking requires source_local_search")
        if use_attention_feedback and not source_local_search:
            raise ValueError("attention feedback requires source_local_search")
        if use_source_reranker and use_attention_feedback:
            raise ValueError("reranking and attention feedback are separate arms")
        if (
            use_attention_feedback
            and self._source_candidate_reranker is None
        ):
            raise RuntimeError(
                "attention feedback requested but no Qwen controller is attached"
            )
        if source_partition_routing and source_partition_slots < 1:
            raise ValueError("source_partition_slots must be positive")
        if source_partition_routing and not source_partition_separator:
            raise ValueError("source_partition_separator must be non-empty")

        self.last_partition_routing_report = {}
        query_embedding = self._embedder.embed_query(query)
        pool_size = max(k, source_candidate_pool)
        routed_source_ids: list[str] | None = None
        active_partition_routing_turn: int | None = None
        active_partition_content_high_watermark: int | None = None
        partition_ids: list[str] = []
        active_partition_candidates: list[RetrievalResult] = []
        active_partition_report: dict[str, Any] = {}
        if source_partition_routing:
            coarse_pool = self.search_hybrid_from_embedding(
                query,
                query_embedding,
                k=pool_size,
                ef_search=ef_search,
                candidates=max(candidates, pool_size),
                alpha=alpha,
            )
            if role_aware_retrieval:
                coarse_pool = role_aware_results(
                    query,
                    coarse_pool,
                    user_weight=role_user_weight,
                    assistant_weight=role_assistant_weight,
                    system_weight=role_system_weight,
                )
            partition_ranking = source_partition_ranking(
                coarse_pool,
                separator=source_partition_separator,
            )
            partition_ids = [
                str(item["partition"])
                for item in partition_ranking[:source_partition_slots]
            ]
            # Bind the complete scan to the database generation immediately
            # before resolving its immutable source set.  A concurrent append
            # anywhere in the remainder of the route invalidates completeness
            # instead of allowing a stale scan report to reach the packer.
            active_partition_routing_turn = self._transcript.current_turn()
            active_partition_content_high_watermark = (
                self._content_high_watermark()
            )
            routed_source_ids = self._retriever.source_ids_in_partitions(
                partition_ids,
                separator=source_partition_separator,
            )
            (
                active_partition_candidates,
                active_partition_report,
            ) = self._scan_active_partition_frontier(
                query,
                query_embedding,
                partition_ids,
                routed_source_ids,
                separator=source_partition_separator,
            )
            pool = self._retriever.hybrid_query_sources(
                query,
                query_embedding,
                routed_source_ids,
                k=pool_size,
                candidates_per_source=source_candidate_pool,
                alpha=alpha,
            )
            self.last_partition_routing_report = {
                "coarse_candidates": len(coarse_pool),
                "selected_partitions": partition_ids,
                "partition_ranking": partition_ranking,
                "routed_sources": len(routed_source_ids),
                "routed_candidates": len(pool),
                **active_partition_report,
            }
            anchors = pool[:k]
        else:
            anchors = self.search_hybrid_from_embedding(
                query,
                query_embedding,
                k=k,
                ef_search=ef_search,
                candidates=candidates,
                alpha=alpha,
            )
            pool = self.search_hybrid_from_embedding(
                query,
                query_embedding,
                k=pool_size,
                ef_search=ef_search,
                candidates=max(candidates, pool_size),
                alpha=alpha,
            )
        if role_aware_retrieval:
            pool = role_aware_results(
                query,
                pool,
                user_weight=role_user_weight,
                assistant_weight=role_assistant_weight,
                system_weight=role_system_weight,
            )
            anchors = pool[:k]
        use_source_diversity = (
            multi_fact_source_diversity and is_multi_fact_query(query)
        )
        if use_source_diversity:
            pool = source_diverse_results(pool)
            anchors = pool[:k]
        self.last_source_diversity_report = {
            "enabled": multi_fact_source_diversity,
            "applied": use_source_diversity,
            "activated_prefix_sources": len(
                {
                    str(result.turn.source_id or result.turn.turn_id)
                    for result in pool[:activation_k]
                    if result.turn is not None
                }
            ),
        }

        concept_results: list[RetrievalResult] = []
        concept_member_results: list[RetrievalResult] = []
        concept_name: str | None = None
        if (
            use_source_diversity
            and routed_source_ids
            and self._source_concept_artifact_id is not None
        ):
            artifact = self._associations.get_artifact(
                self._source_concept_artifact_id
            )
            if artifact is not None and artifact.concept_names:
                concept_name = (
                    "autobiographical_completed_event"
                    if "autobiographical_completed_event" in artifact.concept_names
                    else artifact.concept_names[0]
                )
                member_hits = self._associations.concept_members(
                    artifact.artifact_id,
                    concept_name,
                    top_k=max(source_slots * 4, source_slots),
                    source_ids=routed_source_ids,
                    unique_sources=True,
                )
                concept_member_results = [
                    hydrated.model_copy(update={"route": "cav_concept_member"})
                    for hit in member_hits
                    if (
                        hydrated := self._retriever.hydrate_chunk(
                            hit.chunk_id,
                            score=hit.score,
                            route="cav_concept_member",
                        )
                    )
                    is not None
                ]
                concept_source_ids = [
                    hit.source_id for hit in member_hits if hit.source_id is not None
                ]
                if concept_source_ids:
                    concept_results = self._retriever.hybrid_query_sources(
                        query,
                        query_embedding,
                        concept_source_ids,
                        k=max(source_slots, k),
                        candidates_per_source=source_candidate_pool,
                        alpha=alpha,
                    )
                    if role_aware_retrieval:
                        concept_results = role_aware_results(
                            query,
                            concept_results,
                            user_weight=role_user_weight,
                            assistant_weight=role_assistant_weight,
                            system_weight=role_system_weight,
                        )
                    concept_results = source_diverse_results(concept_results)
        self.last_source_diversity_report.update(
            {
                "concept_activation": concept_name,
                "concept_candidates": len(concept_results),
                "concept_members": len(concept_member_results),
            }
        )

        facets = (
            query_facets(query, max_facets=query_facet_max)
            if query_facet_retrieval and query_facet_slots
            else []
        )
        facet_groups: list[list[RetrievalResult]] = []
        if facets:
            per_facet = max(
                2,
                2 * ((query_facet_slots + len(facets) - 1) // len(facets)),
            )
            for facet in facets:
                facet_embedding = self._embedder.embed_query(facet)
                if routed_source_ids is not None:
                    facet_pool = self._retriever.hybrid_query_sources(
                        facet,
                        facet_embedding,
                        routed_source_ids,
                        k=per_facet,
                        candidates_per_source=source_candidate_pool,
                        alpha=alpha,
                    )
                else:
                    facet_pool = self.search_hybrid_from_embedding(
                        facet,
                        facet_embedding,
                        k=per_facet,
                        ef_search=ef_search,
                        candidates=max(candidates, per_facet),
                        alpha=alpha,
                    )
                if role_aware_retrieval:
                    facet_pool = role_aware_results(
                        facet,
                        facet_pool,
                        user_weight=role_user_weight,
                        assistant_weight=role_assistant_weight,
                        system_weight=role_system_weight,
                    )
                facet_groups.append(facet_pool)

        facet_results: list[RetrievalResult] = []
        facet_seen = {result.chunk.chunk_id for result in anchors}
        position = 0
        while facet_groups and len(facet_results) < query_facet_slots:
            added = False
            for group in facet_groups:
                if position >= len(group):
                    continue
                result = group[position]
                if result.chunk.chunk_id in facet_seen:
                    continue
                facet_seen.add(result.chunk.chunk_id)
                facet_results.append(result.model_copy(update={"route": "query_facet"}))
                added = True
                if len(facet_results) >= query_facet_slots:
                    break
            if not added and all(position >= len(group) - 1 for group in facet_groups):
                break
            position += 1
        self.last_query_facet_report = {
            "enabled": query_facet_retrieval,
            "facets": len(facets),
            "reserved_slots": query_facet_slots if facets else 0,
            "candidates_added": len(facet_results),
        }
        expanded = self.expand_source_neighbors(
            anchors,
            radius=neighbor_radius,
        )
        facet_ids = {result.chunk.chunk_id for result in facet_results}
        neighbors = [
            result
            for result in expanded[len(anchors) :]
            if result.chunk.chunk_id not in facet_ids
            and (
                neighbor_direction == "both"
                or result.transition_direction == neighbor_direction
            )
        ][:neighbor_slots]

        anchor_by_source: dict[str, str] = {}
        source_scores: dict[str, float] = {}
        first_pool_result_by_source: dict[str, RetrievalResult] = {}
        for result in [*pool, *facet_results, *concept_results]:
            if result.turn is None:
                continue
            source_id = str(result.turn.source_id or result.turn.turn_id)
            first_pool_result_by_source.setdefault(source_id, result)
        for result in pool[:activation_k]:
            if result.turn is not None:
                source_id = str(result.turn.source_id or result.turn.turn_id)
                anchor_by_source.setdefault(source_id, result.chunk.chunk_id)
                source_scores[source_id] = max(
                    source_scores.get(source_id, 0.0), float(result.score)
                )
        for result in facet_results:
            if result.turn is None:
                continue
            source_id = str(result.turn.source_id or result.turn.turn_id)
            anchor_by_source.setdefault(source_id, result.chunk.chunk_id)
            source_scores[source_id] = max(
                source_scores.get(source_id, 0.0), float(result.score)
            )
        for result in concept_results:
            if result.turn is None:
                continue
            source_id = str(result.turn.source_id or result.turn.turn_id)
            anchor_by_source.setdefault(source_id, result.chunk.chunk_id)
            source_scores[source_id] = max(
                source_scores.get(source_id, 0.0), float(result.score)
            )

        tfisf_ranked = (
            self._retriever.source_tfisf_query(
                query,
                k_sources=source_tfisf_slots,
            )
            if source_tfisf_activation
            else []
        )
        tfisf_admitted: list[str] = []
        for source_id, source_score in tfisf_ranked:
            candidate = first_pool_result_by_source.get(source_id)
            if candidate is None:
                continue
            if source_id not in anchor_by_source:
                tfisf_admitted.append(source_id)
            anchor_by_source.setdefault(source_id, candidate.chunk.chunk_id)
            source_scores[source_id] = max(
                source_scores.get(source_id, 0.0), float(source_score)
            )
        self.last_source_tfisf_report = {
            "enabled": source_tfisf_activation,
            "ranked_sources": [source_id for source_id, _score in tfisf_ranked],
            "admitted_sources": tfisf_admitted,
            "activated_sources": len(anchor_by_source),
        }

        hsc_ranked = (
            self._retriever.source_hsc_expand(
                query_embedding,
                list(anchor_by_source),
                slots=source_hsc_slots,
                hops=source_hsc_hops,
            )
            if source_hsc_activation
            else []
        )
        hsc_source_ids = [source_id for source_id, _score in hsc_ranked]
        hsc_source_scores = dict(hsc_ranked)
        hsc_anchor_by_source = {
            source_id: first_pool_result_by_source[source_id].chunk.chunk_id
            for source_id in hsc_source_ids
            if source_id in first_pool_result_by_source
        }
        self.last_source_hsc_report = {
            "enabled": source_hsc_activation,
            "seed_sources": len(anchor_by_source),
            "expanded_sources": hsc_source_ids,
            "pool_visible_sources": len(hsc_anchor_by_source),
            "reserved_chunk_slots": (
                source_hsc_chunk_slots if source_hsc_activation else 0
            ),
        }

        seen = {
            result.chunk.chunk_id
            for result in [*anchors, *facet_results, *neighbors]
        }
        regular_source_slots = source_slots - len(facet_results) - (
            source_hsc_chunk_slots if source_hsc_activation else 0
        )
        concept_partition_results: list[RetrievalResult] = []
        concept_coverage_reserve = 0
        if (
            concept_name is not None
            and self._source_concept_artifact_id is not None
            and partition_ids
            and regular_source_slots > 0
        ):
            concept_coverage_reserve = min(
                regular_source_slots,
                max(2, math.ceil(regular_source_slots * 0.55)),
            )
            first_quota = math.ceil(concept_coverage_reserve * 2.0 / 3.0)
            partition_quotas = (
                first_quota,
                concept_coverage_reserve - first_quota,
            )
            for partition_id, quota in zip(
                partition_ids[:2], partition_quotas, strict=False
            ):
                if quota <= 0:
                    continue
                partition_sources = self._retriever.source_ids_in_partitions(
                    [partition_id],
                    separator=source_partition_separator,
                )
                hits = self._associations.concept_members(
                    self._source_concept_artifact_id,
                    concept_name,
                    top_k=max(quota, len(partition_sources)),
                    source_ids=partition_sources,
                    unique_sources=True,
                )
                partition_members: list[RetrievalResult] = []
                for hit in hits:
                    hydrated = self._retriever.hydrate_chunk(
                        hit.chunk_id,
                        score=hit.score,
                        route="cav_partition_coverage",
                    )
                    if hydrated is not None:
                        partition_members.append(hydrated)
                concept_partition_results.extend(
                    rank_concept_members(query, partition_members)[:quota]
                )
            self.last_source_diversity_report.update(
                {
                    "concept_coverage_reserve": concept_coverage_reserve,
                    "concept_partition_candidates": len(
                        concept_partition_results
                    ),
                }
            )
        if source_local_search:
            if use_source_reranker and self._source_candidate_reranker is None:
                raise RuntimeError("source reranking requested but no reranker is attached")
            result_limit = regular_source_slots
            if use_source_reranker:
                result_limit = max(
                    result_limit,
                    self._source_candidate_reranker.candidate_pool,
                )
            source_extras = self._retriever.hybrid_query_sources(
                query,
                query_embedding,
                list(anchor_by_source),
                k=result_limit,
                candidates_per_source=source_candidate_pool,
                alpha=alpha,
                source_scores=source_scores,
                anchor_chunk_ids=anchor_by_source,
                exclude_chunk_ids=tuple(seen),
            )
            if role_aware_retrieval:
                source_extras = role_aware_results(
                    query,
                    source_extras,
                    user_weight=role_user_weight,
                    assistant_weight=role_assistant_weight,
                    system_weight=role_system_weight,
                )
            if use_source_diversity:
                source_extras = source_diverse_results(source_extras)
            if use_source_reranker:
                rerank_candidates = source_extras
                if use_source_diversity:
                    # Set queries need a safe scalar control plus attention
                    # exploration. Build the control from the narrower source
                    # frontier that supplies the output budget and auxiliary
                    # source routes; broader activation is visible only to the
                    # Qwen reserve and cannot evict this protected prefix.
                    protected_activation_k = min(
                        activation_k,
                        source_slots
                        + (source_tfisf_slots if source_tfisf_activation else 0)
                        + (source_hsc_slots if source_hsc_activation else 0)
                        + 1,
                    )
                    protected_anchor_by_source: dict[str, str] = {}
                    protected_source_scores: dict[str, float] = {}
                    for result in pool[:protected_activation_k]:
                        if result.turn is None:
                            continue
                        source_id = str(
                            result.turn.source_id or result.turn.turn_id
                        )
                        protected_anchor_by_source.setdefault(
                            source_id, result.chunk.chunk_id
                        )
                        protected_source_scores[source_id] = max(
                            protected_source_scores.get(source_id, 0.0),
                            float(result.score),
                        )
                    protected_extras = self._retriever.hybrid_query_sources(
                        query,
                        query_embedding,
                        list(protected_anchor_by_source),
                        k=regular_source_slots,
                        candidates_per_source=source_candidate_pool,
                        alpha=alpha,
                        source_scores=protected_source_scores,
                        anchor_chunk_ids=protected_anchor_by_source,
                        exclude_chunk_ids=tuple(seen),
                    )
                    if role_aware_retrieval:
                        protected_extras = role_aware_results(
                            query,
                            protected_extras,
                            user_weight=role_user_weight,
                            assistant_weight=role_assistant_weight,
                            system_weight=role_system_weight,
                        )
                    protected_extras = source_diverse_results(protected_extras)
                    protected_ids = {
                        result.chunk.chunk_id for result in protected_extras
                    }
                    concept_member_ids = {
                        result.chunk.chunk_id for result in concept_member_results
                    }
                    rerank_candidates = [
                        *protected_extras,
                        *concept_member_results,
                        *[
                            result
                            for result in source_extras
                            if result.chunk.chunk_id not in protected_ids
                            and result.chunk.chunk_id not in concept_member_ids
                        ],
                    ]
                    self.last_source_diversity_report.update(
                        {
                            "protected_activation_k": protected_activation_k,
                            "protected_candidates": len(protected_extras),
                            "attention_exploration_candidates": max(
                                0,
                                len(rerank_candidates) - len(protected_extras),
                            ),
                        }
                    )
                source_extras = self._source_candidate_reranker.rerank(
                    query,
                    rerank_candidates,
                    top_k=regular_source_slots,
                    unique_sources=use_source_diversity,
                )
                if use_source_diversity and concept_partition_results:
                    scalar_reserve = max(
                        0,
                        regular_source_slots - concept_coverage_reserve,
                    )
                    combined = [
                        *protected_extras[:scalar_reserve],
                        *concept_partition_results,
                        *source_extras,
                    ]
                    source_unique: list[RetrievalResult] = []
                    seen_sources: set[str] = set()
                    for result in combined:
                        source_id = str(
                            result.memory_source_id
                            or (
                                result.turn.source_id
                                if result.turn is not None
                                else None
                            )
                            or result.chunk.turn_id
                        )
                        if source_id in seen_sources:
                            continue
                        seen_sources.add(source_id)
                        source_unique.append(result)
                        if len(source_unique) >= regular_source_slots:
                            break
                    source_extras = source_unique
                report = self._source_candidate_reranker.last_report
                self.last_source_rerank_report = (
                    report.model_dump() if report is not None else {}
                )
        else:
            source_extras = []
            for result in pool:
                if regular_source_slots == 0:
                    break
                if result.turn is None or result.chunk.chunk_id in seen:
                    continue
                source_id = result.turn.source_id or result.turn.turn_id
                anchor_id = anchor_by_source.get(source_id)
                if anchor_id is None:
                    continue
                source_extras.append(
                    result.model_copy(
                        update={
                            "route": "hybrid_source",
                            "anchor_chunk_id": anchor_id,
                        }
                    )
                )
                seen.add(result.chunk.chunk_id)
                if len(source_extras) >= regular_source_slots:
                    break

        hsc_extras: list[RetrievalResult] = []
        if source_hsc_activation and hsc_source_ids:
            hsc_extras = self._retriever.hybrid_query_sources(
                query,
                query_embedding,
                hsc_source_ids,
                k=source_hsc_chunk_slots,
                candidates_per_source=source_candidate_pool,
                alpha=alpha,
                source_scores=hsc_source_scores,
                anchor_chunk_ids=hsc_anchor_by_source,
                exclude_chunk_ids=tuple(
                    {
                        *seen,
                        *(result.chunk.chunk_id for result in source_extras),
                    }
                ),
            )
            hsc_extras = [
                result.model_copy(update={"route": "hsc_contraction"})
                for result in hsc_extras
            ]
            if role_aware_retrieval:
                hsc_extras = role_aware_results(
                    query,
                    hsc_extras,
                    user_weight=role_user_weight,
                    assistant_weight=role_assistant_weight,
                    system_weight=role_system_weight,
                )
            if use_source_diversity:
                hsc_extras = source_diverse_results(hsc_extras)
        source_extras = [*source_extras, *hsc_extras]

        if use_attention_feedback and feedback_slots:
            # Sample every first-round route instead of letting the longest
            # route monopolize the Qwen workspace. IDs/scalars cross rounds;
            # request-token state does not.
            groups = [
                list(anchors),
                list(facet_results),
                list(neighbors),
                list(source_extras),
            ]
            attention_candidates: list[RetrievalResult] = []
            attention_seen: set[str] = set()
            position = 0
            while len(attention_candidates) < self._source_candidate_reranker.candidate_pool:
                added = False
                for group in groups:
                    if position >= len(group):
                        continue
                    result = group[position]
                    if result.chunk.chunk_id not in attention_seen:
                        attention_seen.add(result.chunk.chunk_id)
                        attention_candidates.append(result)
                        added = True
                        if len(attention_candidates) >= self._source_candidate_reranker.candidate_pool:
                            break
                if not added:
                    break
                position += 1

            seeds = self._source_candidate_reranker.select(
                query,
                attention_candidates,
                top_k=feedback_seed_slots,
            )
            initial_report = self._source_candidate_reranker.last_report
            feedback_source_ids: list[str] = []
            feedback_source_scores: dict[str, float] = {}
            feedback_anchor_by_source: dict[str, str] = {}
            for seed in seeds:
                if seed.turn is None:
                    continue
                source_id = seed.turn.source_id or seed.turn.turn_id
                if source_id not in feedback_source_scores:
                    feedback_source_ids.append(source_id)
                    feedback_anchor_by_source[source_id] = seed.chunk.chunk_id
                feedback_source_scores[source_id] = max(
                    feedback_source_scores.get(source_id, 0.0),
                    float(seed.association_score or 0.0),
                )

            activation_pieces = [
                f"Original question: {truncate_to_tokens(query, 96)}",
                "Attended evidence:",
                *[
                    truncate_to_tokens(seed.chunk.text, feedback_evidence_tokens)
                    for seed in seeds
                ],
            ]
            activation_window = truncate_to_tokens(
                "\n".join(activation_pieces),
                feedback_query_tokens,
            )
            feedback_results: list[RetrievalResult] = []
            combined_report = None
            if feedback_source_ids:
                initial_ids = {
                    result.chunk.chunk_id
                    for result in [*anchors, *neighbors, *source_extras]
                }
                second_pool = self._retriever.hybrid_query_sources(
                    query,
                    query_embedding,
                    feedback_source_ids,
                    k=self._source_candidate_reranker.candidate_pool,
                    candidates_per_source=source_candidate_pool,
                    alpha=alpha,
                    source_scores=feedback_source_scores,
                    anchor_chunk_ids=feedback_anchor_by_source,
                    exclude_chunk_ids=tuple(initial_ids),
                )
                activation_selected = self._source_candidate_reranker.select(
                    activation_window,
                    second_pool,
                    top_k=feedback_slots,
                )
                combined_report = self._source_candidate_reranker.last_report
                activation_ids = {
                    result.chunk.chunk_id for result in activation_selected
                }
                feedback_results = [
                    result.model_copy(update={"route": "qwen_activation_feedback"})
                    for result in activation_selected
                ]
                for result in second_pool:
                    if len(feedback_results) >= feedback_slots:
                        break
                    if result.chunk.chunk_id in activation_ids:
                        continue
                    feedback_results.append(
                        result.model_copy(update={"route": "feedback_scalar_fill"})
                    )

            protected_source_count = max(0, source_slots - feedback_slots)
            selected_source = list(source_extras[:protected_source_count])
            selected_ids = {result.chunk.chunk_id for result in selected_source}
            for result in feedback_results:
                if len(selected_source) >= source_slots:
                    break
                if result.chunk.chunk_id in selected_ids:
                    continue
                selected_source.append(result)
                selected_ids.add(result.chunk.chunk_id)
            for result in source_extras[protected_source_count:]:
                if len(selected_source) >= source_slots:
                    break
                if result.chunk.chunk_id in selected_ids:
                    continue
                selected_source.append(result)
                selected_ids.add(result.chunk.chunk_id)
            source_extras = selected_source
            initial_stats = (
                initial_report.model_dump() if initial_report is not None else {}
            )
            combined_stats = (
                combined_report.model_dump() if combined_report is not None else {}
            )
            self.last_source_rerank_report = dict(combined_stats)
            self.last_source_rerank_report.update(
                {
                    "passes": int(initial_stats.get("passes", 0))
                    + int(combined_stats.get("passes", 0)),
                    "total_candidate_inspections": int(
                        initial_stats.get("total_candidate_inspections", 0)
                    )
                    + int(combined_stats.get("total_candidate_inspections", 0)),
                    "max_workspace_candidates": max(
                        int(initial_stats.get("max_workspace_candidates", 0)),
                        int(combined_stats.get("max_workspace_candidates", 0)),
                    ),
                    "max_workspace_tokens": max(
                        int(initial_stats.get("max_workspace_tokens", 0)),
                        int(combined_stats.get("max_workspace_tokens", 0)),
                    ),
                    "qwen_candidates_added": int(
                        initial_stats.get("qwen_candidates_added", 0)
                    )
                    + int(combined_stats.get("qwen_candidates_added", 0)),
                }
            )
            self.last_source_rerank_report.update(
                {
                    "feedback_rounds": 1,
                    "feedback_seed_sources": len(feedback_source_ids),
                    "feedback_candidates_added": sum(
                        result.route in {
                            "qwen_activation_feedback",
                            "feedback_scalar_fill",
                        }
                        for result in source_extras
                    ),
                    "feedback_activation_candidates": sum(
                        result.route == "qwen_activation_feedback"
                        for result in source_extras
                    ),
                    "feedback_query_tokens": count_tokens(activation_window),
                }
            )
        baseline = [*anchors, *facet_results, *neighbors, *source_extras]
        output = baseline
        if active_partition_report.get("active_partition_scan_status") == "applied":
            output, admission = self._admit_active_partition_candidates(
                baseline,
                active_partition_candidates,
                anchor_chunk_ids={result.chunk.chunk_id for result in anchors},
                semantic_complete=bool(
                    active_partition_report.get(
                        "active_partition_semantically_complete",
                        False,
                    )
                ),
            )
            scan_truncated = int(
                active_partition_report.get(
                    "active_partition_candidates_truncated",
                    0,
                )
            )
            total_truncated = scan_truncated + int(
                admission["active_partition_candidates_truncated"]
            )
            admission["active_partition_candidates_truncated"] = total_truncated
            if total_truncated:
                active_partition_report[
                    "active_partition_semantically_complete"
                ] = False
                active_partition_report[
                    "selected_scope_structurally_complete"
                ] = False
                active_partition_report["global_semantic_complete"] = False
            active_partition_report.update(admission)
            self.last_partition_routing_report.update(active_partition_report)

            frontier_ids = tuple(result.chunk.chunk_id for result in output)
            frontier_routes = tuple(str(result.route or "") for result in output)
            active_frontier_rows = tuple(
                sorted(
                    (result.chunk.chunk_id, str(result.route or ""))
                    for result in output
                    if str(result.route or "").casefold().startswith(
                        "active_partition_"
                    )
                )
            )
            transcript_turn = self._transcript.current_turn()
            content_high_watermark = self._content_high_watermark()
            if (
                transcript_turn != active_partition_routing_turn
                or content_high_watermark
                != active_partition_content_high_watermark
            ):
                invalidated_reason = (
                    "transcript_advanced_during_route"
                    if transcript_turn != active_partition_routing_turn
                    else "content_changed_during_route"
                )
                active_partition_report.update(
                    {
                        "active_partition_scan_status": "invalidated",
                        "active_partition_exhaustive": False,
                        "active_partition_semantically_complete": False,
                        "selected_scope_structurally_complete": False,
                        "global_semantic_complete": False,
                        "active_partition_snapshot_invalidated_reason": (
                            invalidated_reason
                        ),
                    }
                )
                self.last_partition_routing_report.update(
                    {
                        **active_partition_report,
                        "active_partition_snapshot_validated": False,
                    }
                )
                return output

            query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
            source_set_sha256 = hashlib.sha256(
                "\0".join(routed_source_ids or ()).encode("utf-8")
            ).hexdigest()
            identity_payload = "\0".join(
                [
                    query_sha256,
                    str(transcript_turn),
                    str(content_high_watermark),
                    source_set_sha256,
                    *partition_ids,
                    *frontier_ids,
                    *frontier_routes,
                ]
            )
            routing_identity = hashlib.sha256(
                identity_payload.encode("utf-8")
            ).hexdigest()
            self._active_partition_routing_snapshot = (
                _ActivePartitionRoutingSnapshot(
                    routing_identity=routing_identity,
                    query_sha256=query_sha256,
                    transcript_turn=transcript_turn,
                    content_high_watermark=content_high_watermark,
                    selected_partitions=tuple(partition_ids),
                    routed_source_ids=tuple(routed_source_ids or ()),
                    frontier_chunk_ids=frontier_ids,
                    frontier_routes=frontier_routes,
                    active_frontier_rows=active_frontier_rows,
                    active_partition_total=int(
                        active_partition_report["active_partition_total"]
                    ),
                    active_partition_inspected=int(
                        active_partition_report["active_partition_inspected"]
                    ),
                    active_partition_exhaustive=bool(
                        active_partition_report["active_partition_exhaustive"]
                    ),
                    active_partition_sources_total=int(
                        active_partition_report[
                            "active_partition_sources_total"
                        ]
                    ),
                    active_partition_structural_rows=int(
                        active_partition_report[
                            "active_partition_structural_rows"
                        ]
                    ),
                    active_partition_structural_hypotheses=int(
                        active_partition_report[
                            "active_partition_structural_hypotheses"
                        ]
                    ),
                    active_partition_candidates_admitted=int(
                        active_partition_report[
                            "active_partition_candidates_admitted"
                        ]
                    ),
                    active_partition_candidates_already_present=int(
                        active_partition_report[
                            "active_partition_candidates_already_present"
                        ]
                    ),
                    active_partition_candidates_replaced=int(
                        active_partition_report[
                            "active_partition_candidates_replaced"
                        ]
                    ),
                    active_partition_candidates_truncated=total_truncated,
                    active_partition_structural_overflow=int(
                        active_partition_report[
                            "active_partition_structural_overflow"
                        ]
                    ),
                    active_partition_scan_contract=str(
                        active_partition_report[
                            "active_partition_scan_contract"
                        ]
                    ),
                    active_partition_semantically_complete=bool(
                        active_partition_report[
                            "active_partition_semantically_complete"
                        ]
                    ),
                    partition_scope_kind=str(
                        active_partition_report["partition_scope_kind"]
                    ),
                    partition_inventory_total=int(
                        active_partition_report["partition_inventory_total"]
                    ),
                    selected_partition_count=int(
                        active_partition_report["selected_partition_count"]
                    ),
                    partition_scope_exhaustive=bool(
                        active_partition_report["partition_scope_exhaustive"]
                    ),
                    selected_scope_structurally_complete=bool(
                        active_partition_report[
                            "selected_scope_structurally_complete"
                        ]
                    ),
                    global_semantic_complete=bool(
                        active_partition_report["global_semantic_complete"]
                    ),
                )
            )
            self.last_partition_routing_report.update(
                {
                    "active_partition_routing_identity": routing_identity,
                    "active_partition_snapshot_validated": False,
                }
            )
        return output

    def search_hybrid_neighbors(
        self,
        query: str,
        *,
        k: int = 10,
        radius: int = 1,
        max_neighbors: int | None = None,
        replacement_slots: int = 0,
        ef_search: int = 50,
        candidates: int = 100,
        alpha: float = 0.65,
    ) -> list[RetrievalResult]:
        """Retrieve hybrid anchors, then walk bounded source-local neighbors."""
        anchors = self.search_hybrid(
            query,
            k=k,
            ef_search=ef_search,
            candidates=candidates,
            alpha=alpha,
        )
        expanded = self.expand_source_neighbors(
            anchors,
            radius=radius,
            max_neighbors=max_neighbors,
        )
        if replacement_slots <= 0:
            return expanded
        neighbor_candidates = expanded[len(anchors) :]
        slots = min(replacement_slots, len(anchors), len(neighbor_candidates))
        return list(anchors[: len(anchors) - slots]) + neighbor_candidates[:slots]

    def expand_source_neighbors(
        self,
        anchors: Sequence[RetrievalResult],
        *,
        radius: int = 1,
        max_neighbors: int | None = None,
    ) -> list[RetrievalResult]:
        """Expand cached retrieval results without embedding the query again."""
        return self._retriever.hydrate_source_neighbors(
            anchors,
            radius=radius,
            max_neighbors=max_neighbors,
        )

    def recall_memories(
        self,
        query: str,
        k: int = 10,
        weights: RankWeights = DEFAULT_WEIGHTS,
        now_turn: int | None = None,
        min_energy: float = 0.0,
        reheat: bool = True,
    ) -> list[MemoryResult]:
        """Rank memory items for a query. Retrieved items are reheated.

        ``now_turn`` defaults to the transcript's current position, which is
        what callers want: decay is measured in turns, and the store knows
        where the conversation is. It is forwarded rather than dropped so a
        test or an ablation can evaluate the store as it would look N turns
        from now without appending N turns.
        """
        query_embedding = self._embedder.embed_query(query)
        return self._memory.retrieve(
            query_embedding,
            k=k,
            weights=weights,
            now_turn=now_turn,
            min_energy=min_energy,
            reheat=reheat,
        )

    def _canonical_source_companion_candidates(
        self,
        query: str,
        source_ids: Sequence[str],
        *,
        preferred_role: str | None,
        excluded_chunk_ids: Sequence[str],
        source_scores: Mapping[str, float],
        anchor_chunk_ids: Mapping[str, str],
    ) -> list[RetrievalResult]:
        """Stream a bounded, query-head-specific raw-source shortlist.

        Dense source-local rank can miss the first answer-bearing turn when a
        later recap repeats more query words.  For the one conservative
        identity relation currently available (museum/gallery venue), retain
        the earliest preferred-role row for each unambiguous canonical key.
        The scan retains only chunk IDs and keys for at most four rows/source;
        raw text, keys, and activations are not persisted.
        """

        if not source_ids or re.search(
            r"\b(?:museum|museums|gallery|galleries)\b",
            query,
            re.IGNORECASE,
        ) is None:
            return []
        from memory_condense.coverage_selector import (
            _canonical_answer_object_key,
        )

        selected_sources = list(
            dict.fromkeys(str(source_id) for source_id in source_ids if source_id)
        )
        placeholders = ",".join("?" for _ in selected_sources)
        source_expr = "COALESCE(t.source_id, t.turn_id)"
        excluded = {str(chunk_id) for chunk_id in excluded_chunk_ids}
        selected_ids: dict[str, list[str]] = {
            source_id: [] for source_id in selected_sources
        }
        seen_keys: dict[str, set[str]] = {
            source_id: set() for source_id in selected_sources
        }
        rows = self._db.execute(
            "SELECT c.chunk_id, c.text, " + source_expr + ", t.role "
            "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
            f"WHERE {source_expr} IN ({placeholders}) "
            "ORDER BY t.ordinal, c.rowid",
            tuple(selected_sources),
        )
        for chunk_id, text, source_id, role in rows:
            source_key = str(source_id)
            if (
                source_key not in selected_ids
                or str(chunk_id) in excluded
                or len(selected_ids[source_key])
                >= _SOURCE_CANONICAL_COMPANIONS_PER_SOURCE
                or (
                    preferred_role is not None
                    and str(role).casefold() != preferred_role
                )
            ):
                continue
            answer_key = _canonical_answer_object_key(query, str(text))
            if answer_key is None or answer_key in seen_keys[source_key]:
                continue
            seen_keys[source_key].add(answer_key)
            selected_ids[source_key].append(str(chunk_id))

        results: list[RetrievalResult] = []
        for source_id in selected_sources:
            for chunk_id in selected_ids[source_id]:
                hydrated = self._retriever.hydrate_chunk(
                    chunk_id,
                    score=float(source_scores.get(source_id, 0.0)),
                    route="source_canonical_companion",
                    anchor_chunk_id=anchor_chunk_ids.get(source_id),
                )
                if hydrated is not None:
                    results.append(
                        hydrated.model_copy(
                            update={
                                "memory_source_id": source_id,
                                "source_heat": max(
                                    0.0,
                                    float(source_scores.get(source_id, 0.0)),
                                ),
                            }
                        )
                    )
        return results

    def _performance_source_companion_candidates(
        self,
        query: str,
        source_ids: Sequence[str],
        *,
        preferred_role: str | None,
        excluded_chunk_ids: Sequence[str],
        source_scores: Mapping[str, float],
        anchor_chunk_ids: Mapping[str, str],
    ) -> list[RetrievalResult]:
        """Retain the first direct performance occurrence in each source.

        Source-local dense rank often favors generic playlists, future plans,
        or later summaries over a short artist-and-venue fact.  A single
        streaming pass over already activated sources supplies that missing
        primary event without growing the route union or retaining row text.
        """

        if not source_ids or not is_performance_query(query):
            return []
        selected_sources = list(
            dict.fromkeys(str(source_id) for source_id in source_ids if source_id)
        )
        placeholders = ",".join("?" for _ in selected_sources)
        source_expr = "COALESCE(t.source_id, t.turn_id)"
        excluded = {str(chunk_id) for chunk_id in excluded_chunk_ids}
        selected_ids: dict[str, str] = {}
        rows = self._db.execute(
            "SELECT c.chunk_id, c.text, " + source_expr + ", t.role "
            "FROM chunks AS c JOIN turns AS t ON t.turn_id = c.turn_id "
            f"WHERE {source_expr} IN ({placeholders}) "
            "ORDER BY t.ordinal, c.rowid",
            tuple(selected_sources),
        )
        for chunk_id, text, source_id, role in rows:
            source_key = str(source_id)
            if (
                source_key in selected_ids
                or source_key not in selected_sources
                or str(chunk_id) in excluded
                or (
                    preferred_role is not None
                    and str(role).casefold() != preferred_role
                )
                or not is_direct_past_performance(query, str(text))
            ):
                continue
            selected_ids[source_key] = str(chunk_id)

        results: list[RetrievalResult] = []
        for source_id in selected_sources:
            chunk_id = selected_ids.get(source_id)
            if chunk_id is None:
                continue
            hydrated = self._retriever.hydrate_chunk(
                chunk_id,
                score=float(source_scores.get(source_id, 0.0)),
                route="source_performance_companion",
                anchor_chunk_id=anchor_chunk_ids.get(source_id),
            )
            if hydrated is not None:
                results.append(
                    hydrated.model_copy(
                        update={
                            "memory_source_id": source_id,
                            "source_heat": max(
                                0.0,
                                float(source_scores.get(source_id, 0.0)),
                            ),
                        }
                    )
                )
        return results

    def _hydrate_source_metadata_companions(
        self,
        user_text: str,
        expansions: Sequence[RetrievalResult],
        query_embedding: np.ndarray | None,
    ) -> tuple[list[RetrievalResult], set[str]]:
        """Ensure routed sources carry one bounded, query-selected raw payload.

        Metadata-only routes retain the historical one-row hydration.  For an
        explicit complete-set query, the same bounded source-local chooser is
        run for every source already activated in the final route union.  A
        selected raw row that is absent replaces one deterministic low-value
        row from its own source, so neither source reachability nor candidate
        count can grow.  No answer labels or benchmark categories are read.
        """

        from memory_condense.coverage_selector import compile_set_program

        output = list(expansions)
        source_rows: dict[str, list[tuple[int, RetrievalResult]]] = {}
        metadata_rows: dict[str, list[tuple[int, RetrievalResult]]] = {}
        content_sources: set[str] = set()
        metadata_chunk_ids: list[str] = []
        source_order: list[str] = []
        for index, result in enumerate(output):
            source_id = _retrieval_source_id(result)
            if source_id not in source_rows:
                source_order.append(source_id)
                source_rows[source_id] = []
            source_rows[source_id].append((index, result))
            if parse_source_metadata(result.chunk.text) is None:
                content_sources.add(source_id)
            else:
                metadata_rows.setdefault(source_id, []).append((index, result))
                metadata_chunk_ids.append(result.chunk.chunk_id)

        choose_companions = getattr(
            self._context_candidate_selector,
            "select_source_companions",
            None,
        )
        program = compile_set_program(user_text)
        refresh_all_sources = bool(
            program.requires_completeness and callable(choose_companions)
        )
        requested = (
            list(source_order)
            if refresh_all_sources
            else [
                source_id
                for source_id in source_order
                if source_id in metadata_rows and source_id not in content_sources
            ]
        )

        def empty_report(
            *,
            fallback_reason: str = "",
            fallback_sources: Sequence[str] = (),
        ) -> dict[str, Any]:
            return {
                "requested_sources": requested,
                "hydrated_sources": [],
                "refreshed_sources": [],
                "already_present_sources": [],
                "orphan_sources": [],
                "orphan_count": 0,
                "direct_date_retained": 0,
                "candidate_count_before": len(output),
                "candidate_count_after": len(output),
                "max_candidates_per_source": 1,
                "companion_candidate_count": 0,
                "selector_used": False,
                "selector_fallback_sources": list(fallback_sources),
                "selector_fallback_reason": fallback_reason,
                "semantic_selector_report": {},
                "selected_chunk_ids": {},
                "refresh_all_activated_sources": refresh_all_sources,
                "choice_diagnostics": [],
            }

        if not requested:
            self.last_source_companion_report = empty_report()
            return output, set()

        vector = (
            np.asarray(query_embedding, dtype=np.float32)
            if query_embedding is not None
            else np.asarray(self._embedder.embed_query(user_text), dtype=np.float32)
        )
        source_scores = {
            source_id: max(
                float(result.score) for _index, result in source_rows[source_id]
            )
            for source_id in requested
        }
        anchor_chunk_ids = {
            source_id: (
                source_rows[source_id][0][1].anchor_chunk_id
                or source_rows[source_id][0][1].chunk.chunk_id
            )
            for source_id in requested
        }
        max_per_source = (
            _SOURCE_COMPANION_MAX_PER_SOURCE
            if callable(choose_companions)
            else 1
        )
        try:
            hybrid_companions = self._retriever.hybrid_query_source_companions(
                user_text,
                vector,
                requested,
                metadata_chunk_ids=metadata_chunk_ids,
                max_sources=len(requested),
                max_per_source=max_per_source,
                candidates_per_source=64,
                source_scores=source_scores,
                anchor_chunk_ids=anchor_chunk_ids,
            )
            canonical_companions = (
                self._canonical_source_companion_candidates(
                    user_text,
                    requested,
                    preferred_role=program.preferred_evidence_role,
                    excluded_chunk_ids=metadata_chunk_ids,
                    source_scores=source_scores,
                    anchor_chunk_ids=anchor_chunk_ids,
                )
                if refresh_all_sources
                else []
            )
            performance_companions = (
                self._performance_source_companion_candidates(
                    user_text,
                    requested,
                    preferred_role=program.preferred_evidence_role,
                    excluded_chunk_ids=metadata_chunk_ids,
                    source_scores=source_scores,
                    anchor_chunk_ids=anchor_chunk_ids,
                )
                if refresh_all_sources
                else []
            )
        except Exception as exc:
            if bool(getattr(self._context_candidate_selector, "strict", False)):
                raise
            metadata_orphans = [
                source_id
                for source_id in requested
                if source_id not in content_sources
            ]
            report = empty_report(
                fallback_reason=type(exc).__name__,
                fallback_sources=requested,
            )
            report["orphan_sources"] = metadata_orphans
            report["orphan_count"] = len(metadata_orphans)
            self.last_source_companion_report = report
            return output, set(metadata_orphans)

        candidates_by_source: dict[str, list[RetrievalResult]] = {
            source_id: [] for source_id in requested
        }
        canonical_by_source: dict[str, list[RetrievalResult]] = {
            source_id: [] for source_id in requested
        }
        for result in canonical_companions:
            source_id = _retrieval_source_id(result)
            if source_id in canonical_by_source:
                canonical_by_source[source_id].append(result)
        performance_by_source: dict[str, list[RetrievalResult]] = {
            source_id: [] for source_id in requested
        }
        for result in performance_companions:
            source_id = _retrieval_source_id(result)
            if source_id in performance_by_source:
                performance_by_source[source_id].append(result)
        canonical_primary_rule = bool(
            refresh_all_sources
            and re.search(
                r"\b(?:museum|museums|gallery|galleries)\b",
                user_text,
                re.IGNORECASE,
            )
            and (
                program.quantifier.value == "fixed_cardinality"
                or program.ordering.value != "none"
            )
        )
        # In an ordered/fixed venue set, the earliest preferred-role canonical
        # occurrence is the source's primary event anchor.  Do not let a later
        # high-overlap recap with no venue identity compete it away.  Sources
        # without a conservative canonical parse keep the generic top-N path.
        canonical_primary_sources = {
            source_id
            for source_id, candidates in canonical_by_source.items()
            if canonical_primary_rule and candidates
        }
        performance_primary_rule = bool(
            refresh_all_sources
            and is_performance_query(user_text)
            and (
                program.quantifier.value in {"all", "fixed_cardinality"}
                or program.ordering.value != "none"
            )
        )
        performance_primary_sources = {
            source_id
            for source_id, candidates in performance_by_source.items()
            if performance_primary_rule and candidates
        }
        primary_sources = canonical_primary_sources | performance_primary_sources
        for source_id in canonical_primary_sources:
            candidates_by_source[source_id].append(
                canonical_by_source[source_id][0]
            )
        for source_id in performance_primary_sources - canonical_primary_sources:
            candidates_by_source[source_id].append(
                performance_by_source[source_id][0]
            )
        for result in [
            *canonical_companions,
            *performance_companions,
            *hybrid_companions,
        ]:
            source_id = _retrieval_source_id(result)
            if source_id not in candidates_by_source:
                continue
            if source_id in primary_sources:
                continue
            if any(
                prior.chunk.chunk_id == result.chunk.chunk_id
                for prior in candidates_by_source[source_id]
            ):
                continue
            candidates_by_source[source_id].append(result)

        selectable = {
            source_id: tuple(candidates)
            for source_id, candidates in candidates_by_source.items()
            if candidates
            and (refresh_all_sources or len(candidates) > 1)
        }
        semantic_choices: dict[str, RetrievalResult] = {}
        selector_used = False
        selector_fallback_sources: list[str] = []
        selector_fallback_reasons: list[str] = []
        semantic_selector_report: dict[str, Any] = {}
        selector_reports: list[dict[str, Any]] = []

        def dump_report(raw_report: Any) -> dict[str, Any]:
            dumped = getattr(raw_report, "model_dump", None)
            if callable(dumped):
                return dict(dumped())
            if isinstance(raw_report, Mapping):
                return dict(raw_report)
            return {}

        def nested_value(report: Mapping[str, Any], key: str) -> Any:
            if key in report:
                return report[key]
            for nested_key in ("provider_report", "score_report"):
                nested = report.get(nested_key)
                if isinstance(nested, Mapping):
                    value = nested_value(nested, key)
                    if value is not None:
                        return value
            return None

        selection_batches: list[dict[str, tuple[RetrievalResult, ...]]] = []
        if selectable:
            if refresh_all_sources:
                current: dict[str, tuple[RetrievalResult, ...]] = {}
                current_count = 0
                for source_id, candidates in selectable.items():
                    if current and current_count + len(candidates) > 128:
                        selection_batches.append(current)
                        current = {}
                        current_count = 0
                    current[source_id] = candidates
                    current_count += len(candidates)
                if current:
                    selection_batches.append(current)
            else:
                selection_batches.append(selectable)

        for batch in selection_batches:
            selector_used = True
            batch_unavailable = False
            try:
                proposed = choose_companions(user_text, batch)
            except Exception as exc:
                if bool(getattr(self._context_candidate_selector, "strict", False)):
                    raise
                proposed = {}
                selector_fallback_sources.extend(
                    source_id
                    for source_id in batch
                    if source_id not in selector_fallback_sources
                )
                selector_fallback_reasons.append(type(exc).__name__)
                batch_unavailable = True
            raw_report = getattr(
                self._context_candidate_selector,
                "last_source_companion_report",
                None,
            )
            batch_report = dump_report(raw_report)
            if batch_report:
                selector_reports.append(batch_report)
            if not isinstance(proposed, Mapping):
                proposed = {}
                selector_fallback_sources.extend(
                    source_id
                    for source_id in batch
                    if source_id not in selector_fallback_sources
                )
                selector_fallback_reasons.append("invalid_selection_mapping")
                batch_unavailable = True

            selected_ids = nested_value(batch_report, "selected_chunk_ids")
            selected_ids = selected_ids if isinstance(selected_ids, Mapping) else {}
            membership_scores = nested_value(
                batch_report,
                "selected_membership_scores",
            )
            membership_scores = (
                membership_scores
                if isinstance(membership_scores, Mapping)
                else {}
            )
            input_count = int(nested_value(batch_report, "input_candidates") or 0)
            inspected_count = int(
                nested_value(batch_report, "inspected_candidates") or 0
            )
            fallback_reason = str(
                nested_value(batch_report, "fallback_reason") or ""
            )
            all_inspected = bool(
                input_count >= sum(len(rows) for rows in batch.values())
                and inspected_count >= input_count
                and not fallback_reason
            )
            for source_id, candidates in batch.items():
                if batch_unavailable:
                    continue
                proposed_result = proposed.get(source_id)
                # Exact object provenance matters: a provider may reorder the
                # supplied raw rows, but it may not fabricate a replacement
                # that merely reuses one of their IDs.
                selected = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate is proposed_result
                    ),
                    None,
                )
                inspected_winner = bool(
                    source_id in membership_scores
                    or (
                        all_inspected
                        and selected is not None
                        and str(selected_ids.get(source_id, ""))
                        == selected.chunk.chunk_id
                    )
                )
                if selected is None or (refresh_all_sources and not inspected_winner):
                    if source_id not in selector_fallback_sources:
                        selector_fallback_sources.append(source_id)
                    selector_fallback_reasons.append(
                        "invalid_selection"
                        if selected is None
                        else "uninspected_selection"
                    )
                    continue
                semantic_choices[source_id] = selected

        if selector_reports:
            if refresh_all_sources:
                semantic_selector_report = {
                    "batch_count": len(selector_reports),
                    "input_sources": sum(
                        int(nested_value(report, "input_sources") or 0)
                        for report in selector_reports
                    ),
                    "input_candidates": sum(
                        int(nested_value(report, "input_candidates") or 0)
                        for report in selector_reports
                    ),
                    "inspected_candidates": sum(
                        int(nested_value(report, "inspected_candidates") or 0)
                        for report in selector_reports
                    ),
                    "selected_chunk_ids": {
                        source_id: result.chunk.chunk_id
                        for source_id, result in semantic_choices.items()
                    },
                    "retained_transformer_state_bytes": max(
                        (
                            int(
                                nested_value(
                                    report,
                                    "retained_transformer_state_bytes",
                                )
                                or 0
                            )
                            for report in selector_reports
                        ),
                        default=0,
                    ),
                    "fallback_reasons": list(
                        dict.fromkeys(
                            reason
                            for reason in selector_fallback_reasons
                            if reason
                        )
                    ),
                }
            else:
                semantic_selector_report = selector_reports[-1]

        companion_by_source: dict[str, RetrievalResult] = {}
        choice_diagnostics: list[dict[str, Any]] = []
        for source_id in requested:
            candidates = candidates_by_source[source_id]
            if not candidates:
                continue
            companion = semantic_choices.get(source_id)
            if companion is None and not refresh_all_sources:
                companion = candidates[0]
            if companion is None:
                continue
            companion_by_source[source_id] = companion
            choice_diagnostics.append(
                {
                    "source_id": source_id,
                    "candidate_count": len(candidates),
                    "candidate_chunk_ids": [
                        candidate.chunk.chunk_id for candidate in candidates
                    ],
                    "selected_chunk_id": companion.chunk.chunk_id,
                    "selected_local_rank": next(
                        rank
                        for rank, candidate in enumerate(candidates, start=1)
                        if candidate is companion
                    ),
                    "selected_by": (
                        "semantic"
                        if source_id in semantic_choices
                        else "retrieval"
                    ),
                }
            )

        hydrated_sources: list[str] = []
        refreshed_sources: list[str] = []
        already_present_sources: list[str] = []
        active_partition_protected_sources: list[str] = []
        for source_id in requested:
            companion = companion_by_source.get(source_id)
            if companion is None:
                continue
            if any(
                result.chunk.chunk_id == companion.chunk.chunk_id
                for _index, result in source_rows[source_id]
            ):
                already_present_sources.append(source_id)
                continue

            def replacement_key(
                row: tuple[int, RetrievalResult],
            ) -> tuple[int, float, int]:
                index, activation = row
                if parse_source_metadata(activation.chunk.text) is not None:
                    tier = 0
                    # Synthetic timestamps carry no answer payload; preserve
                    # the first routed anchor rather than letting an arbitrary
                    # score difference between duplicate metadata rows choose
                    # provenance.
                    value = 0.0
                    tie_break = index
                else:
                    role = (
                        activation.turn.role.casefold()
                        if activation.turn is not None
                        else ""
                    )
                    route = (activation.route or "").casefold()
                    tier = (
                        1
                        if (
                            (
                                program.preferred_evidence_role is not None
                                and role != program.preferred_evidence_role
                            )
                            or "support" in route
                        )
                        else 2
                    )
                    value = float(activation.score)
                    tie_break = index
                return tier, value, tie_break

            replaceable_rows = [
                row
                for row in source_rows[source_id]
                if not str(row[1].route or "").casefold().startswith(
                    "active_partition_"
                )
            ]
            if not replaceable_rows:
                active_partition_protected_sources.append(source_id)
                continue
            index, activation = min(
                replaceable_rows,
                key=replacement_key,
            )
            activation_route = str(activation.route or "")
            copied_route = (
                companion.route
                if activation_route.casefold().startswith("active_partition_")
                else activation.route
            )
            output[index] = companion.model_copy(
                update={
                    "score": float(activation.score),
                    "route": copied_route,
                    "anchor_chunk_id": (
                        activation.anchor_chunk_id or activation.chunk.chunk_id
                    ),
                    "memory_source_id": source_id,
                    "source_heat": activation.source_heat,
                    "source_token_budget": activation.source_token_budget,
                }
            )
            hydrated_sources.append(source_id)
            if refresh_all_sources:
                refreshed_sources.append(source_id)

        orphan_sources = [
            source_id
            for source_id in requested
            if source_id not in content_sources
            and source_id not in companion_by_source
        ]
        selector_fallback_reason = ";".join(
            dict.fromkeys(
                reason for reason in selector_fallback_reasons if reason
            )
        )
        companion_report = {
            "requested_sources": requested,
            "hydrated_sources": hydrated_sources,
            "refreshed_sources": refreshed_sources,
            "already_present_sources": already_present_sources,
            "orphan_sources": orphan_sources,
            "orphan_count": len(orphan_sources),
            "direct_date_retained": 0,
            "candidate_count_before": len(expansions),
            "candidate_count_after": len(output),
            "max_candidates_per_source": (
                max_per_source
                + (
                    _SOURCE_CANONICAL_COMPANIONS_PER_SOURCE
                    if canonical_companions
                    else 0
                )
                + (
                    _SOURCE_PERFORMANCE_COMPANIONS_PER_SOURCE
                    if performance_companions
                    else 0
                )
            ),
            "companion_candidate_count": sum(
                len(candidates) for candidates in candidates_by_source.values()
            ),
            "selector_used": selector_used,
            "selector_fallback_sources": selector_fallback_sources,
            "selector_fallback_reason": selector_fallback_reason,
            "semantic_selector_report": semantic_selector_report,
            "selected_chunk_ids": {
                source_id: result.chunk.chunk_id
                for source_id, result in companion_by_source.items()
            },
            "refresh_all_activated_sources": refresh_all_sources,
            "choice_diagnostics": choice_diagnostics,
        }
        if active_partition_protected_sources:
            companion_report["active_partition_protected_sources"] = (
                active_partition_protected_sources
            )
        self.last_source_companion_report = companion_report
        return output, set(orphan_sources)

    # -- context assembly ---------------------------------------------------

    def build_context(
        self,
        user_text: str,
        system_prompt: str = "",
        recent_turns: int = 8,
        k_memories: int = 8,
        k_expansions: int = 10,
        hybrid: bool = True,
        reheat_memories: bool = True,
        use_consolidation: bool = True,
        learn_consolidation: bool = True,
        consolidation_memory_slots: int = 1,
        consolidation_chunk_slots: int = 1,
        consolidation_min_count: int = 2,
        consolidation_hops: int = 1,
        consolidation_candidates: int = 32,
        consolidation_diffusion_width: int = 32,
        access_event_id: str | None = None,
        expansion_results: Sequence[RetrievalResult] | None = None,
    ) -> PackedContext:
        """Assemble a token-budgeted prompt for ``user_text``.

        Memory header + recent window + verbatim expansions, each capped
        independently so context cost stays predictable as the conversation
        grows. Established live-consolidation edges may propose bounded
        additive candidates before packing; they never evict a direct result
        merely to reserve a graph slot. Only direct results that actually
        survive packing train the graph afterward, preventing a graph-selected
        result from reinforcing itself merely because the graph selected it.
        """
        self.last_source_companion_report = {
            "requested_sources": [],
            "hydrated_sources": [],
            "refreshed_sources": [],
            "already_present_sources": [],
            "orphan_sources": [],
            "orphan_count": 0,
            "direct_date_retained": 0,
            "candidate_count_before": 0,
            "candidate_count_after": 0,
            "max_candidates_per_source": 1,
            "companion_candidate_count": 0,
            "selector_used": False,
            "selector_fallback_sources": [],
            "selector_fallback_reason": "",
            "semantic_selector_report": {},
            "selected_chunk_ids": {},
            "refresh_all_activated_sources": False,
            "choice_diagnostics": [],
        }
        # Ranking is a read; only memories that survive the header budget are
        # genuine accesses.  Reheating all top-k candidates here kept dropped
        # rows artificially warm and defeated pruning by access frequency.
        memories = (
            self.recall_memories(user_text, k=k_memories, reheat=False)
            if k_memories
            else []
        )

        expansions: list[RetrievalResult] = list(expansion_results or ())
        routing_snapshot: _ActivePartitionRoutingSnapshot | None = None
        candidate_snapshot = self._active_partition_routing_snapshot
        self.last_partition_routing_report.pop(
            "active_partition_snapshot_invalidated_reason",
            None,
        )
        if expansion_results is not None and candidate_snapshot is not None:
            report_identity = str(
                self.last_partition_routing_report.get(
                    "active_partition_routing_identity",
                    "",
                )
            )
            if (
                report_identity == candidate_snapshot.routing_identity
                and candidate_snapshot.matches(
                    user_text,
                    self._transcript.current_turn(),
                    self._content_high_watermark(),
                    expansions,
                )
            ):
                routing_snapshot = candidate_snapshot
                self.last_partition_routing_report[
                    "active_partition_snapshot_prevalidated"
                ] = True
                self.last_partition_routing_report[
                    "active_partition_snapshot_validated"
                ] = False
            else:
                self.last_partition_routing_report[
                    "active_partition_snapshot_prevalidated"
                ] = False
                self.last_partition_routing_report[
                    "active_partition_snapshot_validated"
                ] = False
        query_embedding: np.ndarray | None = None
        if expansion_results is not None:
            if use_consolidation and expansions:
                query_embedding = self._embedder.embed_query(user_text)
        elif k_expansions:
            query_embedding = self._embedder.embed_query(user_text)
            if hybrid:
                expansions = self.search_hybrid_from_embedding(
                    user_text,
                    query_embedding,
                    k=k_expansions,
                )
            else:
                expansions = self._retriever.query(
                    query_embedding,
                    k=k_expansions,
                )

        if use_consolidation and (memories or expansions):
            memories, expansions = expand_context_associations(
                memories,
                expansions,
                store=self._consolidation,
                get_memory=self._memory.get,
                hydrate_chunk=self._retriever.hydrate_chunk,
                now_turn=self._transcript.current_turn(),
                memory_slots=consolidation_memory_slots,
                chunk_slots=consolidation_chunk_slots,
                max_candidates=consolidation_candidates,
                min_coactivation_count=consolidation_min_count,
                diffusion_hops=consolidation_hops,
                diffusion_width=consolidation_diffusion_width,
                chunk_relevance=(
                    lambda chunk_ids: self._retriever.cosine_scores(
                        query_embedding,
                        chunk_ids,
                    )
                    if query_embedding is not None
                    else {}
                ),
            )

        orphan_metadata_sources: set[str] = set()
        if self._packer.budget.source_metadata_expansions and expansions:
            expansions, orphan_metadata_sources = (
                self._hydrate_source_metadata_companions(
                    user_text,
                    expansions,
                    query_embedding,
                )
            )

        turns = self._transcript.get_recent(recent_turns) if recent_turns else []
        recent = [(t.role, t.text) for t in turns]
        source_metadata: dict[str, str] = {}
        if self._packer.budget.source_metadata_expansions:
            source_ids = [
                str(
                    result.memory_source_id
                    or (result.turn.source_id if result.turn is not None else None)
                    or result.chunk.turn_id
                )
                for result in expansions
            ]
            source_metadata = self._transcript.source_metadata(source_ids)
            if orphan_metadata_sources and _DIRECT_DATE_QUERY_RE.search(user_text):
                for source_id in orphan_metadata_sources:
                    source_metadata.pop(source_id, None)
                self.last_source_companion_report["direct_date_retained"] = len(
                    orphan_metadata_sources
                )

        # Consolidation and source-companion refresh are allowed to reshape
        # the ordinary expansion frontier. Typed selected-partition
        # completeness belongs only to the exact active IDs/routes audited by
        # the scan, so bind that subset again at the last possible point
        # before the packer sees the proof. Ordinary rows may safely reorder.
        if routing_snapshot is not None:
            report_identity = str(
                self.last_partition_routing_report.get(
                    "active_partition_routing_identity",
                    "",
                )
            )
            if (
                report_identity == routing_snapshot.routing_identity
                and routing_snapshot.matches(
                    user_text,
                    self._transcript.current_turn(),
                    self._content_high_watermark(),
                    expansions,
                )
            ):
                self.last_partition_routing_report[
                    "active_partition_snapshot_validated"
                ] = True
            else:
                routing_snapshot = None
                self.last_partition_routing_report[
                    "active_partition_snapshot_validated"
                ] = False
                self.last_partition_routing_report[
                    "active_partition_snapshot_invalidated_reason"
                ] = "audited_frontier_changed_before_pack"

        active_partition_kwargs: dict[str, Any] = {}
        if routing_snapshot is not None:
            pack_fields = routing_snapshot.pack_fields()
            try:
                pack_parameters = inspect.signature(self._packer.pack).parameters
            except (TypeError, ValueError):
                pack_parameters = {}
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in pack_parameters.values()
            )
            active_partition_kwargs = {
                key: value
                for key, value in pack_fields.items()
                if accepts_kwargs or key in pack_parameters
            }
        packed = self._packer.pack(
            system_prompt=system_prompt,
            memories=memories,
            recent_turns=recent,
            expansions=expansions,
            user_text=user_text,
            source_metadata=source_metadata,
            **active_partition_kwargs,
        )
        selector_report = getattr(
            self._context_candidate_selector,
            "last_report",
            None,
        )
        self.last_coverage_selection_report = (
            selector_report.model_dump()
            if selector_report is not None
            else {}
        )
        self.last_coverage_candidate_trace = list(
            self._packer.last_expansion_trace
            if self._context_candidate_selector is not None
            else ()
        )
        memory_by_id = {result.item.mem_id: result for result in memories}
        chunk_by_id = {result.chunk.chunk_id: result for result in expansions}
        # Keep the independently retrieved subset explicit for delayed Qwen
        # inspection. The text-free IDs are already part of the packed result;
        # no activation or prompt buffer is retained.
        direct_memory_ids = [
            mem_id
            for mem_id in packed.memory_ids
            if memory_by_id[mem_id].route != "live_consolidation"
        ]
        direct_chunk_ids = [
            chunk_id
            for chunk_id in packed.expansion_chunk_ids
            if chunk_by_id[chunk_id].route != "live_consolidation"
        ]
        event_id = access_event_id or self._context_event_id(
            user_text,
            direct_memory_ids,
            direct_chunk_ids,
        )
        packed = packed.model_copy(
            update={
                "direct_memory_ids": direct_memory_ids,
                "direct_expansion_chunk_ids": direct_chunk_ids,
                "consolidation_event_id": event_id,
            }
        )
        if reheat_memories:
            now_turn = self._transcript.current_turn()
            ranked_by_id = {result.item.mem_id: result.item for result in memories}
            self._memory.touch_many(
                [ranked_by_id[mem_id] for mem_id in packed.memory_ids],
                now_turn=now_turn,
            )
        if learn_consolidation:
            # Learn from independent retrieval evidence only.  A candidate
            # admitted by this graph must later be found directly before it can
            # strengthen the assembly, avoiding a self-confirming feedback loop.
            activations = context_activations(
                direct_memory_ids,
                direct_chunk_ids,
            )
            if activations:
                self._consolidation.observe(event_id, activations)
                packed = packed.model_copy(update={"consolidation_learned": True})
        return packed

    def observe_context_access(
        self,
        memory_ids: Sequence[str],
        chunk_ids: Sequence[str],
        *,
        access_event_id: str,
        now_turn: int | None = None,
        node_activations: Mapping[ConsolidationNode, float] | None = None,
        pair_affinities: Mapping[
            tuple[ConsolidationNode, ConsolidationNode], float
        ]
        | None = None,
        causal_chunk_ids: Sequence[str] = (),
    ) -> ConsolidationUpdate:
        """Explicitly reinforce one externally assembled, bounded context.

        Rank-discounted activity is the provider-free default.  A transient
        Qwen prefix inspection may instead pass CAV-derived node activity and
        bounded QK/OV ``pair_affinities``. ``causal_chunk_ids`` marks newly
        produced response/tool evidence in a completed interaction. Only the
        resulting scalar update is durable; the caller remains responsible for
        discarding the workspace.
        """

        allowed = set(context_activations(memory_ids, chunk_ids))
        if node_activations is None:
            activations = context_activations(memory_ids, chunk_ids)
        else:
            activations = {
                node: float(value)
                for node, value in node_activations.items()
                if node in allowed
            }
        filtered_pairs = {
            pair: float(value)
            for pair, value in (pair_affinities or {}).items()
            if pair[0] in allowed and pair[1] in allowed
        }
        causal_targets = tuple(
            node
            for chunk_id in dict.fromkeys(str(value) for value in causal_chunk_ids)
            if (node := ConsolidationNode.chunk(chunk_id)) in allowed
        )
        return self._consolidation.observe(
            access_event_id,
            activations,
            pair_affinities=filtered_pairs,
            causal_targets=causal_targets,
            now_turn=now_turn,
        )

    def consolidate_context_with_qwen(
        self,
        user_text: str,
        packed: PackedContext,
        linker: object,
        *,
        access_event_id: str | None = None,
        now_turn: int | None = None,
        causal_chunk_ids: Sequence[str] = (),
    ) -> tuple[object, ConsolidationUpdate]:
        """Inspect a completed turn with Qwen and apply one scalar update.

        This is intentionally separate from :meth:`build_context` so callers
        may run the bounded prefix pass after the response or on a background
        queue instead of adding it to answer latency. Build with
        ``learn_consolidation=False`` first; mixing the rank fallback and Qwen
        learning for one context would count that prompt twice.
        """

        if packed.consolidation_learned:
            raise ValueError(
                "context already used rank-based consolidation; build with "
                "learn_consolidation=False before Qwen consolidation"
            )
        memories = [
            item
            for mem_id in packed.direct_memory_ids
            if (item := self._memory.get(mem_id)) is not None
        ]
        chunks = [
            result
            for chunk_id in packed.direct_expansion_chunk_ids
            if (
                result := self._retriever.hydrate_chunk(
                    chunk_id,
                    score=0.0,
                    route="direct_for_consolidation",
                )
            )
            is not None
        ]
        result, activations = inspect_qwen_context_hyperplane(
            linker,
            user_text,
            memories,
            chunks,
        )
        event_id = (
            access_event_id
            or packed.consolidation_event_id
            or self._context_event_id(
                user_text,
                packed.direct_memory_ids,
                packed.direct_expansion_chunk_ids,
            )
        )
        update = self.observe_context_access(
            packed.direct_memory_ids,
            packed.direct_expansion_chunk_ids,
            access_event_id=event_id,
            now_turn=now_turn,
            node_activations=activations,
            causal_chunk_ids=causal_chunk_ids,
        )
        return result, update

    def _context_event_id(
        self,
        user_text: str,
        memory_ids: Sequence[str],
        chunk_ids: Sequence[str],
    ) -> str:
        """Stable retry identity without retaining the prompt or context text."""

        payload = "\x1f".join(
            [
                str(self._transcript.current_turn()),
                user_text,
                *memory_ids,
                *chunk_ids,
            ]
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"context:{self._transcript.current_turn()}:{digest}"

    # -- accessors ----------------------------------------------------------

    @property
    def transcript(self) -> TranscriptStore:
        """Access the transcript store directly."""
        return self._transcript

    @property
    def memory(self) -> MemoryStore:
        """Access the memory store directly."""
        return self._memory

    @property
    def retriever(self) -> SimilarityRetriever:
        """Access the chunk retriever directly."""
        return self._retriever

    @property
    def associations(self) -> AssociationStore:
        """Compact external CAV/QK/OV artifacts; never transformer token state."""
        return self._associations

    @property
    def consolidation(self) -> LiveConsolidationStore:
        """Prompt-driven associations spanning semantic memory and evidence."""

        return self._consolidation

    @property
    def database_path(self) -> Path:
        """SQLite path for independent read-only experiment workers."""
        return self._db.path

    @property
    def validator(self) -> Validator:
        """Access the provenance validator directly."""
        return self._validator

    def heat_counts(self, now_turn: int | None = None) -> dict[str, int]:
        """Current HOT/WARM/COLD distribution of active memory items."""
        return self._memory.heat_counts(now_turn=now_turn)

    def close(self) -> None:
        """Persist index and close database."""
        try:
            if self._persist_index_on_close:
                self._retriever.save()
        finally:
            self._retriever.release()
            self._db.close()

    def __enter__(self) -> MemoryCondenser:
        return self

    def __exit__(self, *args) -> None:
        self.close()
