"""Stateful application facade composed from focused workflow mixins."""

from __future__ import annotations

import hashlib
import inspect
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from memory_condense.application.condenser_contracts import (
    ActivePartitionHypothesis as _ActivePartitionHypothesis,
    ActivePartitionRoutingSnapshot as _ActivePartitionRoutingSnapshot,
    SourceCandidateReranker,
    SourceCompanionSelector,
)
from memory_condense.application.discourse_workflow import DiscourseWorkflowMixin
from memory_condense.application.graph_workflow import GraphWorkflowMixin
from memory_condense.application.ingest_workflow import IngestWorkflowMixin
from memory_condense.application.partition_workflow import PartitionWorkflowMixin
from memory_condense.application.query_routing import (
    SAFE_ASSOCIATION_LEXICAL_THRESHOLD,
    SAFE_ASSOCIATION_MAX_TOKEN_INCREASE,
    is_multi_fact_query,
    query_facets,
    rank_concept_members,
    role_aware_results,
    select_source_partitions,
    source_diverse_results,
    source_partition_ranking,
)
from memory_condense.application.retrieval_workflow import RetrievalWorkflowMixin
from memory_condense.application.source_companions import SourceCompanionWorkflowMixin
from memory_condense.associations.association_store import AssociationStore
from memory_condense.associations.consolidation import (
    ConsolidationNode,
    ConsolidationUpdate,
    LiveConsolidationStore,
    context_activations,
    expand_context_associations,
    inspect_qwen_context_hyperplane,
)
from memory_condense.domain.schemas import PackedContext, RetrievalResult
from memory_condense.ingest.chunker import Chunker
from memory_condense.ingest.extractor import Extractor, RuleBasedExtractor
from memory_condense.ingest.validator import Validator
from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.persistence.db import Database
from memory_condense.persistence.memory_store import MemoryStore
from memory_condense.persistence.transcript_store import TranscriptStore
from memory_condense.search.indexes.retrieval import SimilarityRetriever
from memory_condense.search.packing.context_packer import (
    ContextBudget,
    ContextPacker,
    ExpansionSelector,
)


_DIRECT_DATE_QUERY_RE = re.compile(
    r"\bwhen\b|\b(?:what|which)\s+(?:(?:was|is|were|are)\s+)?"
    r"(?:the\s+)?(?:date|day)\b",
    re.IGNORECASE,
)


class MemoryCondenser(
    IngestWorkflowMixin,
    RetrievalWorkflowMixin,
    DiscourseWorkflowMixin,
    PartitionWorkflowMixin,
    GraphWorkflowMixin,
    SourceCompanionWorkflowMixin,
):
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
        self._init_discourse_workflow()
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


__all__ = [
    "MemoryCondenser",
    "SAFE_ASSOCIATION_LEXICAL_THRESHOLD",
    "SAFE_ASSOCIATION_MAX_TOKEN_INCREASE",
    "SourceCandidateReranker",
    "SourceCompanionSelector",
    "is_multi_fact_query",
    "query_facets",
    "rank_concept_members",
    "role_aware_results",
    "select_source_partitions",
    "source_diverse_results",
    "source_partition_ranking",
]
