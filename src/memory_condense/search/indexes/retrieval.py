"""Compatibility facade for the layered similarity retriever."""

from __future__ import annotations

from memory_condense.search.indexes.hybrid_queries import HybridQueryMixin
from memory_condense.search.indexes.index_lifecycle import IndexLifecycleMixin
from memory_condense.search.indexes.retrieval_models import (
    DEFAULT_SPAN_TOKENS,
    PartitionContentRow,
    hydrate_chunk_result,
    load_chunk_payload,
    load_turn_payload,
)
from memory_condense.search.indexes.span_source_queries import SpanSourceQueryMixin


class SimilarityRetriever(
    IndexLifecycleMixin,
    SpanSourceQueryMixin,
    HybridQueryMixin,
):
    """Chunk retrieval over hnswlib (dense) and BM25 (lexical).

    ``query`` is the pure-dense baseline path and is kept deliberately
    untouched — the eval ablations compare against it. ``hybrid_query``
    unions dense and lexical candidates and reranks them with
    ``ranking.blend_hybrid``.
    """


__all__ = [
    "DEFAULT_SPAN_TOKENS",
    "PartitionContentRow",
    "SimilarityRetriever",
    "hydrate_chunk_result",
    "load_chunk_payload",
    "load_turn_payload",
]
