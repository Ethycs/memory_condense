"""Compatibility facade for the layered similarity retriever."""

from __future__ import annotations

# Retain the historical module bindings for import and diagnostic compatibility.
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import hnswlib
import numpy as np

from memory_condense.domain import ranking
from memory_condense.search.indexes import lexical
from memory_condense.associations.association_store import AssociationStore
from memory_condense.persistence.db import Database
from memory_condense.search.indexes.hybrid_queries import HybridQueryMixin
from memory_condense.search.indexes.index_lifecycle import (
    IndexLifecycleMixin,
    _LABEL_KEY,
)
from memory_condense.search.indexes.lexical import LexicalIndex
from memory_condense.search.indexes.retrieval_models import (
    DEFAULT_SPAN_TOKENS,
    PartitionContentRow,
    hydrate_chunk_result,
    load_chunk_payload,
    load_turn_payload,
)
from memory_condense.search.indexes.source_hierarchy import SourceContractionIndex
from memory_condense.search.indexes.span_source_queries import SpanSourceQueryMixin
from memory_condense.domain.schemas import Chunk, RetrievalResult, Turn
from memory_condense.persistence.transcript_store import parse_source_metadata


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
