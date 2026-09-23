"""Resident semantic addresses over the leaves of a compiled summary hierarchy.

Embedding inputs are stored summaries only. Leaf addresses preserve the exact
raw section partition; a lossy parent cannot prevent a matching leaf from
entering the shortlist. Neither compilation nor search reads raw transcripts.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib

import numpy as np

from memory_condense.domain._discourse_identity import canonical_json, quote_sha256
from memory_condense.search.hot_retrieval import ExactDenseAddressIndex
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan, SectionSummaryIndex
from memory_condense.search.section_summary import bound_int, exact_text


def summary_embedding_identity(encoder) -> str:
    """Bind the same model/checkpoint/execution controls at compile and query."""
    return canonical_json({"model_id": encoder.model_name, "model_revision": encoder.model_revision,
        "checkpoint_sha256": encoder.checkpoint_sha256, "execution": encoder.execution_identity})


class SemanticSectionIndex:
    """An immutable hierarchy, summary vectors, and a matching query encoder.

    The model stays owned by the caller. `compile` performs ingest-time document
    embedding; `route` embeds exactly one live query. `route_vector` supports
    separately timed/cached query vectors, but requires their encoder identity
    explicitly and must not be used to claim live encoding latency.
    """
    __slots__ = ("hierarchy", "embedding_identity", "sections", "_dense", "_lexical",
                 "receipt_sha256", "_metadata", "_locked")

    def __setattr__(self, name, value):
        if getattr(self, "_locked", False):
            raise AttributeError("semantic summary indexes are immutable snapshots")
        object.__setattr__(self, name, value)

    def __init__(self, hierarchy: SectionSummaryIndex, matrix: np.ndarray, *, embedding_identity: str):
        if type(hierarchy) is not SectionSummaryIndex:
            raise TypeError("semantic addresses require an authenticated summary hierarchy")
        exact_text(embedding_identity, "embedding_identity")
        self.hierarchy = hierarchy
        self.embedding_identity = embedding_identity
        self.sections = tuple(s for s in hierarchy.sections if not s.child_section_ids)
        if not self.sections:
            raise ValueError("semantic summary index requires at least one leaf")
        # The existing dense implementation copies/validates a normalized FP32
        # matrix. Mutating the caller's array cannot mutate resident addresses.
        self._dense = ExactDenseAddressIndex([s.section_id for s in self.sections], matrix)
        self._lexical = SectionSummaryIndex(self.sections)
        self._metadata = canonical_json({
            "format": "memory-condense-semantic-section-index-v1", "hierarchy_sha256": hierarchy.receipt_sha256,
            "embedding_identity": embedding_identity, "section_ids": [s.section_id for s in self.sections],
            "summary_sha256s": [quote_sha256(s.summary) for s in self.sections],
            "matrix_sha256": hashlib.sha256(np.asarray(matrix).tobytes(order="C")).hexdigest(),
            "dimension": self._dense.dimension, "dtype": "float32", "normalized": True,
            "document_inputs": "stored leaf summaries only", "raw_content_inspections": 0,
            "lexical_leaf_index_sha256": self._lexical.receipt_sha256})
        self.receipt_sha256 = quote_sha256(self._metadata)
        self._locked = True

    @classmethod
    def compile(cls, hierarchy: SectionSummaryIndex, *, encoder):
        leaves = tuple(s for s in hierarchy.sections if not s.child_section_ids)
        if not leaves:
            raise ValueError("cannot compile an empty summary hierarchy")
        before = summary_embedding_identity(encoder)
        vectors = np.array(encoder.embed_queries([s.summary for s in leaves]), dtype=np.float32, copy=True)
        if summary_embedding_identity(encoder) != before:
            raise ValueError("summary embedding identity changed during compilation")
        if vectors.ndim != 2 or vectors.shape[0] != len(leaves) or not np.isfinite(vectors).all():
            raise ValueError("summary encoder returned invalid or missing rows")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            raise ValueError("summary embeddings must be finite and nonzero")
        vectors /= norms
        return cls(hierarchy, vectors, embedding_identity=before)

    @property
    def metadata_json(self) -> str:
        return self._metadata

    @property
    def vector_nbytes(self) -> int:
        return self._dense.nbytes

    def route(self, query: str, *, encoder, max_sections: int = 8, lexical_reserve: int = 0,
              eligible_source_ids: Sequence[str] | None = None) -> SectionRoutePlan:
        exact_text(query, "query")
        before = summary_embedding_identity(encoder)
        if before != self.embedding_identity:
            raise ValueError("query encoder does not match the compiled summary vectors")
        # Scope and budget validation precede encoding, including an empty scope.
        scope = self._scope(eligible_source_ids)
        self._limits(max_sections, lexical_reserve)
        if scope is not None and not any(s.source_id in scope for s in self.sections):
            return self._plan(query, (), 0, scope, max_sections, lexical_reserve)
        vector = encoder.embed_query(query)
        if summary_embedding_identity(encoder) != before:
            raise ValueError("query encoder identity changed during encoding")
        return self.route_vector(query, vector, embedding_identity=before, max_sections=max_sections,
                                 lexical_reserve=lexical_reserve, eligible_source_ids=scope)

    @staticmethod
    def _scope(value):
        scope = None if value is None else tuple(value)
        if scope is not None and (any(type(s) is not str or not s for s in scope) or len(set(scope)) != len(scope)):
            raise ValueError("source scope must contain unique nonempty exact IDs")
        return scope

    @staticmethod
    def _limits(max_sections, lexical_reserve):
        bound_int(max_sections, "max_sections", 1)
        bound_int(lexical_reserve, "lexical_reserve", 0)
        if lexical_reserve > max_sections:
            raise ValueError("lexical reserve exceeds shortlist capacity")

    def _plan(self, query, routes, eligible_count, scope, limit, reserve):
        return SectionRoutePlan(self.hierarchy.receipt_sha256, quote_sha256(query), tuple(routes),
            eligible_count, scope, limit, routing_backend="summary_hybrid" if reserve else "summary_dense")

    def route_vector(self, query: str, vector: np.ndarray, *, embedding_identity: str,
                     max_sections: int = 8, lexical_reserve: int = 0,
                     eligible_source_ids: Sequence[str] | None = None) -> SectionRoutePlan:
        exact_text(query, "query")
        self._limits(max_sections, lexical_reserve)
        scope = self._scope(eligible_source_ids)
        if embedding_identity != self.embedding_identity:
            raise ValueError("query vector encoder identity changed")
        scores = self._dense.score_all(vector)
        eligible = [i for i, s in enumerate(self.sections) if scope is None or s.source_id in scope]
        # Exact similarity sorting has an explicit stable section-ID tie break.
        ordered = sorted(eligible, key=lambda i: (-float(scores[i]), self.sections[i].section_id))
        routes = []
        if lexical_reserve:
            sparse = self._lexical.route(query, max_sections=lexical_reserve, eligible_source_ids=scope)
            routes.extend(sparse.routes)
        seen = {r.section.section_id for r in routes}
        for i in ordered:
            section = self.sections[i]
            if len(routes) >= max_sections:
                break
            if section.section_id not in seen:
                routes.append(SectionRoute(section, 1.0 / (len(routes) + 1), ()))
                seen.add(section.section_id)
        return self._plan(query, routes, len(eligible), scope, max_sections, lexical_reserve)
