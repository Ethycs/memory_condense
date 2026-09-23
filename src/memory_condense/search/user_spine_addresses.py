"""Separate user-spine semantic addresses from the attached assistant summaries.

The original section descriptors remain the hydration authority. Both address
matrices contain summaries only; the query is encoded once and shared by them.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math

import numpy as np

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.search.hot_retrieval import ExactDenseAddressIndex
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan, SectionSummaryIndex
from memory_condense.search.section_summary import exact_text
from memory_condense.search.summary_semantic_index import SemanticSectionIndex, summary_embedding_identity
from memory_condense.domain._discourse_identity import quote_sha256


def user_spine_text(section):
    body = json.loads(section.summary)
    if type(body) is not dict or set(body) != {
        "user_spine", "attached_context_not_user_assertions", "transcript_date_range"
    }:
        raise ValueError("user addresses require structured spine summaries")
    spine = body["user_spine"]
    return "Unowned prelude." if spine is None else exact_text(spine, "user spine")


class UserSpineAddressIndex:
    __slots__ = ("hierarchy", "sections", "embedding_identity", "_spine", "_combined", "_lexical", "receipt_sha256", "_locked")

    def __setattr__(self, name, value):
        if getattr(self, "_locked", False):
            raise AttributeError("user-spine address indexes are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, hierarchy, combined_matrix, spine_matrix, *, embedding_identity):
        if type(hierarchy) is not SectionSummaryIndex:
            raise TypeError("user-spine addresses require an authenticated hierarchy")
        self.hierarchy = hierarchy
        self.sections = tuple(s for s in hierarchy.sections if not s.child_section_ids)
        if not self.sections:
            raise ValueError("user-spine addresses require leaves")
        self.embedding_identity = exact_text(embedding_identity, "embedding_identity")
        ids = [s.section_id for s in self.sections]
        self._spine = ExactDenseAddressIndex(ids, spine_matrix)
        self._combined = ExactDenseAddressIndex(ids, combined_matrix)
        if self._spine.dimension != self._combined.dimension:
            raise ValueError("summary channel dimensions differ")
        views = tuple(replace(s, summary=user_spine_text(s), receipt_sha256="") for s in self.sections)
        self._lexical = SectionSummaryIndex(views)
        self.receipt_sha256 = identity_sha256({"format": "memory-condense-user-spine-addresses-v1",
            "hierarchy_sha256": hierarchy.receipt_sha256, "embedding_identity": embedding_identity,
            "section_ids": ids, "spine_summary_sha256s": [quote_sha256(s.summary) for s in views],
            "spine_matrix_sha256": hashlib.sha256(np.asarray(spine_matrix).tobytes(order="C")).hexdigest(),
            "combined_matrix_sha256": hashlib.sha256(np.asarray(combined_matrix).tobytes(order="C")).hexdigest()})
        self._locked = True

    def route(self, query, *, encoder, user_weight=0.8, max_sections=6, lexical_reserve=0, eligible_source_ids=None):
        exact_text(query, "query")
        if summary_embedding_identity(encoder) != self.embedding_identity:
            raise ValueError("query encoder does not match user-spine addresses")
        self._limits(user_weight, max_sections, lexical_reserve)
        scope = SemanticSectionIndex._scope(eligible_source_ids)
        vector = encoder.embed_query(query)
        if summary_embedding_identity(encoder) != self.embedding_identity:
            raise ValueError("query encoder identity changed")
        return self.route_vector(query, vector, embedding_identity=self.embedding_identity, user_weight=user_weight,
            max_sections=max_sections, lexical_reserve=lexical_reserve, eligible_source_ids=scope)

    @staticmethod
    def _limits(weight, maximum, reserve):
        if isinstance(weight, bool) or not isinstance(weight, (float, int)) or not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("user channel weight must lie in [0,1]")
        SemanticSectionIndex._limits(maximum, reserve)

    def route_vector(self, query, vector, *, embedding_identity, user_weight=0.8, max_sections=6,
                     lexical_reserve=0, eligible_source_ids=None):
        exact_text(query, "query")
        self._limits(user_weight, max_sections, lexical_reserve)
        scope = SemanticSectionIndex._scope(eligible_source_ids)
        if embedding_identity != self.embedding_identity:
            raise ValueError("query vector encoder changed")
        scores = user_weight * self._spine.score_all(vector) + (1 - user_weight) * self._combined.score_all(vector)
        eligible = [i for i, s in enumerate(self.sections) if scope is None or s.source_id in scope]
        ordered = sorted(eligible, key=lambda i: (-float(scores[i]), self.sections[i].section_id))
        selected = []
        if lexical_reserve:
            sparse = self._lexical.route(query, max_sections=lexical_reserve, eligible_source_ids=scope)
            selected.extend(r.section.section_id for r in sparse.routes)
        for i in ordered:
            if len(selected) >= max_sections:
                break
            section_id = self.sections[i].section_id
            if section_id not in selected:
                selected.append(section_id)
        originals = {s.section_id: s for s in self.sections}
        routes = tuple(SectionRoute(originals[section_id], 1 / (i + 1), ()) for i, section_id in enumerate(selected))
        return SectionRoutePlan(self.hierarchy.receipt_sha256, quote_sha256(query), routes, len(eligible),
            scope, max_sections, routing_backend="summary_hybrid" if lexical_reserve else "summary_dense")
