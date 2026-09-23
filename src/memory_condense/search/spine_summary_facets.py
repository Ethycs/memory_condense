"""Additional semantic addresses for exact passages inside user-spine summaries.

Facet boundaries operate on generated summaries only. Every address maps back
to its unchanged leaf descriptor; answer evidence still requires raw hydration.
Whole-summary addresses remain available through the existing user index.
"""
from dataclasses import dataclass
import hashlib
import re
from types import MappingProxyType

import numpy as np

from memory_condense.domain._discourse_identity import identity_sha256, quote_sha256
from memory_condense.search.hot_retrieval import ExactDenseAddressIndex
from memory_condense.search.section_routing import SectionRoute, SectionRoutePlan, SectionSummaryIndex
from memory_condense.search.section_summary import exact_text
from memory_condense.search.user_spine_addresses import user_spine_text


_BOUNDARY = re.compile(
    r"(?<=[.!?;])\s+|\s+(?:and|but)\s+(?=(?:asks?|requests?|wonders?|considers?|wants?|seeks?|they|the user|user|he|she)\b)",
    re.IGNORECASE,
)
_WORDS = re.compile(r"\S+")


@dataclass(frozen=True)
class SummaryFacet:
    facet_id: str
    section_id: str
    start_char: int
    end_char: int
    text: str

    def identity_payload(self):
        return {"facet_id": self.facet_id, "section_id": self.section_id,
            "start_char": self.start_char, "end_char": self.end_char,
            "text_sha256": quote_sha256(self.text)}


def summary_facets(section, *, max_words=48, stride_words=24):
    if type(max_words) is not int or type(stride_words) is not int or not 1 <= stride_words <= max_words:
        raise ValueError("facet window and stride must be positive bounded integers")
    text = user_spine_text(section)
    cuts = [(0, 0), *((m.start(), m.end()) for m in _BOUNDARY.finditer(text)), (len(text), len(text))]
    intervals = []
    for left, right in zip(cuts, cuts[1:]):
        start, end = left[1], right[0]
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start == end:
            continue
        words = list(_WORDS.finditer(text, start, end))
        for offset in range(0, len(words), stride_words):
            last = min(len(words), offset + max_words)
            intervals.append((words[offset].start(), words[last - 1].end()))
            if last == len(words):
                break
    return tuple(SummaryFacet(f"{section.section_id}:{start:06d}:{end:06d}", section.section_id,
        start, end, text[start:end]) for start, end in sorted(set(intervals)))


class SpineFacetAddressIndex:
    """Max-passage similarity supplements the unchanged whole-summary route."""
    __slots__ = ("hierarchy", "sections", "by_id", "facets", "embedding_identity", "_dense", "receipt_sha256", "_locked")

    def __setattr__(self, name, value):
        if getattr(self, "_locked", False):
            raise AttributeError("summary facet addresses are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, hierarchy, matrix, *, embedding_identity):
        if type(hierarchy) is not SectionSummaryIndex:
            raise TypeError("facet addresses require an authenticated summary hierarchy")
        exact_text(embedding_identity, "embedding_identity")
        self.hierarchy = hierarchy
        self.sections = tuple(s for s in hierarchy.sections if not s.child_section_ids)
        self.by_id = MappingProxyType({s.section_id: s for s in self.sections})
        self.facets = tuple(f for s in self.sections for f in summary_facets(s))
        self.embedding_identity = embedding_identity
        self._dense = ExactDenseAddressIndex([f.facet_id for f in self.facets], matrix)
        self.receipt_sha256 = identity_sha256({"format": "memory-condense-spine-summary-facet-addresses-v1",
            "hierarchy_sha256": hierarchy.receipt_sha256, "embedding_identity": embedding_identity,
            "facets": [f.identity_payload() for f in self.facets],
            "matrix_sha256": hashlib.sha256(np.asarray(matrix).tobytes()).hexdigest(),
            "max_words": 48, "stride_words": 24, "aggregation": "maximum passage cosine per original leaf"})
        self._locked = True

    def route_vector(self, query, vector, *, embedding_identity, max_sections=8):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonempty text")
        if embedding_identity != self.embedding_identity:
            raise ValueError("query and facet embedding identities differ")
        if type(max_sections) is not int or max_sections < 1:
            raise ValueError("facet result limit must be a positive integer")
        values = self._dense.score_all(vector)
        best = {}
        for facet, score in zip(self.facets, values, strict=True):
            if facet.section_id not in best or float(score) > best[facet.section_id][0]:
                best[facet.section_id] = (float(score), facet)
        ordered = sorted(best, key=lambda sid: (-best[sid][0], sid))[:max_sections]
        routes = tuple(SectionRoute(self.by_id[sid], 1 / (i + 1), ()) for i, sid in enumerate(ordered))
        plan = SectionRoutePlan(self.hierarchy.receipt_sha256, quote_sha256(query), routes,
            len(self.sections), None, max_sections, routing_backend="summary_dense")
        return plan, [{"section_id": sid, "score": best[sid][0],
                       "winning_facet": best[sid][1].identity_payload()} for sid in ordered]
