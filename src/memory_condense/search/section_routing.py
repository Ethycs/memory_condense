"""Immutable summary hierarchy and route receipts, with a BM25 control route."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, log
from types import MappingProxyType

from memory_condense.domain._discourse_identity import canonical_json, identity_sha256, quote_sha256
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.search.indexes.lexical import BM25_B, BM25_K1, tokenize
from memory_condense.search.section_summary import SectionSummary, bound_int, exact_text


INDEX_FORMAT = "memory-condense-section-summary-index-v1"


@dataclass(frozen=True, slots=True)
class SummaryAttentionPass(SealedIdentity):
    candidate_section_ids: tuple[str, ...]
    selected_section_ids: tuple[str, ...]
    selected_qk_scores: tuple[float, ...]
    selected_ov_transport: tuple[float, ...]
    model_passes: int
    max_workspace_candidates: int
    max_workspace_tokens: int
    candidate_inspections: int
    receipt_sha256: str = ""

    def __post_init__(self):
        for name in ("candidate_section_ids", "selected_section_ids", "selected_qk_scores", "selected_ov_transport"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(self.selected_section_ids) != len(self.selected_qk_scores) or len(self.selected_section_ids) != len(self.selected_ov_transport):
            raise ValueError("attention scores must bind every selected section")
        if not set(self.selected_section_ids) <= set(self.candidate_section_ids):
            raise ValueError("attention selected an uninspected section")
        if len(set(self.candidate_section_ids)) != len(self.candidate_section_ids) or len(set(self.selected_section_ids)) != len(self.selected_section_ids):
            raise ValueError("attention section IDs must be unique")
        if any(not isfinite(score) for score in (*self.selected_qk_scores, *self.selected_ov_transport)):
            raise ValueError("attention scores must be finite")
        for name in ("model_passes", "max_workspace_candidates", "max_workspace_tokens", "candidate_inspections"):
            bound_int(getattr(self, name), name, 1)
        self._seal()


@dataclass(frozen=True, slots=True)
class SummaryAttentionReceipt(SealedIdentity):
    linker_identity_json: str
    rounds: tuple[SummaryAttentionPass, ...]
    max_depth: int
    score_mode: str = "qk_ov"
    retained_transformer_token_state_bytes: int = 0
    raw_content_inspections: int = 0
    receipt_sha256: str = ""

    def __post_init__(self):
        object.__setattr__(self, "rounds", tuple(self.rounds))
        bound_int(self.max_depth, "max_depth", 1)
        if len(self.rounds) > self.max_depth or self.score_mode != "qk_ov":
            raise ValueError("invalid summary attention traversal")
        if self.retained_transformer_token_state_bytes != 0 or self.raw_content_inspections != 0:
            raise ValueError("Qwen routing may inspect summaries only and retain no token state")
        self._seal()


@dataclass(frozen=True, slots=True)
class SectionRoute(SealedIdentity):
    section: SectionSummary
    score: float
    matched_terms: tuple[str, ...]
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.section, SectionSummary) or not 0 < self.score < float("inf"):
            raise ValueError("section routes require a section and a positive finite score")
        object.__setattr__(self, "matched_terms", tuple(self.matched_terms))
        self._seal()


@dataclass(frozen=True, slots=True)
class SummaryReasoningPass(SealedIdentity):
    candidate_section_ids: tuple[str, ...]
    selected_section_ids: tuple[str, ...]
    prompt_sha256: str
    response_sha256: str
    prompt_token_proxy: int
    receipt_sha256: str = ""

    def __post_init__(self):
        for name in ("candidate_section_ids", "selected_section_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            if len(set(getattr(self, name))) != len(getattr(self, name)):
                raise ValueError("reasoning section IDs must be unique")
        if not set(self.selected_section_ids) <= set(self.candidate_section_ids):
            raise ValueError("reasoning selected an uninspected section")
        bound_int(self.prompt_token_proxy, "prompt_token_proxy", 1)
        self._seal()


@dataclass(frozen=True, slots=True)
class SummaryReasoningReceipt(SealedIdentity):
    model_id: str
    gateway_url: str
    passes: tuple[SummaryReasoningPass, ...]
    hierarchy_rounds: int
    max_depth: int
    max_calls: int
    group_size: int
    max_prompt_tokens: int
    raw_content_inspections: int = 0
    retained_local_transformer_token_state_bytes: int = 0
    receipt_sha256: str = ""

    def __post_init__(self):
        object.__setattr__(self, "passes", tuple(self.passes))
        exact_text(self.model_id, "model_id")
        if "qwen" not in self.model_id.casefold():
            raise ValueError("summary reasoning requires a Qwen model")
        for name in ("max_depth", "max_calls", "group_size", "max_prompt_tokens"):
            bound_int(getattr(self, name), name, 1)
        bound_int(self.hierarchy_rounds, "hierarchy_rounds")
        if self.hierarchy_rounds > self.max_depth or len(self.passes) > self.max_calls:
            raise ValueError("summary reasoning exceeded its traversal bounds")
        if any(len(row.candidate_section_ids) > self.group_size or row.prompt_token_proxy > self.max_prompt_tokens
               for row in self.passes):
            raise ValueError("summary reasoning exceeded its prompt bounds")
        if self.raw_content_inspections != 0 or self.retained_local_transformer_token_state_bytes != 0:
            raise ValueError("summary reasoning cannot consume raw text or retain local token state")
        self._seal()


@dataclass(frozen=True, slots=True)
class SectionRoutePlan(SealedIdentity):
    index_sha256: str
    query_sha256: str
    routes: tuple[SectionRoute, ...]
    matched_section_count: int
    eligible_source_ids: tuple[str, ...] | None
    max_sections: int
    frontier_closed: bool = False
    routing_backend: str = "summary_bm25"
    attention_receipt: SummaryAttentionReceipt | None = None
    reasoning_receipt: SummaryReasoningReceipt | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "routes", tuple(self.routes))
        bound_int(self.max_sections, "max_sections", 1)
        bound_int(self.matched_section_count, "matched_section_count")
        if len(self.routes) > self.max_sections or self.matched_section_count < len(self.routes):
            raise ValueError("section route counts are inconsistent")
        if self.frontier_closed is not False:
            raise ValueError("summary routing cannot certify factual closure")
        if self.routing_backend not in {"summary_bm25", "summary_dense", "summary_hybrid", "qwen_hierarchical_summaries", "qwen_summary_reasoning"} or (
            (self.attention_receipt is not None) != (self.routing_backend == "qwen_hierarchical_summaries")
        ):
            raise ValueError("section route backend and attention receipt disagree")
        if (self.reasoning_receipt is not None) != (self.routing_backend == "qwen_summary_reasoning"):
            raise ValueError("section route backend and reasoning receipt disagree")
        if self.eligible_source_ids is not None:
            object.__setattr__(self, "eligible_source_ids", tuple(self.eligible_source_ids))
            if any(r.section.source_id not in self.eligible_source_ids for r in self.routes):
                raise ValueError("route escaped explicit source scope")
        if len({r.section.section_id for r in self.routes}) != len(self.routes):
            raise ValueError("section routes must be unique")
        self._seal()


class SectionSummaryIndex:
    """Immutable routing snapshot; postings contain summary terms only.

    No source ID, section ID, raw text, or character coordinate contributes to
    relevance. The serialized snapshot binds every summary to its raw section.
    """

    __slots__ = ("sections", "receipt_sha256", "_postings", "_lengths", "_avgdl", "_children", "_roots", "_locked")

    def __setattr__(self, name, value):
        if getattr(self, "_locked", False):
            raise AttributeError("section summary indexes are immutable snapshots")
        object.__setattr__(self, name, value)

    def __init__(self, sections: Sequence[SectionSummary]):
        self.sections = tuple(sorted(sections, key=lambda section: section.section_id))
        if len({section.section_id for section in self.sections}) != len(self.sections):
            raise ValueError("section IDs must be unique")
        by_id = {section.section_id: i for i, section in enumerate(self.sections)}
        children = {}
        parented: set[int] = set()
        for i, section in enumerate(self.sections):
            if any(child not in by_id for child in section.child_section_ids):
                raise ValueError("hierarchy contains a missing child section")
            child_indexes = tuple(by_id[child] for child in section.child_section_ids)
            if child_indexes:
                if any(child in parented for child in child_indexes):
                    raise ValueError("a section can have only one parent")
                parented.update(child_indexes)
                if tuple(span for child in child_indexes for span in self.sections[child].spans) != section.spans:
                    raise ValueError("hierarchy children must partition the parent's exact raw spans")
                children[i] = child_indexes
        self._children = MappingProxyType(children)
        self._roots = tuple(i for i in range(len(self.sections)) if i not in parented)
        reachable: set[int] = set()
        pending = list(self._roots)
        while pending:
            i = pending.pop()
            if i in reachable:
                raise ValueError("section hierarchy contains a cycle")
            reachable.add(i)
            pending.extend(children.get(i, ()))
        if len(reachable) != len(self.sections):
            raise ValueError("section hierarchy contains a cycle")
        postings: dict[str, list[tuple[int, int]]] = {}
        lengths = []
        for index, section in enumerate(self.sections):
            terms = tokenize(section.summary)
            lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                postings.setdefault(term, []).append((index, frequency))
        self._postings = MappingProxyType({term: tuple(rows) for term, rows in postings.items()})
        self._lengths = tuple(lengths)
        self._avgdl = sum(lengths) / len(lengths) if lengths else 0.0
        self.receipt_sha256 = identity_sha256(self._body())
        self._locked = True

    def _body(self) -> dict:
        return {"format": INDEX_FORMAT, "sections": [section.identity_payload() for section in self.sections]}

    def to_json(self) -> str:
        return canonical_json({**self._body(), "receipt_sha256": self.receipt_sha256}) + "\n"

    @classmethod
    def from_json(cls, text: str) -> SectionSummaryIndex:
        body = json.loads(text)
        if set(body) != {"format", "sections", "receipt_sha256"} or body["format"] != INDEX_FORMAT:
            raise ValueError("unsupported section index format")
        result = cls(tuple(SectionSummary.from_dict(section) for section in body["sections"]))
        if result.receipt_sha256 != body["receipt_sha256"]:
            raise ValueError("section index receipt changed")
        return result

    def route(self, query: str, *, max_sections: int = 4,
              eligible_source_ids: Sequence[str] | None = None) -> SectionRoutePlan:
        exact_text(query, "query")
        bound_int(max_sections, "max_sections", 1)
        scope = None if eligible_source_ids is None else tuple(eligible_source_ids)
        if scope is not None and (len(set(scope)) != len(scope) or any(not s for s in scope)):
            raise ValueError("source scope must contain unique nonempty exact IDs")
        source_set = None if scope is None else set(scope)
        scores: dict[int, float] = {}
        hits: dict[int, list[str]] = {}
        for term in sorted(set(tokenize(query))):
            posting = self._postings.get(term, ())
            idf = log(1 + (len(self.sections) - len(posting) + 0.5) / (len(posting) + 0.5))
            for index, frequency in posting:
                if source_set is not None and self.sections[index].source_id not in source_set:
                    continue
                norm = BM25_K1 * (1 - BM25_B + BM25_B * self._lengths[index] / self._avgdl)
                scores[index] = scores.get(index, 0.0) + idf * frequency * (BM25_K1 + 1) / (frequency + norm)
                hits.setdefault(index, []).append(term)
        # Descend whenever a child/subtree has a matching summary. A lossy
        # parent summary may not hide a matching descendant. If only a parent
        # matches, retain that whole raw section; never invent a child choice.
        subtree_scores = dict(scores)
        traversal = []
        pending = list(self._roots)
        while pending:
            i = pending.pop()
            traversal.append(i)
            pending.extend(self._children.get(i, ()))
        for i in reversed(traversal):
            subtree_scores[i] = max([scores.get(i, 0.0), *(
                subtree_scores[child] for child in self._children.get(i, ())
            )])
        terminal = []
        pending = list(self._roots)
        while pending:
            i = pending.pop()
            matching_children = [child for child in self._children.get(i, ()) if subtree_scores[child] > 0]
            if matching_children:
                pending.extend(matching_children)
            elif scores.get(i, 0.0) > 0:
                terminal.append(i)
        ordered = sorted(terminal, key=lambda i: (-scores[i], self.sections[i].section_id))
        routes = tuple(SectionRoute(self.sections[i], scores[i], tuple(hits[i]))
                       for i in ordered[:max_sections])
        return SectionRoutePlan(self.receipt_sha256, quote_sha256(query), routes,
                                len(ordered), scope, max_sections)
