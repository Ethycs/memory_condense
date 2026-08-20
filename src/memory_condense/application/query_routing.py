"""Pure query and source-routing helpers for the condenser facade."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from memory_condense.domain.ranking import round_robin_unique
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.search.indexes.lexical import tokenize


SAFE_ASSOCIATION_LEXICAL_THRESHOLD = 0.9
SAFE_ASSOCIATION_MAX_TOKEN_INCREASE = 0

_DATED_QUESTION_RE = re.compile(r"^\[Question asked at .+?\]\s*", re.DOTALL)
_FACET_SPLIT_RE = re.compile(r"\s*,\s*(?:and\s+)?|\s+and\s+", re.IGNORECASE)


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
        normalized_margin = (margin - low) / (high - low) if high > low else 0.0
        score = 0.8 * lexical + 0.2 * normalized_margin
        ranked.append((score, -index, result.model_copy(update={"score": score})))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [result for _score, _index, result in ranked]


def query_facets(query: str, *, max_facets: int = 4) -> list[str]:
    """Extract explicit list facets for bounded multi-query retrieval."""

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
    """Prefer user evidence for explicitly autobiographical questions."""

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
        ranked.append((adjusted, index, result.model_copy(update={"score": adjusted})))
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
    return result.durable_source_id


def source_diverse_results(
    candidates: Sequence[RetrievalResult],
) -> list[RetrievalResult]:
    """Round-robin ranked chunks by durable source, preserving local order."""

    groups: dict[str, list[RetrievalResult]] = {}
    for result in candidates:
        groups.setdefault(result.source_key, []).append(result)
    return round_robin_unique(list(groups.values()))


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
    """Rank coarse partitions by reciprocal-rank heat over chunk hits."""

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
    """Expose the bounded coarse-cue evidence used for partition routing."""

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
        source_id = result.source_key
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
        key=lambda partition: (-scores[partition], first_rank[partition], partition),
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


__all__ = [
    "SAFE_ASSOCIATION_LEXICAL_THRESHOLD",
    "SAFE_ASSOCIATION_MAX_TOKEN_INCREASE",
    "is_multi_fact_query",
    "query_facets",
    "rank_concept_members",
    "role_aware_results",
    "select_source_partitions",
    "source_diverse_results",
    "source_partition_ranking",
]
