"""Attention search over contextual cards followed by raw-memory hydration."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from memory_condense.associations.head_memory_models import (
    AssociativeMemoryCandidate,
    NestedMemoryInspection,
)
from memory_condense.associations.qwen_memory_linker import QwenMemoryLinker
from memory_condense.domain._discourse_identity import quote_sha256
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.search.context_cards import (
    ContextCard,
    ContextMemory,
    ContextSupport,
)


@dataclass(frozen=True, slots=True)
class HydratedContextEvidence:
    card_id: str
    memory_id: str
    source_id: str
    ordinal: int
    text: str
    text_sha256: str
    qk_score: float
    ov_transport: float
    selection_score: float | None = None
    selection_score_kind: str = "qk_ov_components"
    selection_backend: str = "qwen_memory_linker"
    qk_ov_measured: bool = True


@dataclass(frozen=True, slots=True)
class ContextCardSearchResult:
    inspection: NestedMemoryInspection | None
    selected_card_ids: tuple[str, ...]
    evidence: tuple[HydratedContextEvidence, ...]
    omitted_memory_ids: tuple[str, ...]
    requires_raw_fallback: bool
    eligible_source_ids: tuple[str, ...] = ()
    covered_source_ids: tuple[str, ...] = ()
    uncovered_source_ids: tuple[str, ...] = ()
    eligible_target_memory_ids: tuple[str, ...] = ()
    covered_target_memory_ids: tuple[str, ...] = ()
    uncovered_target_memory_ids: tuple[str, ...] = ()
    attention_overflow: bool = False
    source_scope_complete: bool = False


@dataclass(frozen=True, slots=True)
class ContextCardSourceRoute:
    source_id: str
    score: float
    matched_terms: tuple[str, ...]


_TERM = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "for",
        "how",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "what",
        "when",
        "which",
        "who",
    }
)


def _lexical_terms(text: str) -> tuple[str, ...]:
    terms: list[str] = []
    for raw in _TERM.findall(text.casefold()):
        if raw in _STOP or len(raw) < 2:
            continue
        term = raw
        for suffix in ("ing", "ed", "es", "s"):
            if term.endswith(suffix) and len(term) > len(suffix) + 3:
                term = term[: -len(suffix)]
                break
        terms.append(term)
    return tuple(terms)


def lexical_route_context_card_sources(
    query: str,
    cards: Sequence[ContextCard],
    *,
    max_sources: int = 1,
    eligible_source_ids: Sequence[str] | None = None,
) -> tuple[ContextCardSourceRoute, ...]:
    """Cheaply gate source groups before transformer attention.

    This is a shadow source router, not a replacement for the production
    hybrid source gate. It lets the assay preserve the intended pipeline
    boundary when only a sidecar card catalog is available.
    """

    normalized_query = str(query).strip()
    if not normalized_query:
        raise ValueError("query must be non-empty")
    if isinstance(max_sources, bool) or int(max_sources) < 1:
        raise ValueError("max_sources must be positive")
    if eligible_source_ids is None:
        source_terms: dict[str, list[str]] = {}
    else:
        ordered_eligible = tuple(str(source).strip() for source in eligible_source_ids)
        if any(not source for source in ordered_eligible):
            raise ValueError("eligible_source_ids entries must be non-empty")
        if len(ordered_eligible) != len(set(ordered_eligible)):
            raise ValueError("eligible_source_ids entries must be unique")
        source_terms = {source: [] for source in ordered_eligible}
    for card in cards:
        if card.routing_text and (
            eligible_source_ids is None or card.source_id in source_terms
        ):
            source_terms.setdefault(card.source_id, []).extend(
                _lexical_terms(card.routing_text)
            )
    query_terms = tuple(dict.fromkeys(_lexical_terms(normalized_query)))
    source_sets = {source: set(terms) for source, terms in source_terms.items()}
    source_count = len(source_sets)
    routes: list[ContextCardSourceRoute] = []
    for source, terms in source_terms.items():
        frequencies = Counter(terms)
        counts = {term: frequencies[term] for term in query_terms if term in frequencies}
        matched = tuple(counts)
        score = 0.0
        for term, frequency in counts.items():
            document_frequency = sum(term in values for values in source_sets.values())
            inverse_frequency = math.log(
                1.0 + (source_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            score += inverse_frequency * (1.0 + math.log(frequency))
        if score > 0.0:
            routes.append(ContextCardSourceRoute(source, score, matched))
    routes.sort(key=lambda route: (-route.score, route.source_id))
    if routes:
        return tuple(routes[: int(max_sources)])
    # A lexical miss must broaden, never declare absence.
    return tuple(
        ContextCardSourceRoute(source, 0.0, ()) for source in sorted(source_terms)
    )


def attention_search_context_cards(
    query: str,
    cards: Sequence[ContextCard],
    memories: Sequence[ContextMemory],
    *,
    linker: QwenMemoryLinker,
    source_order: Sequence[str],
    eligible_target_memory_ids: Sequence[str] | None = None,
    source_scope_complete: bool = False,
    group_size: int = 8,
    beam_per_group: int = 2,
    top_k: int = 8,
    exclude_memory_ids: Sequence[str] = (),
    max_raw_chunks: int = 16,
    max_raw_tokens: int = 2048,
) -> ContextCardSearchResult:
    """Select cards first, then exclude/dedupe and hydrate complete raw chunks.

    ``eligible_target_memory_ids`` names the raw memories whose usable-card
    coverage the caller expects. When omitted, every supplied memory in an
    eligible source is treated as a target. Any target without a non-empty card
    makes ``requires_raw_fallback`` true, so a partial card sidecar can only be
    used as an additive route.
    """

    normalized_query = str(query).strip()
    if not normalized_query:
        raise ValueError("query must be non-empty")
    if linker.max_candidates < 2:
        raise ValueError("nested card search requires max_candidates >= 2")
    if not 1 <= group_size <= linker.max_candidates:
        raise ValueError("group_size must be in [1, linker.max_candidates]")
    if not 1 <= beam_per_group < linker.max_candidates:
        raise ValueError("beam_per_group must be in [1, linker.max_candidates)")
    if top_k < 1 or max_raw_chunks < 1 or max_raw_tokens < 1:
        raise ValueError("top_k and hydration budgets must be positive")
    if not isinstance(source_scope_complete, bool):
        raise TypeError("source_scope_complete must be bool")

    ordered_sources = tuple(str(source).strip() for source in source_order)
    if any(not source for source in ordered_sources):
        raise ValueError("source_order entries must be non-empty")
    if len(ordered_sources) != len(set(ordered_sources)):
        raise ValueError("source_order entries must be unique")
    eligible_sources = set(ordered_sources)

    memory_by_id: dict[str, ContextMemory] = {}
    for memory in memories:
        if memory.memory_id in memory_by_id:
            raise ValueError(f"duplicate memory_id: {memory.memory_id}")
        memory_by_id[memory.memory_id] = memory

    source_rank = {source: index for index, source in enumerate(ordered_sources)}
    if eligible_target_memory_ids is None:
        eligible_targets = tuple(
            memory.memory_id
            for memory in sorted(
                (
                    memory
                    for memory in memories
                    if memory.source_id in eligible_sources
                ),
                key=lambda memory: (
                    source_rank[memory.source_id],
                    memory.ordinal,
                    memory.turn_start_char,
                    memory.memory_id,
                ),
            )
        )
    else:
        eligible_targets = tuple(
            str(memory_id).strip() for memory_id in eligible_target_memory_ids
        )
        if any(not memory_id for memory_id in eligible_targets):
            raise ValueError("eligible_target_memory_ids entries must be non-empty")
        if len(eligible_targets) != len(set(eligible_targets)):
            raise ValueError("eligible_target_memory_ids entries must be unique")
        for memory_id in eligible_targets:
            memory = memory_by_id.get(memory_id)
            if memory is None:
                raise ValueError(f"unknown eligible target memory: {memory_id}")
            if memory.source_id not in eligible_sources:
                raise ValueError(
                    f"eligible target memory is outside source_order: {memory_id}"
                )
    eligible_target_set = set(eligible_targets)

    card_by_id: dict[str, ContextCard] = {}
    cards_by_source = {source: [] for source in ordered_sources}
    seen_card_ids: set[str] = set()
    for card in cards:
        if card.card_id in seen_card_ids:
            raise ValueError(f"duplicate card_id: {card.card_id}")
        seen_card_ids.add(card.card_id)
        target = memory_by_id.get(card.target_memory_id)
        if (
            card.source_id not in eligible_sources
            or not card.routing_text
            or not card.facts
            or card.target_memory_id not in eligible_target_set
            or target is None
            or target.source_id != card.source_id
        ):
            continue
        card_by_id[card.card_id] = card
        cards_by_source[card.source_id].append(card)

    covered_target_set = {card.target_memory_id for card in card_by_id.values()}
    covered_targets = tuple(
        memory_id for memory_id in eligible_targets if memory_id in covered_target_set
    )
    uncovered_targets = tuple(
        memory_id for memory_id in eligible_targets if memory_id not in covered_target_set
    )
    targets_by_source = {
        source: tuple(
            memory_id
            for memory_id in eligible_targets
            if memory_by_id[memory_id].source_id == source
        )
        for source in ordered_sources
    }
    covered_sources = tuple(
        source
        for source in ordered_sources
        if targets_by_source[source]
        and all(
            memory_id in covered_target_set for memory_id in targets_by_source[source]
        )
    )
    covered_source_set = set(covered_sources)
    uncovered_sources = tuple(
        source for source in ordered_sources if source not in covered_source_set
    )
    coverage_incomplete = bool(
        uncovered_targets or uncovered_sources or not source_scope_complete
    )

    def candidate_groups(size: int) -> list[tuple[AssociativeMemoryCandidate, ...]]:
        groups: list[tuple[AssociativeMemoryCandidate, ...]] = []
        for source in ordered_sources:
            source_cards = sorted(
                cards_by_source[source],
                key=lambda card: (
                    memory_by_id[card.target_memory_id].ordinal,
                    memory_by_id[card.target_memory_id].turn_start_char,
                    card.card_id,
                ),
            )
            for start in range(0, len(source_cards), size):
                groups.append(
                    tuple(
                        AssociativeMemoryCandidate(
                            episode_id=card.card_id,
                            text=card.routing_text,
                            route="context_card",
                            metadata={
                                "card_id": card.card_id,
                                "card_receipt_sha256": card.receipt_sha256,
                                "source_id": card.source_id,
                            },
                        )
                        for card in source_cards[start : start + size]
                    )
                )
        return groups

    groups = candidate_groups(group_size)
    if not groups:
        return ContextCardSearchResult(
            inspection=None,
            selected_card_ids=(),
            evidence=(),
            omitted_memory_ids=(),
            requires_raw_fallback=True,
            eligible_source_ids=ordered_sources,
            covered_source_ids=covered_sources,
            uncovered_source_ids=uncovered_sources,
            eligible_target_memory_ids=eligible_targets,
            covered_target_memory_ids=covered_targets,
            uncovered_target_memory_ids=uncovered_targets,
            source_scope_complete=source_scope_complete,
        )

    effective_group_size = group_size
    linker_requires_raw_fallback = False
    while True:
        try:
            inspection = linker.inspect_nested(
                normalized_query,
                groups,
                beam_per_group=beam_per_group,
                top_k=min(top_k, len(card_by_id)),
                score_mode="qk_ov",
            )
            fallback_signal = getattr(linker, "requires_raw_fallback", False)
            if type(fallback_signal) is not bool:
                raise TypeError("linker requires_raw_fallback must be bool")
            linker_requires_raw_fallback = fallback_signal
            break
        except MemoryError:
            if effective_group_size == 1:
                return ContextCardSearchResult(
                    inspection=None,
                    selected_card_ids=(),
                    evidence=(),
                    omitted_memory_ids=(),
                    requires_raw_fallback=True,
                    eligible_source_ids=ordered_sources,
                    covered_source_ids=covered_sources,
                    uncovered_source_ids=uncovered_sources,
                    eligible_target_memory_ids=eligible_targets,
                    covered_target_memory_ids=covered_targets,
                    uncovered_target_memory_ids=uncovered_targets,
                    attention_overflow=True,
                    source_scope_complete=source_scope_complete,
                )
            effective_group_size = max(1, effective_group_size // 2)
            groups = candidate_groups(effective_group_size)
    selected_ids = tuple(hit.episode_id for hit in inspection.hits)
    if len(selected_ids) != len(set(selected_ids)) or any(
        card_id not in card_by_id for card_id in selected_ids
    ):
        raise RuntimeError("attention linker returned duplicate or foreign card IDs")

    # Selection is deliberately complete before S0/EM overlap is excluded.
    excluded = {str(memory_id) for memory_id in exclude_memory_ids}
    seen: set[str] = set()
    evidence: list[HydratedContextEvidence] = []
    omitted: list[str] = []
    used_tokens = 0

    def append_omitted(memory_id: str) -> None:
        if memory_id not in omitted:
            omitted.append(memory_id)

    def checked_support_memory(
        card: ContextCard, support: ContextSupport
    ) -> ContextMemory:
        memory_id = support.memory_id
        memory = memory_by_id.get(memory_id)
        if memory is None:
            raise RuntimeError(f"card support is unavailable: {memory_id}")
        actual_hash = quote_sha256(memory.text)
        if (
            actual_hash != support.memory_text_sha256
            or actual_hash != memory.text_sha256
            or memory.source_id != card.source_id
            or support.source_id != memory.source_id
            or support.ordinal != memory.ordinal
        ):
            raise RuntimeError(f"card support changed: {memory_id}")
        return memory

    def append_evidence(
        card: ContextCard,
        hit: object,
        memory: ContextMemory,
    ) -> None:
        nonlocal used_tokens
        actual_hash = quote_sha256(memory.text)
        metadata = getattr(hit, "metadata", {})
        measured = metadata.get("qk_ov_measured", True)
        if type(measured) is not bool:
            raise RuntimeError("linker qk_ov_measured metadata must be bool")
        score = metadata.get("distilled_student_score")
        if score is not None:
            score = float(score)
            if not math.isfinite(score):
                raise RuntimeError("linker selection score must be finite")
        score_kind = str(
            metadata.get(
                "score_kind",
                "qk_ov_components" if measured else "unscored_raw_fallback",
            )
        ).strip()
        backend = str(
            metadata.get(
                "selection_backend",
                "qwen_memory_linker" if measured else "unknown_fail_open",
            )
        ).strip()
        if not score_kind or not backend:
            raise RuntimeError("linker selection metadata must be non-empty")
        seen.add(memory.memory_id)
        used_tokens += count_tokens(memory.text)
        evidence.append(
            HydratedContextEvidence(
                card_id=card.card_id,
                memory_id=memory.memory_id,
                source_id=memory.source_id,
                ordinal=memory.ordinal,
                text=memory.text,
                text_sha256=actual_hash,
                qk_score=float(getattr(hit, "qk_score")),
                ov_transport=float(getattr(hit, "ov_transport")),
                selection_score=score,
                selection_score_kind=score_kind,
                selection_backend=backend,
                qk_ov_measured=measured,
            )
        )

    for hit in inspection.hits:
        card = card_by_id[hit.episode_id]
        # Validate the complete sealed support window before any exclusion or
        # deduplication can hide a stale/missing chunk from integrity checks.
        resolved_support = {
            support.memory_id: checked_support_memory(card, support)
            for support in card.support
        }
        if len(resolved_support) != len(card.support):
            raise RuntimeError("card support contains duplicate memory IDs")
        required_ids = tuple(
            dict.fromkeys(
                (
                    card.target_memory_id,
                    *(
                        citation.memory_id
                        for fact in card.facts
                        for citation in fact.citations
                    ),
                )
            )
        )
        if any(memory_id not in resolved_support for memory_id in required_ids):
            raise RuntimeError("card citation is absent from sealed support")
        required_memories = [
            resolved_support[memory_id]
            for memory_id in required_ids
            if memory_id not in excluded and memory_id not in seen
        ]
        required_tokens = sum(count_tokens(memory.text) for memory in required_memories)
        if (
            len(evidence) + len(required_memories) > max_raw_chunks
            or used_tokens + required_tokens > max_raw_tokens
        ):
            for memory in required_memories:
                append_omitted(memory.memory_id)
            required_id_set = set(required_ids)
            for support in reversed(card.support):
                if (
                    support.memory_id not in required_id_set
                    and support.memory_id not in excluded
                    and support.memory_id not in seen
                ):
                    append_omitted(support.memory_id)
            # Never spend the remaining budget on incidental neighborhood when
            # the selected card's target/citation unit could not be preserved.
            continue
        for memory in required_memories:
            append_evidence(card, hit, memory)

        # Once the target and every cited raw memory are secure, walk backward
        # through the remaining bounded local neighborhood.
        for support in reversed(card.support):
            memory_id = support.memory_id
            if memory_id in excluded or memory_id in seen:
                continue
            memory = resolved_support[memory_id]
            token_count = count_tokens(memory.text)
            if (
                len(evidence) >= max_raw_chunks
                or used_tokens + token_count > max_raw_tokens
            ):
                append_omitted(memory_id)
                continue
            append_evidence(card, hit, memory)
    return ContextCardSearchResult(
        inspection=inspection,
        selected_card_ids=selected_ids,
        evidence=tuple(evidence),
        omitted_memory_ids=tuple(omitted),
        requires_raw_fallback=(
            coverage_incomplete
            or bool(omitted)
            or linker_requires_raw_fallback
        ),
        eligible_source_ids=ordered_sources,
        covered_source_ids=covered_sources,
        uncovered_source_ids=uncovered_sources,
        eligible_target_memory_ids=eligible_targets,
        covered_target_memory_ids=covered_targets,
        uncovered_target_memory_ids=uncovered_targets,
        source_scope_complete=source_scope_complete,
    )


__all__ = [
    "ContextCardSourceRoute",
    "ContextCardSearchResult",
    "HydratedContextEvidence",
    "attention_search_context_cards",
    "lexical_route_context_card_sources",
]
