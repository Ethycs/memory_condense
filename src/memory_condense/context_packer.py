"""Deterministic, budgeted context assembly.

The point of this module is that context cost is *predictable*: every section
has a hard token ceiling, so a long conversation can never produce a surprise
token spike. Anything that does not fit is dropped and counted, never silently
truncated away without a record.

Section order follows the design:

    1. system / policies
    2. memory header   (typed bullets — active + pinned + top-ranked only)
    3. recent turns    (chronological)
    4. expansions      (verbatim chunk quotes, only when precision matters)
    5. the current user message
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

import pysbd

from memory_condense._tokenizer import count_tokens, truncate_to_tokens
from memory_condense.lexical import tokenize
from memory_condense.schemas import (
    MemoryItem,
    MemoryResult,
    PackedContext,
    RetrievalResult,
)


@dataclass(frozen=True)
class ContextBudget:
    """Hard per-section token ceilings (design defaults)."""

    recent_window_tokens: int = 4500
    memory_header_tokens: int = 900
    expansion_tokens: int = 800
    # Retrieval asks for ten candidates by default.  The token ceiling, not an
    # unrelated count of three, should decide how many of those candidates
    # reach the prompt.  This raised assembled recall in the B0 investigation
    # without increasing the 800-token expansion budget.
    max_expansions: int = 10
    # Learned candidates are additive and may use otherwise-idle token budget;
    # they never consume one of the direct-retrieval slots.
    max_consolidation_expansions: int = 3
    max_expansion_tokens: int = 250
    budget_aware_expansions: bool = False
    # Opt-in: apply diminishing returns to repeated excerpts from the same
    # durable source while performing budget-aware selection.
    source_diverse_expansions: bool = False
    # Opt-in lexical sentence extraction after chunk retrieval. This spends
    # prompt tokens on the sentences most directly tied to the live query
    # while retaining the durable chunk ID for provenance.
    query_aware_sentence_expansions: bool = False
    max_sentences_per_expansion: int = 2
    # Opt-in rate-distortion filter. Candidate-set IDF supplies information
    # weights; relevance and marginal concept/source novelty are divided by
    # rendered token cost without disturbing the retriever's evidence order.
    information_gain_expansions: bool = False
    min_information_gain_per_token: float = 0.0
    # Opt-in: use diffused source heat as weighted-fair prompt exposure. The
    # default preserves the established retrieval ordering exactly.
    heat_weighted_expansions: bool = False
    max_source_expansion_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.max_consolidation_expansions < 0:
            raise ValueError("max_consolidation_expansions must be non-negative")
        if self.max_sentences_per_expansion < 1:
            raise ValueError("max_sentences_per_expansion must be positive")
        if self.min_information_gain_per_token < 0.0:
            raise ValueError("min_information_gain_per_token must be non-negative")
        if not 0.0 < self.max_source_expansion_fraction <= 1.0:
            raise ValueError("max_source_expansion_fraction must lie in (0, 1]")

    def total(self) -> int:
        return (
            self.recent_window_tokens
            + self.memory_header_tokens
            + self.expansion_tokens
        )


MEMORY_HEADER_PREFIX = "Relevant memory:"
EXPANSION_PREFIX = "Supporting excerpts:"


class ContextPacker:
    """Packs memory, recent turns, and expansions into a budgeted message list."""

    def __init__(self, budget: ContextBudget | None = None) -> None:
        self.budget = budget or ContextBudget()
        self._sentence_segmenter = (
            pysbd.Segmenter(language="en", clean=False)
            if self.budget.query_aware_sentence_expansions
            else None
        )

    # -- public API ---------------------------------------------------------

    def pack(
        self,
        system_prompt: str = "",
        memories: list[MemoryResult] | list[MemoryItem] | None = None,
        recent_turns: list[tuple[str, str]] | None = None,
        expansions: list[RetrievalResult] | None = None,
        user_text: str | None = None,
    ) -> PackedContext:
        """Assemble a `PackedContext`. Every section is independently capped."""
        memories = memories or []
        recent_turns = recent_turns or []
        expansions = expansions or []

        header, header_tokens, header_dropped, memory_ids = (
            self._build_memory_header(memories)
        )
        kept_turns, turn_tokens, turns_dropped = self._fit_recent_turns(recent_turns)
        (
            exp_texts,
            expansion_chunk_ids,
            exp_tokens,
            exp_dropped,
            source_tokens,
        ) = self._build_expansions(expansions, query=user_text or "")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if header:
            messages.append({"role": "system", "content": header})
        for role, text in kept_turns:
            messages.append({"role": role, "content": text})
        if exp_texts:
            block = EXPANSION_PREFIX + "\n" + "\n".join(exp_texts)
            messages.append({"role": "system", "content": block})
        if user_text is not None:
            messages.append({"role": "user", "content": user_text})

        token_counts = {
            "system": count_tokens(system_prompt) if system_prompt else 0,
            "memory_header": header_tokens,
            "recent_turns": turn_tokens,
            "expansions": exp_tokens,
            "user": count_tokens(user_text) if user_text else 0,
        }
        dropped = {
            "memories": header_dropped,
            "recent_turns": turns_dropped,
            "expansions": exp_dropped,
        }

        return PackedContext(
            messages=messages,
            memory_header=header,
            memory_ids=memory_ids,
            expansions=exp_texts,
            expansion_chunk_ids=expansion_chunk_ids,
            recent_turns=kept_turns,
            token_counts=token_counts,
            expansion_source_token_counts=source_tokens,
            dropped=dropped,
        )

    # -- section builders ---------------------------------------------------

    def _build_memory_header(
        self, memories: list[MemoryResult] | list[MemoryItem]
    ) -> tuple[str, int, int, list[str]]:
        """Typed bullets, highest-ranked first, capped at the header budget."""
        items = [m.item if isinstance(m, MemoryResult) else m for m in memories]
        active = [i for i in items if i.status.value == "active"]

        if not active:
            return "", 0, 0, []

        lines: list[str] = []
        memory_ids: list[str] = []
        used = count_tokens(MEMORY_HEADER_PREFIX)
        dropped = 0

        for item in active:
            bullet = self._format_memory(item)
            cost = count_tokens(bullet) + 1  # +1 for the newline
            if used + cost > self.budget.memory_header_tokens:
                dropped += 1
                continue
            lines.append(bullet)
            memory_ids.append(item.mem_id)
            used += cost

        if not lines:
            return "", 0, dropped, []

        header = MEMORY_HEADER_PREFIX + "\n" + "\n".join(lines)
        return header, count_tokens(header), dropped, memory_ids

    @staticmethod
    def _format_memory(item: MemoryItem) -> str:
        pin_marker = "*" if item.is_pinned else ""
        line = f"- [{item.type.value}]{pin_marker} {item.content.strip()}"
        if item.details:
            line += f" ({item.details.strip()})"
        return line

    def _fit_recent_turns(
        self, recent_turns: list[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], int, int]:
        """Keep the most recent turns that fit, returned oldest-first."""
        kept: list[tuple[str, str]] = []
        used = 0

        for role, text in reversed(recent_turns):
            cost = count_tokens(text)
            if used + cost > self.budget.recent_window_tokens:
                break
            kept.append((role, text))
            used += cost

        kept.reverse()
        return kept, used, len(recent_turns) - len(kept)

    def _build_expansions(
        self,
        expansions: list[RetrievalResult],
        *,
        query: str = "",
    ) -> tuple[list[str], list[str], int, int, dict[str, int]]:
        """Verbatim excerpts, each capped, and capped again in aggregate.

        The final excerpt is shortened to the remaining aggregate budget.  The
        old implementation dropped it wholesale, often leaving a material
        fraction of the fixed budget unused even though more ranked evidence
        was available.
        """
        ranked = (
            self._heat_weighted_order(expansions)
            if self.budget.heat_weighted_expansions
            else expansions
        )
        if self.budget.information_gain_expansions:
            ranked = self._information_gain_order(ranked, query=query)
        elif self.budget.budget_aware_expansions:
            ranked = self._budget_aware_order(ranked, query=query)
        texts: list[str] = []
        chunk_ids: list[str] = []
        used = count_tokens(EXPANSION_PREFIX)
        source_tokens: dict[str, int] = defaultdict(int)
        direct_kept = 0
        consolidation_kept = 0

        for result in ranked:
            is_consolidation = result.route == "live_consolidation"
            if is_consolidation:
                if consolidation_kept >= self.budget.max_consolidation_expansions:
                    continue
            elif direct_kept >= self.budget.max_expansions:
                continue
            remaining = self.budget.expansion_tokens - used
            label = f"[{len(texts) + 1}] "
            # Reserve the label and the newline accounted for by this packer.
            content_budget = min(
                self.budget.max_expansion_tokens,
                remaining - count_tokens(label) - 1,
            )
            if content_budget <= 0:
                break
            prepared = self._prepare_expansion_text(result.chunk.text, query)
            snippet = truncate_to_tokens(prepared, content_budget)
            if not snippet:
                continue
            entry = label + snippet
            cost = count_tokens(entry) + 1
            # Token boundaries can shift where the label meets the excerpt.
            # Tighten by the exact overage so the hard ceiling remains exact.
            if used + cost > self.budget.expansion_tokens:
                snippet = truncate_to_tokens(
                    snippet, max(0, content_budget - (used + cost - self.budget.expansion_tokens))
                )
                entry = label + snippet
                cost = count_tokens(entry) + 1
            if not snippet or used + cost > self.budget.expansion_tokens:
                break
            texts.append(entry)
            chunk_ids.append(result.chunk.chunk_id)
            if is_consolidation:
                consolidation_kept += 1
            else:
                direct_kept += 1
            used += cost
            source_id = result.memory_source_id or result.chunk.turn_id
            source_tokens[source_id] += count_tokens(snippet)

        if not texts:
            return [], [], 0, len(expansions), {}

        return (
            texts,
            chunk_ids,
            used,
            len(expansions) - len(texts),
            dict(source_tokens),
        )

    def _budget_aware_order(
        self,
        expansions: list[RetrievalResult],
        *,
        query: str = "",
    ) -> list[RetrievalResult]:
        """Choose high-utility evidence under the hard token ceiling.

        Retrieval score divided by square-root token cost is a conservative
        length correction: it stops a few long, marginal candidates from
        hiding short precise evidence without collapsing into a pure
        score-per-token policy that over-favors tiny fragments. Selected rows
        return in original rank order for deterministic prompt rendering.
        """

        prefix_cost = count_tokens(EXPANSION_PREFIX)
        available = max(0, self.budget.expansion_tokens - prefix_cost)
        ranked: list[tuple[float, int, int, bool, RetrievalResult]] = []
        for index, result in enumerate(expansions):
            prepared = self._prepare_expansion_text(result.chunk.text, query)
            snippet = truncate_to_tokens(
                prepared, self.budget.max_expansion_tokens
            )
            if not snippet:
                continue
            # Two tokens safely approximate the rendered label and newline;
            # the exact pack below remains the authoritative hard cap.
            cost = count_tokens(snippet) + 2
            utility = max(0.0, float(result.score)) / math.sqrt(max(1, cost))
            ranked.append(
                (
                    utility,
                    index,
                    cost,
                    result.route == "live_consolidation",
                    result,
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected: list[tuple[int, RetrievalResult]] = []
        used = 0
        direct = 0
        consolidation = 0
        source_counts: dict[str, int] = defaultdict(int)
        remaining = list(ranked)
        while remaining:
            if self.budget.source_diverse_expansions:
                remaining.sort(
                    key=lambda item: (
                        -item[0]
                        / (
                            1
                            + source_counts[
                                self._result_source_id(item[4])
                            ]
                        ),
                        item[1],
                    )
                )
            _utility, index, cost, is_consolidation, result = remaining.pop(0)
            if is_consolidation:
                if consolidation >= self.budget.max_consolidation_expansions:
                    continue
            elif direct >= self.budget.max_expansions:
                continue
            if used + cost > available:
                continue
            selected.append((index, result))
            used += cost
            if is_consolidation:
                consolidation += 1
            else:
                direct += 1
            source_counts[self._result_source_id(result)] += 1
        selected.sort(key=lambda item: item[0])
        return [result for _index, result in selected]

    def _prepare_expansion_text(self, text: str, query: str) -> str:
        """Return a deterministic query-focused excerpt when enabled.

        The retriever still chooses and scores durable chunks. This method is
        deliberately only a packing transform: it keeps the best lexical
        sentence matches in their original order and stores no model state.
        If neither the query nor any sentence has a usable lexical match, the
        original text is retained so dense-only semantic hits are not erased.
        """

        stripped = text.strip()
        if not self.budget.query_aware_sentence_expansions or not query.strip():
            return stripped
        if self._sentence_segmenter is None:
            return stripped

        sentences = [
            segment.strip()
            for segment in self._sentence_segmenter.segment(stripped)
            if segment.strip()
        ]
        if len(sentences) <= self.budget.max_sentences_per_expansion:
            return stripped

        query_terms = set(tokenize(query))
        if not query_terms:
            return stripped

        scored: list[tuple[float, int]] = []
        for index, sentence in enumerate(sentences):
            sentence_terms = set(tokenize(sentence))
            overlap = query_terms.intersection(sentence_terms)
            if not overlap:
                continue
            # Exact numbers and long identifiers are unusually discriminative
            # in long-chat recall. Length normalization prevents a long
            # sentence from winning merely by containing more words.
            overlap_weight = sum(
                3.0 if term.isdigit() or len(term) >= 8 else 1.0
                for term in overlap
            )
            score = overlap_weight / math.sqrt(max(1, len(sentence_terms)))
            scored.append((score, index))

        if not scored:
            return stripped

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = sorted(
            index
            for _score, index in scored[
                : self.budget.max_sentences_per_expansion
            ]
        )
        return " ".join(sentences[index] for index in selected)

    def _information_gain_order(
        self,
        expansions: list[RetrievalResult],
        *,
        query: str,
    ) -> list[RetrievalResult]:
        """Filter low-yield evidence using estimated information per token.

        This is a deterministic rate-distortion proxy rather than a claim to
        know true answer mutual information. Candidate-set IDF estimates term
        surprise, normalized retrieval score estimates semantic relevance,
        and accepted evidence discounts repeated concepts, sources, and numeric
        facts. Crucially, this is a monotone filter over retrieval order: it can
        remove a low-yield item but cannot promote a weaker candidate over a
        required higher-ranked item.
        """

        if not expansions:
            return []

        prepared: list[dict[str, object]] = []
        document_frequency: dict[str, int] = defaultdict(int)
        raw_scores = [max(0.0, float(result.score)) for result in expansions]
        low_score = min(raw_scores, default=0.0)
        high_score = max(raw_scores, default=0.0)
        for index, result in enumerate(expansions):
            text = self._prepare_expansion_text(result.chunk.text, query)
            snippet = truncate_to_tokens(text, self.budget.max_expansion_tokens)
            terms = set(tokenize(snippet))
            for term in terms:
                document_frequency[term] += 1
            cost = count_tokens(snippet) + 2 if snippet else 0
            relative_score = (
                (raw_scores[index] - low_score) / (high_score - low_score)
                if high_score > low_score
                else 0.0
            )
            normalized_score = max(
                min(1.0, raw_scores[index]),
                0.5 * relative_score,
            )
            prepared.append(
                {
                    "index": index,
                    "result": result,
                    "terms": terms,
                    "cost": cost,
                    "score": normalized_score,
                    "source": self._result_source_id(result),
                    "consolidation": result.route == "live_consolidation",
                }
            )

        count = len(expansions)
        idf = {
            term: math.log2((count + 1.0) / (frequency + 1.0)) + 1.0
            for term, frequency in document_frequency.items()
        }
        query_terms = set(tokenize(query))
        # A set/sequence answer has a higher distortion cost than a singleton:
        # superficially repetitive excerpts may each carry a different required
        # member. Retain more evidence for these queries instead of teaching the
        # redundancy filter that "another concert" or "another change" is noise.
        multi_fact_markers = {
            "all",
            "each",
            "order",
            "ordered",
            "earliest",
            "latest",
            "sequence",
            "chronological",
            "compare",
            "differences",
            "between",
        }
        effective_threshold = self.budget.min_information_gain_per_token
        if query_terms.intersection(multi_fact_markers):
            effective_threshold *= 0.70
        query_weight = sum(idf.get(term, math.log2(count + 1.0)) for term in query_terms)
        selected: list[RetrievalResult] = []
        selected_terms: set[str] = set()
        selected_numbers: set[str] = set()
        selected_sources: set[str] = set()
        for item in prepared:
            cost = int(item["cost"])
            if cost <= 0:
                continue
            terms = set(item["terms"])
            total_information = sum(idf.get(term, 1.0) for term in terms)
            new_information = sum(
                idf.get(term, 1.0) for term in terms - selected_terms
            )
            lexical_relevance = (
                sum(
                    idf.get(term, 1.0)
                    for term in terms.intersection(query_terms)
                )
                / query_weight
                if query_weight > 0.0
                else 0.0
            )
            semantic_relevance = float(item["score"])
            relevance = max(lexical_relevance, semantic_relevance)
            concept_novelty = (
                new_information / total_information
                if total_information > 0.0
                else 0.0
            )
            source_novelty = float(str(item["source"]) not in selected_sources)
            numbers = {term for term in terms if term.isdigit()}
            temporal_novelty = (
                len(numbers - selected_numbers) / len(numbers)
                if numbers
                else 0.0
            )
            novelty = (
                0.65 * concept_novelty
                + 0.25 * source_novelty
                + 0.10 * temporal_novelty
            )
            marginal_information = relevance * (0.60 + 0.40 * novelty)
            gain_rate = marginal_information / max(1, cost)
            if gain_rate < effective_threshold:
                continue
            result = item["result"]
            if not isinstance(result, RetrievalResult):
                continue
            selected.append(result)
            selected_terms.update(terms)
            selected_numbers.update(term for term in terms if term.isdigit())
            selected_sources.add(str(item["source"]))

        return selected

    @staticmethod
    def _result_source_id(result: RetrievalResult) -> str:
        if result.memory_source_id:
            return result.memory_source_id
        if result.turn is not None:
            return str(result.turn.source_id or result.turn.turn_id)
        return result.chunk.turn_id

    def _heat_weighted_order(
        self, expansions: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Order a prefix by weighted-fair source exposure.

        Heat is source-level purchasing power, while chunk length is its cost.
        Sources with insufficient material naturally yield their unused share
        to the remaining queues. Nothing transformer-shaped is retained here.
        """

        source_heat: dict[str, float] = {}
        queues: dict[str, deque[RetrievalResult]] = defaultdict(deque)
        for result in expansions:
            source_id = result.memory_source_id or result.chunk.turn_id
            queues[source_id].append(result)
            if result.source_heat is not None:
                source_heat[source_id] = max(
                    source_heat.get(source_id, 0.0), float(result.source_heat)
                )
        if not source_heat or sum(source_heat.values()) <= 0.0:
            return expansions

        served: dict[str, int] = defaultdict(int)
        ordered: list[RetrievalResult] = []
        source_cap = max(
            1,
            math.ceil(
                self.budget.expansion_tokens
                * self.budget.max_source_expansion_fraction
            ),
        )
        while any(queues.values()):
            choices: list[tuple[float, float, str, RetrievalResult]] = []
            capped: list[tuple[float, float, str, RetrievalResult]] = []
            for source_id, queue in queues.items():
                if not queue:
                    continue
                result = queue[0]
                cost = max(
                    1,
                    min(result.chunk.token_count, self.budget.max_expansion_tokens),
                )
                weight = max(source_heat.get(source_id, 0.0), 1e-12)
                choice = (
                    (served[source_id] + cost) / weight,
                    -float(result.diffusion_heat or 0.0),
                    source_id,
                    result,
                )
                choices.append(choice)
                if served[source_id] == 0 or served[source_id] + cost <= source_cap:
                    capped.append(choice)
            pool = capped or choices
            _, _, source_id, result = min(pool)
            queues[source_id].popleft()
            served[source_id] += max(
                1,
                min(result.chunk.token_count, self.budget.max_expansion_tokens),
            )
            ordered.append(result)
        return ordered
