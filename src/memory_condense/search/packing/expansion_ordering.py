"""Deterministic expansion transforms and budget-aware ordering policies."""

from __future__ import annotations

import math
from collections import defaultdict, deque

from memory_condense.domain._tokenizer import count_tokens, truncate_to_tokens
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.search.indexes.lexical import tokenize
from memory_condense.search.packing.packing_contracts import EXPANSION_PREFIX


class _ExpansionOrderingMixin:
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
