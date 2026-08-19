"""Immutable token-budget policy for context assembly."""

from __future__ import annotations

from dataclasses import dataclass


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
    # A hard coverage reservation is useful only when every admitted event
    # receives enough raw body content, after provenance-label overhead, to
    # convey a value.  When every requested event cannot meet this floor, the
    # packer deterministically reserves the largest feasible prefix and lets
    # the rest degrade to ordinary evidence.
    min_coverage_expansion_tokens: int = 24
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
    # Opt-in: treat standalone source/session timestamps as provenance rather
    # than independent evidence. The timestamp is bound to each selected
    # excerpt from that source, making temporal order recoverable without
    # spending a candidate slot on an anonymous date-only chunk.
    source_metadata_expansions: bool = False
    # Opt-in: use diffused source heat as weighted-fair prompt exposure. The
    # default preserves the established retrieval ordering exactly.
    heat_weighted_expansions: bool = False
    max_source_expansion_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.max_consolidation_expansions < 0:
            raise ValueError("max_consolidation_expansions must be non-negative")
        if self.min_coverage_expansion_tokens < 1:
            raise ValueError("min_coverage_expansion_tokens must be positive")
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
