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

from memory_condense._tokenizer import count_tokens, truncate_to_tokens
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
    max_expansion_tokens: int = 250
    # Opt-in: use diffused source heat as weighted-fair prompt exposure. The
    # default preserves the established retrieval ordering exactly.
    heat_weighted_expansions: bool = False
    max_source_expansion_fraction: float = 1.0

    def __post_init__(self) -> None:
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
        exp_texts, exp_tokens, exp_dropped, source_tokens = self._build_expansions(
            expansions
        )

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
        self, expansions: list[RetrievalResult]
    ) -> tuple[list[str], int, int, dict[str, int]]:
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
        texts: list[str] = []
        used = count_tokens(EXPANSION_PREFIX)
        source_tokens: dict[str, int] = defaultdict(int)

        for result in ranked:
            if len(texts) >= self.budget.max_expansions:
                break
            remaining = self.budget.expansion_tokens - used
            label = f"[{len(texts) + 1}] "
            # Reserve the label and the newline accounted for by this packer.
            content_budget = min(
                self.budget.max_expansion_tokens,
                remaining - count_tokens(label) - 1,
            )
            if content_budget <= 0:
                break
            snippet = truncate_to_tokens(
                result.chunk.text.strip(), content_budget
            )
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
            used += cost
            source_id = result.memory_source_id or result.chunk.turn_id
            source_tokens[source_id] += count_tokens(snippet)

        if not texts:
            return [], 0, len(expansions), {}

        return texts, used, len(expansions) - len(texts), dict(source_tokens)

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
