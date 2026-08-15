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
    max_expansions: int = 3
    max_expansion_tokens: int = 250

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

        header, header_tokens, header_dropped = self._build_memory_header(memories)
        kept_turns, turn_tokens, turns_dropped = self._fit_recent_turns(recent_turns)
        exp_texts, exp_tokens, exp_dropped = self._build_expansions(expansions)

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
            expansions=exp_texts,
            recent_turns=kept_turns,
            token_counts=token_counts,
            dropped=dropped,
        )

    # -- section builders ---------------------------------------------------

    def _build_memory_header(
        self, memories: list[MemoryResult] | list[MemoryItem]
    ) -> tuple[str, int, int]:
        """Typed bullets, highest-ranked first, capped at the header budget."""
        items = [m.item if isinstance(m, MemoryResult) else m for m in memories]
        active = [i for i in items if i.status.value == "active"]

        if not active:
            return "", 0, 0

        lines: list[str] = []
        used = count_tokens(MEMORY_HEADER_PREFIX)
        dropped = 0

        for item in active:
            bullet = self._format_memory(item)
            cost = count_tokens(bullet) + 1  # +1 for the newline
            if used + cost > self.budget.memory_header_tokens:
                dropped += 1
                continue
            lines.append(bullet)
            used += cost

        if not lines:
            return "", 0, dropped

        header = MEMORY_HEADER_PREFIX + "\n" + "\n".join(lines)
        return header, count_tokens(header), dropped

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
    ) -> tuple[list[str], int, int]:
        """Verbatim excerpts, each capped, and capped again in aggregate."""
        texts: list[str] = []
        used = count_tokens(EXPANSION_PREFIX)
        considered = 0

        for result in expansions:
            if len(texts) >= self.budget.max_expansions:
                break
            considered += 1
            snippet = truncate_to_tokens(
                result.chunk.text.strip(), self.budget.max_expansion_tokens
            )
            entry = f"[{len(texts) + 1}] {snippet}"
            cost = count_tokens(entry) + 1
            if used + cost > self.budget.expansion_tokens:
                break
            texts.append(entry)
            used += cost

        if not texts:
            return [], 0, len(expansions)

        return texts, used, len(expansions) - len(texts)
