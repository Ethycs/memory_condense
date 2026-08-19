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

from typing import Any, Mapping

import pysbd

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.schemas import (
    MemoryItem,
    MemoryResult,
    PackedContext,
    RetrievalResult,
)
from memory_condense.search.packing.context_budget import ContextBudget
from memory_condense.search.packing.expansion_assembly import (
    _ExpansionPackingMixin,
)
from memory_condense.search.packing.packing_contracts import (
    EXPANSION_PREFIX,
    MEMORY_HEADER_PREFIX,
    ExpansionSelector,
)
from memory_condense.search.packing.source_provenance import (
    bind_source_metadata,
    is_source_metadata_text,
)

__all__ = [
    "ContextBudget",
    "ContextPacker",
    "ExpansionSelector",
    "is_source_metadata_text",
]


class ContextPacker(_ExpansionPackingMixin):
    """Packs memory, recent turns, and expansions into a budgeted message list."""

    def __init__(
        self,
        budget: ContextBudget | None = None,
        *,
        expansion_selector: ExpansionSelector | None = None,
    ) -> None:
        self.budget = budget or ContextBudget()
        self.expansion_selector = expansion_selector
        # Text-free, per-candidate diagnostics for the most recent expansion
        # packet.  This is intentionally ephemeral: it explains where a
        # bounded candidate was reordered or cut without entering PackedContext
        # or the durable memory store.
        self.last_expansion_trace: list[dict[str, Any]] = []
        self.last_closure_report: dict[str, Any] = {
            "applied": False,
            "closure_scope": "",
            "closure_global_recall_guaranteed": False,
        }
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
        source_metadata: dict[str, str] | None = None,
        active_partition_total: int | None = None,
        active_partition_inspected: int | None = None,
        active_partition_scan: Mapping[str, Any] | None = None,
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
        ) = self._build_expansions(
            expansions,
            query=user_text or "",
            source_metadata=source_metadata or {},
            active_partition_total=active_partition_total,
            active_partition_inspected=active_partition_inspected,
            active_partition_scan=active_partition_scan,
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

    def _bind_source_metadata(
        self,
        selected: list[RetrievalResult],
        *,
        candidate_pool: list[RetrievalResult] | None = None,
        source_metadata: dict[str, str] | None = None,
    ) -> tuple[dict[str, str], list[RetrievalResult]]:
        """Bind source timestamps while keeping the public metadata hook patchable."""

        return bind_source_metadata(
            selected,
            candidate_pool=candidate_pool,
            source_metadata=source_metadata,
            result_source_id=self._result_source_id,
            metadata_predicate=is_source_metadata_text,
        )
