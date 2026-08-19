"""Public contracts and shared labels for context expansion packing."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from memory_condense.domain.schemas import RetrievalResult


MEMORY_HEADER_PREFIX = "Relevant memory:"
EXPANSION_PREFIX = "Supporting excerpts:"


class ExpansionSelector(Protocol):
    """Transient query-conditioned ordering over a bounded evidence subset."""

    last_report: Any
    allow_selected_scope_fixed_k_closure: bool

    def select(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        max_results: int | None = None,
        source_timestamps: Mapping[str, str] | None = None,
        active_partition_total: int | None = None,
        active_partition_inspected: int | None = None,
        active_partition_scan: Mapping[str, Any] | None = None,
    ) -> list[RetrievalResult]: ...
