"""Public contracts and shared labels for context expansion packing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from memory_condense.domain.schemas import RetrievalResult


MEMORY_HEADER_PREFIX = "Relevant memory:"
EXPANSION_PREFIX = "Supporting excerpts:"


def _unique_chunk_ids(values: Sequence[str], label: str) -> tuple[str, ...]:
    rows = tuple(str(value).strip() for value in values)
    if any(not value for value in rows):
        raise ValueError(f"{label} must be non-empty")
    if len(set(rows)) != len(rows):
        raise ValueError(f"{label} must be unique")
    return rows


@dataclass(frozen=True, slots=True)
class AtomicExpansionGroup:
    """One all-or-fallback expansion group in its required render order."""

    ordered_chunk_ids: tuple[str, ...]
    original_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        ordered = _unique_chunk_ids(
            self.ordered_chunk_ids,
            "atomic group ordered chunk IDs",
        )
        originals = _unique_chunk_ids(
            self.original_chunk_ids,
            "atomic group original chunk IDs",
        )
        if not ordered or not originals:
            raise ValueError("an atomic expansion group requires ordered originals")
        if not set(originals).issubset(ordered):
            raise ValueError("atomic group originals must occur in its ordered IDs")
        object.__setattr__(self, "ordered_chunk_ids", ordered)
        object.__setattr__(self, "original_chunk_ids", originals)


@dataclass(frozen=True, slots=True)
class AtomicExpansionContract:
    """Immutable ID-only contract for additive, atomic context expansion."""

    original_chunk_ids: tuple[str, ...]
    companion_chunk_ids: tuple[str, ...]
    groups: tuple[AtomicExpansionGroup, ...]
    max_companion_chunks: int

    def __post_init__(self) -> None:
        originals = _unique_chunk_ids(
            self.original_chunk_ids,
            "atomic original chunk IDs",
        )
        companions = _unique_chunk_ids(
            self.companion_chunk_ids,
            "atomic companion chunk IDs",
        )
        groups = tuple(self.groups)
        if set(originals) & set(companions):
            raise ValueError("atomic original and companion IDs must be disjoint")
        if (
            isinstance(self.max_companion_chunks, bool)
            or not isinstance(self.max_companion_chunks, int)
            or self.max_companion_chunks < 0
        ):
            raise ValueError("max_companion_chunks must be a non-negative integer")
        if len(companions) > self.max_companion_chunks:
            raise ValueError("atomic companions exceed their count allowance")
        known_ids = set((*originals, *companions))
        grouped_ids: set[str] = set()
        grouped_companions: set[str] = set()
        for group in groups:
            ordered = set(group.ordered_chunk_ids)
            if not ordered.issubset(known_ids):
                raise ValueError("atomic group contains an unknown chunk ID")
            if not set(group.original_chunk_ids).issubset(originals):
                raise ValueError("atomic group contains a non-original anchor")
            if grouped_ids & ordered:
                raise ValueError("atomic expansion groups must be disjoint")
            grouped_ids.update(ordered)
            grouped_companions.update(ordered.intersection(companions))
        if grouped_companions != set(companions):
            raise ValueError("every atomic companion must occur in one group")
        object.__setattr__(self, "original_chunk_ids", originals)
        object.__setattr__(self, "companion_chunk_ids", companions)
        object.__setattr__(self, "groups", groups)


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
