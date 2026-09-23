"""Source-preserving composition of raw and projected evidence.

The raw and projection arms are selected independently before this module is
called.  Composition gives the first raw row from every exact opaque source a
protected prefix, follows it with the complete projection selection, and puts
the remaining raw rows last.  Exact chunk-ID deduplication happens only after
those three bands have been assembled.

The ranked-prefix packer may use the hybrid only when every raw source seed is
inside its accepted prefix.  If that proof fails, an exact pre-rendered raw
fallback is returned without reconstruction; when no fallback is supplied the
operation fails closed with a gate-bearing exception.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.search.packing.ranked_prefix_prompt import (
    NoFeasiblePrefixError,
    RankedPrefixPackingAudit,
    pack_ranked_prefix_prompt,
)


RawT = TypeVar("RawT")
ProjectedT = TypeVar("ProjectedT")
PromptT = TypeVar("PromptT")

POLICY_ID = "source-seed-projection-raw-ranked-prefix-v1"
FALLBACK_FORMAT = "memory-condense-exact-raw-prompt-fallback-v1"
AUDIT_FORMAT = "memory-condense-source-preserving-hybrid-audit-v1"
RESULT_FORMAT = "memory-condense-source-preserving-hybrid-pack-v1"


def _require_text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be non-empty exact text")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _text_tuple(
    value: object,
    label: str,
    *,
    unique: bool = False,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be an exact tuple")
    result = value
    if any(type(row) is not str or not row for row in result):
        raise ValueError(f"{label} must contain non-empty exact text")
    if unique and len(result) != len(set(result)):
        raise ValueError(f"{label} must be ordered and unique")
    return result


class HybridEvidencePlane(str, Enum):
    RAW = "raw"
    PROJECTION = "projection"


class HybridPriorityPhase(str, Enum):
    RAW_SOURCE_SEED = "raw_source_seed"
    PROJECTION = "projection"
    RAW_REMAINDER = "raw_remainder"


class HybridPackMode(str, Enum):
    HYBRID = "hybrid"
    RAW_FALLBACK = "raw_fallback"


@dataclass(frozen=True, slots=True)
class HybridEvidence(Generic[RawT, ProjectedT]):
    """One selected-arm item annotated only for composition and rendering."""

    chunk_id: str
    plane: HybridEvidencePlane
    priority_phase: HybridPriorityPhase
    input_ordinal: int
    value: RawT | ProjectedT

    def __post_init__(self) -> None:
        _require_text(self.chunk_id, "hybrid evidence chunk ID")
        if type(self.plane) is not HybridEvidencePlane:
            raise TypeError("hybrid evidence plane must be canonical")
        if type(self.priority_phase) is not HybridPriorityPhase:
            raise TypeError("hybrid evidence phase must be canonical")
        if (
            self.plane is HybridEvidencePlane.RAW
            and self.priority_phase is HybridPriorityPhase.PROJECTION
        ) or (
            self.plane is HybridEvidencePlane.PROJECTION
            and self.priority_phase is not HybridPriorityPhase.PROJECTION
        ):
            raise ValueError("hybrid evidence plane and phase disagree")
        if (
            isinstance(self.input_ordinal, bool)
            or type(self.input_ordinal) is not int
            or self.input_ordinal < 0
        ):
            raise ValueError("hybrid evidence input ordinal must be non-negative")

    def projection(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "input_ordinal": self.input_ordinal,
            "plane": self.plane.value,
            "priority_phase": self.priority_phase.value,
        }


@dataclass(frozen=True, slots=True)
class SourceSeedAudit:
    """The first raw occurrence retained for one exact source identity."""

    chunk_id: str
    raw_input_ordinal: int
    source_id_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.chunk_id, "source-seed chunk ID")
        _nonnegative_int(self.raw_input_ordinal, "source-seed raw input ordinal")
        _require_sha256(self.source_id_sha256, "source-seed source identity")

    def projection(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "raw_input_ordinal": self.raw_input_ordinal,
            "source_id_sha256": self.source_id_sha256,
        }


@dataclass(frozen=True, slots=True)
class ExactDuplicateAudit:
    """One post-selection occurrence excluded by exact chunk identity."""

    chunk_id: str
    retained: HybridEvidence[Any, Any]
    excluded: HybridEvidence[Any, Any]

    def __post_init__(self) -> None:
        _require_text(self.chunk_id, "duplicate chunk ID")
        if type(self.retained) is not HybridEvidence:
            raise TypeError("duplicate retained occurrence must be exact")
        if type(self.excluded) is not HybridEvidence:
            raise TypeError("duplicate excluded occurrence must be exact")
        if self.retained.chunk_id != self.chunk_id:
            raise ValueError("duplicate retained occurrence changed identity")
        if self.excluded.chunk_id != self.chunk_id:
            raise ValueError("duplicate excluded occurrence changed identity")

    def projection(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "excluded": self.excluded.projection(),
            "retained": self.retained.projection(),
        }


@dataclass(frozen=True, slots=True)
class SourceSeedGateAudit:
    """Explicit all-seeds-in-prefix proof and its fail-closed action."""

    required_seed_chunk_ids: tuple[str, ...]
    packed_seed_chunk_ids: tuple[str, ...]
    missing_seed_chunk_ids: tuple[str, ...]
    passed: bool
    action: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        required = _text_tuple(
            self.required_seed_chunk_ids,
            "required seed chunk IDs",
            unique=True,
        )
        packed = _text_tuple(
            self.packed_seed_chunk_ids,
            "packed seed chunk IDs",
            unique=True,
        )
        missing = _text_tuple(
            self.missing_seed_chunk_ids,
            "missing seed chunk IDs",
            unique=True,
        )
        if any(row not in required for row in packed + missing):
            raise ValueError("seed gate cites an unknown seed")
        if tuple(row for row in required if row in set(packed)) != packed:
            raise ValueError("packed seed order changed")
        if tuple(row for row in required if row not in set(packed)) != missing:
            raise ValueError("missing seed partition changed")
        if type(self.passed) is not bool or self.passed is not (not missing):
            raise ValueError("seed gate pass flag changed")
        allowed_actions = {"hybrid", "raw_fallback", "raise"}
        if self.action not in allowed_actions:
            raise ValueError("seed gate action changed")
        if not self.passed and self.action == "hybrid":
            raise ValueError("seed gate action disagrees with proof")

        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("source-seed gate receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "action": self.action,
            "missing_seed_chunk_ids": list(self.missing_seed_chunk_ids),
            "packed_seed_chunk_ids": list(self.packed_seed_chunk_ids),
            "passed": self.passed,
            "required_seed_chunk_ids": list(self.required_seed_chunk_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


class SourceSeedCoverageError(RuntimeError):
    """Raised when the hybrid loses a source seed and has no raw fallback."""

    def __init__(
        self,
        gate: SourceSeedGateAudit,
        audit: SourcePreservingHybridAudit | None = None,
    ) -> None:
        self.gate = gate
        self.audit = audit
        if gate.missing_seed_chunk_ids:
            missing = ", ".join(gate.missing_seed_chunk_ids)
            message = f"hybrid prefix omitted raw source seeds: {missing}"
        else:
            message = "hybrid prompt has no feasible ranked prefix"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ExactRawPromptFallback(Generic[PromptT]):
    """A sealed raw prompt that the composer may reuse but never rebuild."""

    raw_chunk_ids: tuple[str, ...]
    rendered_prompt: PromptT
    context_token_count: int
    prompt_token_count: int
    output_token_reserve: int
    prompt_workspace_token_count: int
    prompt_sha256: str
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _text_tuple(self.raw_chunk_ids, "raw fallback chunk IDs")
        _nonnegative_int(
            self.context_token_count,
            "raw fallback context token count",
        )
        prompt = _nonnegative_int(
            self.prompt_token_count,
            "raw fallback prompt token count",
        )
        reserve = _nonnegative_int(
            self.output_token_reserve,
            "raw fallback output token reserve",
        )
        workspace = _nonnegative_int(
            self.prompt_workspace_token_count,
            "raw fallback prompt workspace token count",
        )
        if workspace != prompt + reserve:
            raise ValueError("raw fallback workspace accounting changed")
        _require_sha256(self.prompt_sha256, "raw fallback prompt")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("raw fallback receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "context_token_count": self.context_token_count,
            "format": FALLBACK_FORMAT,
            "output_token_reserve": self.output_token_reserve,
            "prompt_sha256": self.prompt_sha256,
            "prompt_token_count": self.prompt_token_count,
            "prompt_workspace_token_count": self.prompt_workspace_token_count,
            "raw_chunk_ids": list(self.raw_chunk_ids),
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SourcePreservingHybridAudit:
    """Deterministic selection, deduplication, packing, and gate receipt."""

    raw_selected_chunk_ids: tuple[str, ...]
    projected_selected_chunk_ids: tuple[str, ...]
    source_seeds: tuple[SourceSeedAudit, ...]
    ranked_before_dedup: tuple[HybridEvidence[Any, Any], ...]
    ranked_after_dedup: tuple[HybridEvidence[Any, Any], ...]
    exact_duplicates: tuple[ExactDuplicateAudit, ...]
    packing_status: str
    packing_audit: RankedPrefixPackingAudit | None
    seed_gate: SourceSeedGateAudit
    hybrid_prompt_sha256: str | None
    raw_fallback_receipt_sha256: str | None
    raw_fallback_prompt_sha256: str | None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _text_tuple(self.raw_selected_chunk_ids, "raw selected chunk IDs")
        _text_tuple(
            self.projected_selected_chunk_ids,
            "projected selected chunk IDs",
        )
        if type(self.source_seeds) is not tuple or any(
            type(row) is not SourceSeedAudit for row in self.source_seeds
        ):
            raise TypeError("source seeds must be an exact tuple")
        for name in ("ranked_before_dedup", "ranked_after_dedup"):
            rows = getattr(self, name)
            if type(rows) is not tuple or any(
                type(row) is not HybridEvidence for row in rows
            ):
                raise TypeError(f"{name} must be an exact evidence tuple")
        if type(self.exact_duplicates) is not tuple or any(
            type(row) is not ExactDuplicateAudit for row in self.exact_duplicates
        ):
            raise TypeError("exact duplicates must be an exact tuple")
        if len(self.ranked_before_dedup) != (
            len(self.ranked_after_dedup) + len(self.exact_duplicates)
        ):
            raise ValueError("post-selection dedup accounting changed")
        after_ids = tuple(row.chunk_id for row in self.ranked_after_dedup)
        if len(after_ids) != len(set(after_ids)):
            raise ValueError("ranked hybrid still repeats a chunk ID")
        expected_after: list[HybridEvidence[Any, Any]] = []
        expected_duplicates: list[ExactDuplicateAudit] = []
        retained_by_id: dict[str, HybridEvidence[Any, Any]] = {}
        for row in self.ranked_before_dedup:
            retained = retained_by_id.get(row.chunk_id)
            if retained is None:
                retained_by_id[row.chunk_id] = row
                expected_after.append(row)
            else:
                expected_duplicates.append(
                    ExactDuplicateAudit(
                        chunk_id=row.chunk_id,
                        retained=retained,
                        excluded=row,
                    )
                )
        if tuple(expected_after) != self.ranked_after_dedup:
            raise ValueError("post-selection retained order changed")
        if tuple(expected_duplicates) != self.exact_duplicates:
            raise ValueError("post-selection duplicate audit changed")

        phase_order = {
            HybridPriorityPhase.RAW_SOURCE_SEED: 0,
            HybridPriorityPhase.PROJECTION: 1,
            HybridPriorityPhase.RAW_REMAINDER: 2,
        }
        phases = tuple(
            phase_order[row.priority_phase] for row in self.ranked_before_dedup
        )
        if any(left > right for left, right in zip(phases, phases[1:])):
            raise ValueError("hybrid priority bands changed")
        raw_rows = sorted(
            (
                row
                for row in self.ranked_before_dedup
                if row.plane is HybridEvidencePlane.RAW
            ),
            key=lambda row: row.input_ordinal,
        )
        projected_rows = tuple(
            row
            for row in self.ranked_before_dedup
            if row.plane is HybridEvidencePlane.PROJECTION
        )
        if tuple(row.input_ordinal for row in raw_rows) != tuple(
            range(len(raw_rows))
        ) or tuple(row.chunk_id for row in raw_rows) != self.raw_selected_chunk_ids:
            raise ValueError("raw selected population changed")
        if tuple(row.input_ordinal for row in projected_rows) != tuple(
            range(len(projected_rows))
        ) or tuple(
            row.chunk_id for row in projected_rows
        ) != self.projected_selected_chunk_ids:
            raise ValueError("projected selected population changed")
        seed_rows = tuple(
            row
            for row in self.ranked_before_dedup
            if row.priority_phase is HybridPriorityPhase.RAW_SOURCE_SEED
        )
        if tuple(
            (row.chunk_id, row.input_ordinal) for row in seed_rows
        ) != tuple(
            (row.chunk_id, row.raw_input_ordinal) for row in self.source_seeds
        ):
            raise ValueError("source-seed ordering changed")

        if self.packing_status not in {"packed", "no_feasible_prefix"}:
            raise ValueError("hybrid packing status changed")
        if self.packing_status == "packed":
            if type(self.packing_audit) is not RankedPrefixPackingAudit:
                raise TypeError("packed hybrid requires an exact packing audit")
            if self.packing_audit.candidate_count != len(
                self.ranked_after_dedup
            ):
                raise ValueError("packing audit candidate count changed")
            _require_sha256(self.hybrid_prompt_sha256, "hybrid prompt")
        elif self.packing_audit is not None or self.hybrid_prompt_sha256 is not None:
            raise ValueError("infeasible hybrid cannot claim a packed prompt")
        if type(self.seed_gate) is not SourceSeedGateAudit:
            raise TypeError("seed gate must be exact")
        if self.seed_gate.required_seed_chunk_ids != tuple(
            row.chunk_id for row in self.source_seeds
        ):
            raise ValueError("seed gate disagrees with source seeds")
        if self.raw_fallback_receipt_sha256 is not None:
            _require_sha256(
                self.raw_fallback_receipt_sha256,
                "raw fallback receipt",
            )
        if self.raw_fallback_prompt_sha256 is not None:
            _require_sha256(
                self.raw_fallback_prompt_sha256,
                "raw fallback prompt",
            )
        if (self.raw_fallback_receipt_sha256 is None) is not (
            self.raw_fallback_prompt_sha256 is None
        ):
            raise ValueError("raw fallback audit fields disagree")
        if self.packing_status == "no_feasible_prefix" and (
            self.seed_gate.action == "hybrid"
        ):
            raise ValueError("infeasible hybrid cannot pass to the provider")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("source-preserving hybrid audit receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "audit_format": AUDIT_FORMAT,
            "exact_duplicates": [row.projection() for row in self.exact_duplicates],
            "hybrid_prompt_sha256": self.hybrid_prompt_sha256,
            "packing": (
                None
                if self.packing_audit is None
                else self.packing_audit.projection()
            ),
            "packing_status": self.packing_status,
            "policy_id": POLICY_ID,
            "projected_selected_chunk_ids": list(
                self.projected_selected_chunk_ids
            ),
            "ranked_after_dedup": [
                row.projection() for row in self.ranked_after_dedup
            ],
            "ranked_before_dedup": [
                row.projection() for row in self.ranked_before_dedup
            ],
            "raw_fallback_receipt_sha256": self.raw_fallback_receipt_sha256,
            "raw_fallback_prompt_sha256": self.raw_fallback_prompt_sha256,
            "raw_selected_chunk_ids": list(self.raw_selected_chunk_ids),
            "seed_gate": self.seed_gate.projection(),
            "source_seeds": [row.projection() for row in self.source_seeds],
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


@dataclass(frozen=True, slots=True)
class SourcePreservingHybridPack(Generic[RawT, ProjectedT, PromptT]):
    """The effective provider prompt and its hybrid/fallback decision."""

    mode: HybridPackMode
    packed_items: tuple[HybridEvidence[RawT, ProjectedT], ...]
    dropped_items: tuple[HybridEvidence[RawT, ProjectedT], ...]
    rendered_prompt: PromptT
    context_token_count: int
    prompt_token_count: int
    output_token_reserve: int
    prompt_workspace_token_count: int
    effective_prompt_sha256: str
    raw_fallback_reused: bool
    audit: SourcePreservingHybridAudit
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.mode) is not HybridPackMode:
            raise TypeError("hybrid pack mode must be canonical")
        for name in ("packed_items", "dropped_items"):
            rows = getattr(self, name)
            if type(rows) is not tuple or any(
                type(row) is not HybridEvidence for row in rows
            ):
                raise TypeError(f"{name} must be an exact evidence tuple")
        _nonnegative_int(
            self.context_token_count,
            "effective context token count",
        )
        prompt = _nonnegative_int(
            self.prompt_token_count,
            "effective prompt token count",
        )
        reserve = _nonnegative_int(
            self.output_token_reserve,
            "effective output token reserve",
        )
        workspace = _nonnegative_int(
            self.prompt_workspace_token_count,
            "effective prompt workspace token count",
        )
        if workspace != prompt + reserve:
            raise ValueError("effective workspace accounting changed")
        _require_sha256(self.effective_prompt_sha256, "effective prompt")
        if type(self.raw_fallback_reused) is not bool:
            raise TypeError("raw fallback reused flag must be exact")
        if type(self.audit) is not SourcePreservingHybridAudit:
            raise TypeError("source-preserving hybrid audit must be exact")
        if self.mode is HybridPackMode.HYBRID:
            if (
                self.raw_fallback_reused
                or self.audit.seed_gate.action != "hybrid"
                or self.audit.packing_status != "packed"
                or self.audit.packing_audit is None
                or self.audit.hybrid_prompt_sha256
                != self.effective_prompt_sha256
            ):
                raise ValueError("hybrid result disagrees with seed gate")
            packed_count = self.audit.packing_audit.packed_count
            if self.packed_items != self.audit.ranked_after_dedup[:packed_count]:
                raise ValueError("hybrid packed items disagree with audit")
            if self.dropped_items != self.audit.ranked_after_dedup[packed_count:]:
                raise ValueError("hybrid dropped items disagree with audit")
            accepted_probes = tuple(
                row
                for row in self.audit.packing_audit.probes
                if row.prefix_count == packed_count
            )
            if len(accepted_probes) != 1:
                raise ValueError("hybrid accepted-prefix proof changed")
            accepted = accepted_probes[0]
            if (
                self.context_token_count != accepted.context_token_count
                or self.prompt_token_count != accepted.prompt_token_count
                or self.prompt_workspace_token_count
                != accepted.prompt_workspace_token_count
                or self.output_token_reserve
                != self.audit.packing_audit.output_token_reserve
            ):
                raise ValueError("hybrid token accounting disagrees with audit")
        elif (
            not self.raw_fallback_reused
            or self.audit.seed_gate.action != "raw_fallback"
            or self.audit.raw_fallback_receipt_sha256 is None
            or self.audit.raw_fallback_prompt_sha256
            != self.effective_prompt_sha256
        ):
            raise ValueError("raw fallback result disagrees with seed gate")
        else:
            if tuple(row.chunk_id for row in self.packed_items) != (
                self.audit.raw_selected_chunk_ids
            ) or any(
                row.plane is not HybridEvidencePlane.RAW
                for row in self.packed_items
            ):
                raise ValueError("raw fallback items disagree with audit")
            if tuple(row.input_ordinal for row in self.packed_items) != tuple(
                range(len(self.packed_items))
            ):
                raise ValueError("raw fallback item order changed")
            if tuple(row.chunk_id for row in self.dropped_items) != (
                self.audit.projected_selected_chunk_ids
            ) or any(
                row.plane is not HybridEvidencePlane.PROJECTION
                for row in self.dropped_items
            ):
                raise ValueError("raw fallback projection omissions changed")
        expected = identity_sha256(self.projection(include_receipt=False))
        if self.receipt_sha256 and self.receipt_sha256 != expected:
            raise ValueError("source-preserving hybrid result receipt changed")
        object.__setattr__(self, "receipt_sha256", expected)

    def projection(self, *, include_receipt: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "audit_receipt_sha256": self.audit.receipt_sha256,
            "context_token_count": self.context_token_count,
            "dropped_items": [row.projection() for row in self.dropped_items],
            "effective_prompt_sha256": self.effective_prompt_sha256,
            "format": RESULT_FORMAT,
            "mode": self.mode.value,
            "output_token_reserve": self.output_token_reserve,
            "packed_items": [row.projection() for row in self.packed_items],
            "prompt_token_count": self.prompt_token_count,
            "prompt_workspace_token_count": self.prompt_workspace_token_count,
            "raw_fallback_reused": self.raw_fallback_reused,
        }
        if include_receipt:
            value["receipt_sha256"] = self.receipt_sha256
        return value


def _validate_fallback(
    fallback: ExactRawPromptFallback[PromptT],
    *,
    raw_chunk_ids: tuple[str, ...],
    raw_wrappers: tuple[HybridEvidence[RawT, ProjectedT], ...],
    count_context_tokens: Callable[
        [Sequence[HybridEvidence[RawT, ProjectedT]]], int
    ],
    count_prompt_tokens: Callable[[PromptT], int],
    prompt_sha256: Callable[[PromptT], str],
    max_context_tokens: int,
    max_prompt_tokens: int,
    output_token_reserve: int,
) -> None:
    if fallback.raw_chunk_ids != raw_chunk_ids:
        raise ValueError("raw fallback evidence order changed")
    if fallback.output_token_reserve != output_token_reserve:
        raise ValueError("raw fallback output token reserve changed")
    if fallback.context_token_count > max_context_tokens:
        raise ValueError("raw fallback exceeds the context token cap")
    if fallback.prompt_workspace_token_count > max_prompt_tokens:
        raise ValueError("raw fallback exceeds the prompt workspace cap")
    observed_context = _nonnegative_int(
        count_context_tokens(raw_wrappers),
        "count_context_tokens result",
    )
    if observed_context != fallback.context_token_count:
        raise ValueError("raw fallback context token count changed")
    observed_prompt = _nonnegative_int(
        count_prompt_tokens(fallback.rendered_prompt),
        "count_prompt_tokens result",
    )
    if observed_prompt != fallback.prompt_token_count:
        raise ValueError("raw fallback prompt token count changed")
    observed_sha = _require_sha256(
        prompt_sha256(fallback.rendered_prompt),
        "prompt_sha256 result",
    )
    if observed_sha != fallback.prompt_sha256:
        raise ValueError("raw fallback prompt bytes changed")


def pack_source_preserving_hybrid(
    raw_items: Sequence[RawT],
    projected_items: Sequence[ProjectedT],
    *,
    raw_chunk_id: Callable[[RawT], str],
    raw_source_id: Callable[[RawT], str],
    projected_chunk_id: Callable[[ProjectedT], str],
    count_context_tokens: Callable[
        [Sequence[HybridEvidence[RawT, ProjectedT]]], int
    ],
    render_prompt: Callable[
        [Sequence[HybridEvidence[RawT, ProjectedT]]], PromptT
    ],
    count_prompt_tokens: Callable[[PromptT], int],
    prompt_sha256: Callable[[PromptT], str],
    max_context_tokens: int,
    max_prompt_tokens: int,
    output_token_reserve: int = 0,
    raw_fallback: ExactRawPromptFallback[PromptT] | None = None,
) -> SourcePreservingHybridPack[RawT, ProjectedT, PromptT]:
    """Pack a projection-first hybrid without sacrificing raw-source reach.

    ``raw_items`` and ``projected_items`` are complete, independently selected
    arm outputs.  The function never selects within either arm.  Source values
    are used only as exact dictionary keys and are represented in the audit by
    SHA-256; they are never parsed, normalized, or ranked.

    A supplied ``raw_fallback`` is integrity-checked eagerly.  If the hybrid
    prefix loses any first-per-source raw seed, its exact ``rendered_prompt``
    object and sealed accounting are returned.  With no fallback, the same
    failed proof raises :class:`SourceSeedCoverageError`.  The gate proves one
    retained raw representative per source; it deliberately does not promise
    that every lower-ranked raw remainder survives the shared prefix cap.
    """

    for callback, label in (
        (raw_chunk_id, "raw_chunk_id"),
        (raw_source_id, "raw_source_id"),
        (projected_chunk_id, "projected_chunk_id"),
        (count_context_tokens, "count_context_tokens"),
        (render_prompt, "render_prompt"),
        (count_prompt_tokens, "count_prompt_tokens"),
        (prompt_sha256, "prompt_sha256"),
    ):
        if not callable(callback):
            raise TypeError(f"{label} must be callable")

    context_cap = _nonnegative_int(max_context_tokens, "max_context_tokens")
    prompt_cap = _nonnegative_int(max_prompt_tokens, "max_prompt_tokens")
    reserve = _nonnegative_int(output_token_reserve, "output_token_reserve")

    raws = tuple(raw_items)
    projections = tuple(projected_items)
    raw_ids: list[str] = []
    source_by_chunk: dict[str, str] = {}
    seed_ordinals: list[int] = []
    seen_sources: set[str] = set()
    source_seed_audits: list[SourceSeedAudit] = []
    for ordinal, item in enumerate(raws):
        chunk_id = _require_text(raw_chunk_id(item), "raw chunk ID")
        source_id = _require_text(raw_source_id(item), "raw source ID")
        known_source = source_by_chunk.setdefault(chunk_id, source_id)
        if known_source != source_id:
            raise ValueError(
                f"raw chunk ID {chunk_id!r} maps to conflicting exact sources"
            )
        raw_ids.append(chunk_id)
        if source_id not in seen_sources:
            seen_sources.add(source_id)
            seed_ordinals.append(ordinal)
            source_seed_audits.append(
                SourceSeedAudit(
                    chunk_id=chunk_id,
                    raw_input_ordinal=ordinal,
                    source_id_sha256=quote_sha256(source_id),
                )
            )

    seed_ordinal_set = set(seed_ordinals)
    raw_wrappers = tuple(
        HybridEvidence(
            chunk_id=raw_ids[ordinal],
            plane=HybridEvidencePlane.RAW,
            priority_phase=(
                HybridPriorityPhase.RAW_SOURCE_SEED
                if ordinal in seed_ordinal_set
                else HybridPriorityPhase.RAW_REMAINDER
            ),
            input_ordinal=ordinal,
            value=item,
        )
        for ordinal, item in enumerate(raws)
    )
    projected_wrappers = tuple(
        HybridEvidence(
            chunk_id=_require_text(
                projected_chunk_id(item),
                "projected chunk ID",
            ),
            plane=HybridEvidencePlane.PROJECTION,
            priority_phase=HybridPriorityPhase.PROJECTION,
            input_ordinal=ordinal,
            value=item,
        )
        for ordinal, item in enumerate(projections)
    )
    if raw_fallback is not None:
        if type(raw_fallback) is not ExactRawPromptFallback:
            raise TypeError("raw_fallback must be an exact fallback contract")
        _validate_fallback(
            raw_fallback,
            raw_chunk_ids=tuple(raw_ids),
            raw_wrappers=raw_wrappers,
            count_context_tokens=count_context_tokens,
            count_prompt_tokens=count_prompt_tokens,
            prompt_sha256=prompt_sha256,
            max_context_tokens=context_cap,
            max_prompt_tokens=prompt_cap,
            output_token_reserve=reserve,
        )

    ranked_before_dedup = (
        tuple(raw_wrappers[ordinal] for ordinal in seed_ordinals)
        + projected_wrappers
        + tuple(
            row
            for ordinal, row in enumerate(raw_wrappers)
            if ordinal not in seed_ordinal_set
        )
    )
    retained_by_id: dict[str, HybridEvidence[RawT, ProjectedT]] = {}
    ranked_after_dedup: list[HybridEvidence[RawT, ProjectedT]] = []
    duplicate_audits: list[ExactDuplicateAudit] = []
    for row in ranked_before_dedup:
        retained = retained_by_id.get(row.chunk_id)
        if retained is not None:
            duplicate_audits.append(
                ExactDuplicateAudit(
                    chunk_id=row.chunk_id,
                    retained=retained,
                    excluded=row,
                )
            )
            continue
        retained_by_id[row.chunk_id] = row
        ranked_after_dedup.append(row)

    required_seed_ids = tuple(row.chunk_id for row in source_seed_audits)
    try:
        hybrid_pack = pack_ranked_prefix_prompt(
            tuple(ranked_after_dedup),
            count_context_tokens=count_context_tokens,
            render_prompt=render_prompt,
            count_prompt_tokens=count_prompt_tokens,
            max_context_tokens=context_cap,
            max_prompt_tokens=prompt_cap,
            output_token_reserve=reserve,
        )
    except NoFeasiblePrefixError:
        hybrid_pack = None
        packing_status = "no_feasible_prefix"
        hybrid_prompt_sha = None
        packed_seed_ids: tuple[str, ...] = ()
    else:
        packing_status = "packed"
        hybrid_prompt_sha = _require_sha256(
            prompt_sha256(hybrid_pack.rendered_prompt),
            "prompt_sha256 result",
        )
        packed_seed_ids = tuple(
            row.chunk_id
            for row in hybrid_pack.packed_items
            if row.priority_phase is HybridPriorityPhase.RAW_SOURCE_SEED
        )
    packed_seed_set = set(packed_seed_ids)
    missing_seed_ids = tuple(
        row for row in required_seed_ids if row not in packed_seed_set
    )
    hybrid_usable = packing_status == "packed" and not missing_seed_ids
    action = (
        "hybrid"
        if hybrid_usable
        else "raw_fallback" if raw_fallback is not None else "raise"
    )
    gate = SourceSeedGateAudit(
        required_seed_chunk_ids=required_seed_ids,
        packed_seed_chunk_ids=packed_seed_ids,
        missing_seed_chunk_ids=missing_seed_ids,
        passed=not missing_seed_ids,
        action=action,
    )
    audit = SourcePreservingHybridAudit(
        raw_selected_chunk_ids=tuple(raw_ids),
        projected_selected_chunk_ids=tuple(
            row.chunk_id for row in projected_wrappers
        ),
        source_seeds=tuple(source_seed_audits),
        ranked_before_dedup=ranked_before_dedup,
        ranked_after_dedup=tuple(ranked_after_dedup),
        exact_duplicates=tuple(duplicate_audits),
        packing_status=packing_status,
        packing_audit=None if hybrid_pack is None else hybrid_pack.audit,
        seed_gate=gate,
        hybrid_prompt_sha256=hybrid_prompt_sha,
        raw_fallback_receipt_sha256=(
            None if raw_fallback is None else raw_fallback.receipt_sha256
        ),
        raw_fallback_prompt_sha256=(
            None if raw_fallback is None else raw_fallback.prompt_sha256
        ),
    )

    if not hybrid_usable:
        if raw_fallback is None:
            raise SourceSeedCoverageError(gate, audit)
        return SourcePreservingHybridPack(
            mode=HybridPackMode.RAW_FALLBACK,
            packed_items=raw_wrappers,
            dropped_items=projected_wrappers,
            rendered_prompt=raw_fallback.rendered_prompt,
            context_token_count=raw_fallback.context_token_count,
            prompt_token_count=raw_fallback.prompt_token_count,
            output_token_reserve=raw_fallback.output_token_reserve,
            prompt_workspace_token_count=(
                raw_fallback.prompt_workspace_token_count
            ),
            effective_prompt_sha256=raw_fallback.prompt_sha256,
            raw_fallback_reused=True,
            audit=audit,
        )

    if hybrid_pack is None:  # pragma: no cover - narrowed by hybrid_usable
        raise RuntimeError("usable hybrid has no ranked-prefix pack")
    return SourcePreservingHybridPack(
        mode=HybridPackMode.HYBRID,
        packed_items=hybrid_pack.packed_items,
        dropped_items=hybrid_pack.dropped_items,
        rendered_prompt=hybrid_pack.rendered_prompt,
        context_token_count=hybrid_pack.context_token_count,
        prompt_token_count=hybrid_pack.prompt_token_count,
        output_token_reserve=reserve,
        prompt_workspace_token_count=hybrid_pack.prompt_workspace_token_count,
        effective_prompt_sha256=hybrid_prompt_sha,
        raw_fallback_reused=False,
        audit=audit,
    )


__all__ = [
    "AUDIT_FORMAT",
    "FALLBACK_FORMAT",
    "POLICY_ID",
    "RESULT_FORMAT",
    "ExactDuplicateAudit",
    "ExactRawPromptFallback",
    "HybridEvidence",
    "HybridEvidencePlane",
    "HybridPackMode",
    "HybridPriorityPhase",
    "SourcePreservingHybridAudit",
    "SourcePreservingHybridPack",
    "SourceSeedAudit",
    "SourceSeedCoverageError",
    "SourceSeedGateAudit",
    "pack_source_preserving_hybrid",
]
