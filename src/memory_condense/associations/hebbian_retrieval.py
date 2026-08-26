"""Bounded retrieval through a live same-turn Hebbian access graph.

The graph's nodes are source-grounded conceptual chunks.  It persists only
chunk IDs and scalar co-access statistics; query text, token IDs, attention
maps, residuals, and K/V state never enter this layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from memory_condense.associations.association_store import AssociationStore
from memory_condense.associations.coaccess_graph import rank_discount
from memory_condense.associations.expansion_guards import (
    exceeds_prompt_budget,
    guard_expansion_request,
    require_artifact,
)
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.domain.schemas import RetrievalResult


Hydrate = Callable[..., RetrievalResult | None]
HebbianExpansionStatus = Literal[
    "replaced",
    "no_neighbor",
    "all_protected",
    "no_slot",
    "hydration_failed",
    "token_budget_rollback",
]
_HEBBIAN_EXPANSION_STATUSES = frozenset(
    {
        "replaced",
        "no_neighbor",
        "all_protected",
        "no_slot",
        "hydration_failed",
        "token_budget_rollback",
    }
)


def _nonempty_id(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{label} must be an exact non-empty string")
    return value


def _finite_float(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be an exact finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized


def _exact_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an exact integer")
    return value


@dataclass(frozen=True, slots=True)
class HebbianNeighborCandidateReceipt:
    """Text-free scalar provenance for one bounded graph candidate."""

    rank: int
    chunk_id: str
    score: float
    support: int
    anchor_chunk_id: str
    coaccess_count: int
    last_reinforced_turn: int

    def __post_init__(self) -> None:
        rank = _exact_int(self.rank, "neighbor rank")
        support = _exact_int(self.support, "neighbor support")
        coaccess_count = _exact_int(self.coaccess_count, "neighbor coaccess count")
        reinforced_turn = _exact_int(
            self.last_reinforced_turn,
            "neighbor last reinforced turn",
        )
        if rank < 1:
            raise ValueError("neighbor rank must be positive")
        if support < 1:
            raise ValueError("neighbor support must be positive")
        if coaccess_count < 1:
            raise ValueError("neighbor coaccess count must be positive")
        if reinforced_turn < 0:
            raise ValueError("neighbor last reinforced turn must be non-negative")
        score = _finite_float(self.score, "neighbor score")
        if score < 0.0:
            raise ValueError("neighbor score must be non-negative")
        object.__setattr__(self, "chunk_id", _nonempty_id(self.chunk_id, "chunk_id"))
        object.__setattr__(
            self,
            "anchor_chunk_id",
            _nonempty_id(self.anchor_chunk_id, "anchor_chunk_id"),
        )
        object.__setattr__(self, "score", score)

    def identity_payload(self) -> dict[str, object]:
        """Return the exact scalar row embedded in the expansion identity."""
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "support": self.support,
            "anchor_chunk_id": self.anchor_chunk_id,
            "coaccess_count": self.coaccess_count,
            "last_reinforced_turn": self.last_reinforced_turn,
        }


@dataclass(frozen=True, slots=True)
class HebbianExpansionReceipt(SealedIdentity):
    """Immutable text-free audit record for one Hebbian expansion attempt."""

    _SEAL_FIELD = "receipt_sha256"
    _SEAL_MISMATCH = "Hebbian expansion receipt does not match its contents"

    artifact_id: str
    now_turn: int
    result_cap: int
    hebbian_slots: int
    max_seed_concepts: int
    max_candidates: int
    half_life_turns: float
    min_score: float
    lexical_protection_threshold: float | None
    max_prompt_token_increase: int | None
    status: HebbianExpansionStatus
    base_chunk_ids: tuple[str, ...]
    base_activations: tuple[tuple[str, float], ...]
    neighbor_candidates: tuple[HebbianNeighborCandidateReceipt, ...]
    protected_chunk_ids: tuple[str, ...]
    replaceable_chunk_ids: tuple[str, ...]
    hydration_failed_chunk_ids: tuple[str, ...]
    proposed_removed_chunk_ids: tuple[str, ...]
    proposed_added_chunk_ids: tuple[str, ...]
    removed_chunk_ids: tuple[str, ...]
    added_chunk_ids: tuple[str, ...]
    final_chunk_ids: tuple[str, ...]
    base_chunk_token_total: int
    proposed_chunk_token_total: int
    final_chunk_token_total: int
    retained_request_token_state_bytes: int = 0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_id",
            _nonempty_id(self.artifact_id, "artifact_id"),
        )
        for name in (
            "now_turn",
            "result_cap",
            "hebbian_slots",
            "max_seed_concepts",
            "max_candidates",
            "base_chunk_token_total",
            "proposed_chunk_token_total",
            "final_chunk_token_total",
            "retained_request_token_state_bytes",
        ):
            object.__setattr__(self, name, _exact_int(getattr(self, name), name))
        if self.now_turn < 0 or self.result_cap < 0 or self.hebbian_slots < 0:
            raise ValueError("turn, result cap, and Hebbian slots must be non-negative")
        if self.max_seed_concepts < 1 or self.max_candidates < 1:
            raise ValueError("seed and candidate caps must be positive")
        if min(
            self.base_chunk_token_total,
            self.proposed_chunk_token_total,
            self.final_chunk_token_total,
        ) < 0:
            raise ValueError("chunk token totals must be non-negative")
        if self.retained_request_token_state_bytes != 0:
            raise ValueError("Hebbian expansion cannot retain request token state")
        if self.status not in _HEBBIAN_EXPANSION_STATUSES:
            raise ValueError("unsupported Hebbian expansion status")

        for name in (
            "base_chunk_ids",
            "protected_chunk_ids",
            "replaceable_chunk_ids",
            "hydration_failed_chunk_ids",
            "proposed_removed_chunk_ids",
            "proposed_added_chunk_ids",
            "removed_chunk_ids",
            "added_chunk_ids",
            "final_chunk_ids",
        ):
            if type(getattr(self, name)) is not tuple:
                raise TypeError(f"{name} must be an exact tuple")
            values = tuple(
                _nonempty_id(value, name) for value in getattr(self, name)
            )
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique chunk IDs")
            object.__setattr__(self, name, values)
        if len(self.base_chunk_ids) > self.result_cap:
            raise ValueError("base membership exceeds result_cap")

        if type(self.base_activations) is not tuple:
            raise TypeError("base_activations must be an exact tuple")
        activations: list[tuple[str, float]] = []
        for raw_item in self.base_activations:
            if type(raw_item) is not tuple or len(raw_item) != 2:
                raise ValueError("base activations must be (chunk_id, activation) pairs")
            chunk_id = _nonempty_id(raw_item[0], "activation chunk_id")
            activation = _finite_float(raw_item[1], "base activation")
            if not 0.0 <= activation <= 1.0:
                raise ValueError("base activations must lie in [0, 1]")
            activations.append((chunk_id, activation))
        if len({chunk_id for chunk_id, _ in activations}) != len(activations):
            raise ValueError("base activation chunk IDs must be unique")
        activation_ids = tuple(chunk_id for chunk_id, _ in activations)
        if activation_ids != self.base_chunk_ids[: len(activation_ids)]:
            raise ValueError("base activations must be an ordered prefix of anchors")
        if len(activation_ids) > self.max_seed_concepts:
            raise ValueError("base activations exceed max_seed_concepts")
        object.__setattr__(self, "base_activations", tuple(activations))
        expected_graph_activations = tuple(
            (chunk_id, rank_discount(rank))
            for rank, chunk_id in enumerate(
                self.base_chunk_ids[: self.max_seed_concepts],
                start=1,
            )
        )

        if type(self.neighbor_candidates) is not tuple:
            raise TypeError("neighbor_candidates must be an exact tuple")
        candidates = tuple(self.neighbor_candidates)
        if any(type(item) is not HebbianNeighborCandidateReceipt for item in candidates):
            raise TypeError(
                "neighbor candidates must be exact HebbianNeighborCandidateReceipt values"
            )
        if len(candidates) > self.max_candidates:
            raise ValueError("neighbor candidates exceed max_candidates")
        if tuple(item.rank for item in candidates) != tuple(
            range(1, len(candidates) + 1)
        ):
            raise ValueError("neighbor candidate ranks must be contiguous and ordered")
        candidate_ids = tuple(item.chunk_id for item in candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("neighbor candidate chunk IDs must be unique")
        if set(candidate_ids).intersection(self.base_chunk_ids):
            raise ValueError("neighbor candidates must exclude every base anchor")
        if any(item.anchor_chunk_id not in activation_ids for item in candidates):
            raise ValueError("neighbor candidates must name an active base anchor")
        object.__setattr__(self, "neighbor_candidates", candidates)

        object.__setattr__(
            self,
            "half_life_turns",
            _finite_float(self.half_life_turns, "half_life_turns"),
        )
        object.__setattr__(self, "min_score", _finite_float(self.min_score, "min_score"))
        if self.half_life_turns <= 0.0:
            raise ValueError("half_life_turns must be positive")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must lie in [0, 1]")
        if self.lexical_protection_threshold is not None:
            object.__setattr__(
                self,
                "lexical_protection_threshold",
                _finite_float(
                    self.lexical_protection_threshold,
                    "lexical_protection_threshold",
                ),
            )
            if not 0.0 <= self.lexical_protection_threshold <= 1.0:
                raise ValueError("lexical_protection_threshold must lie in [0, 1]")
        if self.max_prompt_token_increase is not None:
            increase = _exact_int(
                self.max_prompt_token_increase,
                "max_prompt_token_increase",
            )
            if increase < 0:
                raise ValueError("max_prompt_token_increase must be non-negative")
            object.__setattr__(self, "max_prompt_token_increase", increase)

        base_set = set(self.base_chunk_ids)
        protected_set = set(self.protected_chunk_ids)
        replaceable_set = set(self.replaceable_chunk_ids)
        if protected_set.intersection(replaceable_set):
            raise ValueError("protected and replaceable anchors must be disjoint")
        if not protected_set.union(replaceable_set).issubset(base_set):
            raise ValueError("protected/replaceable IDs must be base anchors")
        if self.protected_chunk_ids != tuple(
            chunk_id for chunk_id in self.base_chunk_ids if chunk_id in protected_set
        ):
            raise ValueError("protected anchors changed base order")
        if self.replaceable_chunk_ids != tuple(
            chunk_id
            for chunk_id in reversed(self.base_chunk_ids)
            if chunk_id in replaceable_set
        ):
            raise ValueError("replaceable anchors changed reverse base order")
        if self.status in {
            "all_protected",
            "hydration_failed",
            "token_budget_rollback",
            "replaced",
        } and protected_set.union(replaceable_set) != base_set:
            raise ValueError("receipt must classify every base anchor")

        candidate_set = set(candidate_ids)
        if not set(self.hydration_failed_chunk_ids).issubset(candidate_set):
            raise ValueError("hydration failures must be neighbor candidates")
        if not set(self.proposed_removed_chunk_ids).issubset(replaceable_set):
            raise ValueError("proposed removals must be replaceable base anchors")
        if not set(self.proposed_added_chunk_ids).issubset(candidate_set):
            raise ValueError("proposed additions must be neighbor candidates")
        if len(self.proposed_removed_chunk_ids) != len(
            self.proposed_added_chunk_ids
        ):
            raise ValueError("proposed removals and additions must be balanced")
        if len(self.proposed_added_chunk_ids) > self.hebbian_slots:
            raise ValueError("proposed additions exceed reserved Hebbian slots")
        expected_composed = tuple(
            chunk_id
            for chunk_id in self.base_chunk_ids
            if chunk_id not in set(self.proposed_removed_chunk_ids)
        ) + self.proposed_added_chunk_ids

        if self.status == "replaced":
            if not self.added_chunk_ids:
                raise ValueError("replaced receipts must add at least one chunk")
            if (
                self.removed_chunk_ids != self.proposed_removed_chunk_ids
                or self.added_chunk_ids != self.proposed_added_chunk_ids
                or self.final_chunk_ids != expected_composed
                or self.final_chunk_token_total != self.proposed_chunk_token_total
            ):
                raise ValueError("replaced receipt membership algebra changed")
        else:
            if self.removed_chunk_ids or self.added_chunk_ids:
                raise ValueError("no-op receipts cannot report effective changes")
            if (
                self.final_chunk_ids != self.base_chunk_ids
                or self.final_chunk_token_total != self.base_chunk_token_total
            ):
                raise ValueError("no-op receipt must preserve exact base membership")
            if self.status != "token_budget_rollback" and (
                self.proposed_removed_chunk_ids
                or self.proposed_added_chunk_ids
                or self.proposed_chunk_token_total != self.base_chunk_token_total
            ):
                raise ValueError("only token rollback may retain a proposal")

        if self.status == "no_neighbor" and (
            not self.base_chunk_ids
            or self.base_activations != expected_graph_activations
            or candidates
            or self.protected_chunk_ids
            or self.replaceable_chunk_ids
            or self.hydration_failed_chunk_ids
        ):
            raise ValueError("no_neighbor receipt changed its exact runtime shape")
        if self.status == "no_slot" and (
            self.base_activations
            or candidates
            or self.protected_chunk_ids
            or self.replaceable_chunk_ids
            or self.hydration_failed_chunk_ids
        ):
            raise ValueError(
                "no_slot receipt cannot report graph work or anchor classification"
            )
        if self.status == "all_protected" and (
            not candidates
            or self.protected_chunk_ids != self.base_chunk_ids
            or self.replaceable_chunk_ids
        ):
            raise ValueError("all_protected receipt classification changed")
        if self.status == "hydration_failed" and (
            self.base_activations != expected_graph_activations
            or not candidates
            or not self.replaceable_chunk_ids
            or self.hydration_failed_chunk_ids != candidate_ids
        ):
            raise ValueError(
                "hydration_failed receipt changed its exact runtime shape"
            )
        if self.status == "token_budget_rollback":
            if (
                not self.proposed_added_chunk_ids
                or self.max_prompt_token_increase is None
                or self.proposed_chunk_token_total
                <= self.base_chunk_token_total + self.max_prompt_token_increase
            ):
                raise ValueError("token rollback receipt has no rejected overspend")
        self._seal()


def retrieval_concept_activations(
    results: Sequence[RetrievalResult],
    *,
    max_concepts: int = 12,
) -> dict[str, float]:
    """Convert one ranked, exposed result set into bounded concept activity."""
    if max_concepts < 1:
        raise ValueError("max_concepts must be positive")
    activations: dict[str, float] = {}
    for rank, result in enumerate(results, start=1):
        chunk_id = result.chunk.chunk_id
        if chunk_id in activations:
            continue
        # Rank discount is stable across retrievers whose raw score scales are
        # not comparable. Every exposed concept remains active, but an early
        # result contributes more to the learned association.
        activations[chunk_id] = rank_discount(rank)
        if len(activations) >= max_concepts:
            break
    return activations


def expand_hebbian_results(
    anchors: Sequence[RetrievalResult],
    artifact_id: str,
    *,
    store: AssociationStore,
    hydrate: Hydrate,
    now_turn: int,
    k: int | None = None,
    hebbian_slots: int = 1,
    max_seed_concepts: int = 12,
    max_candidates: int = 32,
    half_life_turns: float = 200.0,
    min_score: float = 0.05,
    lexical_protection_threshold: float | None = None,
    max_prompt_token_increase: int | None = None,
) -> list[RetrievalResult]:
    """Replace reserved tail slots with learned co-access neighbors.

    Result count never exceeds ``k``. With ``max_prompt_token_increase=0`` the
    learned arm is also rolled back if its replacement chunks contain more
    tokens than the direct retrieval it would displace.
    """
    # Preserve the historical early return exactly: a non-positive cap ignores
    # every other knob and does not require the association artifact.
    result_cap = len(anchors) if k is None else int(k)
    if result_cap <= 0:
        return []
    results, _receipt = expand_hebbian_results_with_receipt(
        anchors,
        artifact_id,
        store=store,
        hydrate=hydrate,
        now_turn=int(now_turn),
        k=result_cap,
        hebbian_slots=int(hebbian_slots),
        max_seed_concepts=int(max_seed_concepts),
        max_candidates=int(max_candidates),
        half_life_turns=float(half_life_turns),
        min_score=float(min_score),
        lexical_protection_threshold=(
            None
            if lexical_protection_threshold is None
            else float(lexical_protection_threshold)
        ),
        max_prompt_token_increase=(
            None
            if max_prompt_token_increase is None
            else int(max_prompt_token_increase)
        ),
    )
    return results


def expand_hebbian_results_with_receipt(
    anchors: Sequence[RetrievalResult],
    artifact_id: str,
    *,
    store: AssociationStore,
    hydrate: Hydrate,
    now_turn: int,
    k: int | None = None,
    hebbian_slots: int = 1,
    max_seed_concepts: int = 12,
    max_candidates: int = 32,
    half_life_turns: float = 200.0,
    min_score: float = 0.05,
    lexical_protection_threshold: float | None = None,
    max_prompt_token_increase: int | None = None,
) -> tuple[list[RetrievalResult], HebbianExpansionReceipt]:
    """Expand through live co-access and return a sealed, text-free receipt."""
    now_turn = _exact_int(now_turn, "now_turn")
    if now_turn < 0:
        raise ValueError("now_turn must be non-negative")
    if k is not None:
        k = _exact_int(k, "k")
        if k < 0:
            raise ValueError("k must be non-negative")
    hebbian_slots = _exact_int(hebbian_slots, "hebbian_slots")
    max_seed_concepts = _exact_int(max_seed_concepts, "max_seed_concepts")
    max_candidates = _exact_int(max_candidates, "max_candidates")
    half_life_turns = _finite_float(half_life_turns, "half_life_turns")
    min_score = _finite_float(min_score, "min_score")
    if lexical_protection_threshold is not None:
        lexical_protection_threshold = _finite_float(
            lexical_protection_threshold,
            "lexical_protection_threshold",
        )
    if max_prompt_token_increase is not None:
        max_prompt_token_increase = _exact_int(
            max_prompt_token_increase,
            "max_prompt_token_increase",
        )
    if hebbian_slots < 0:
        raise ValueError("hebbian_slots must be non-negative")
    if max_seed_concepts < 1:
        raise ValueError("max_seed_concepts must be positive")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if half_life_turns <= 0.0:
        raise ValueError("half_life_turns must be positive")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must lie in [0, 1]")
    guards = guard_expansion_request(
        anchors,
        k=k,
        lexical_protection_threshold=lexical_protection_threshold,
        max_prompt_token_increase=max_prompt_token_increase,
    )
    result_cap = guards.result_cap
    bounded_anchors = guards.bounded_anchors
    base_chunk_ids = tuple(result.chunk.chunk_id for result in bounded_anchors)
    base_chunk_token_total = sum(
        result.chunk.token_count for result in bounded_anchors
    )

    activations: dict[str, float] = {}
    candidates: tuple[HebbianNeighborCandidateReceipt, ...] = ()
    protected_chunk_ids: tuple[str, ...] = ()
    replaceable_chunk_ids: tuple[str, ...] = ()
    hydration_failed_chunk_ids: tuple[str, ...] = ()
    proposed_removed_chunk_ids: tuple[str, ...] = ()
    proposed_added_chunk_ids: tuple[str, ...] = ()
    proposed_chunk_token_total = base_chunk_token_total

    def finish(
        status: HebbianExpansionStatus,
        final_results: Sequence[RetrievalResult],
        *,
        removed_chunk_ids: tuple[str, ...] = (),
        added_chunk_ids: tuple[str, ...] = (),
    ) -> tuple[list[RetrievalResult], HebbianExpansionReceipt]:
        final = list(final_results)
        receipt = HebbianExpansionReceipt(
            artifact_id=artifact_id,
            now_turn=now_turn,
            result_cap=result_cap,
            hebbian_slots=hebbian_slots,
            max_seed_concepts=max_seed_concepts,
            max_candidates=max_candidates,
            half_life_turns=half_life_turns,
            min_score=min_score,
            lexical_protection_threshold=lexical_protection_threshold,
            max_prompt_token_increase=max_prompt_token_increase,
            status=status,
            base_chunk_ids=base_chunk_ids,
            base_activations=tuple(activations.items()),
            neighbor_candidates=candidates,
            protected_chunk_ids=protected_chunk_ids,
            replaceable_chunk_ids=replaceable_chunk_ids,
            hydration_failed_chunk_ids=hydration_failed_chunk_ids,
            proposed_removed_chunk_ids=proposed_removed_chunk_ids,
            proposed_added_chunk_ids=proposed_added_chunk_ids,
            removed_chunk_ids=removed_chunk_ids,
            added_chunk_ids=added_chunk_ids,
            final_chunk_ids=tuple(result.chunk.chunk_id for result in final),
            base_chunk_token_total=base_chunk_token_total,
            proposed_chunk_token_total=proposed_chunk_token_total,
            final_chunk_token_total=sum(
                result.chunk.token_count for result in final
            ),
            retained_request_token_state_bytes=0,
        )
        return final, receipt

    # Unlike the compatibility wrapper, this API publishes an artifact-bound
    # receipt even for a no-op.  Prove that claimed namespace before sealing it.
    require_artifact(store, artifact_id)
    if result_cap <= 0:
        return finish("no_slot", ())
    if not bounded_anchors or hebbian_slots == 0:
        return finish("no_slot", bounded_anchors)
    activations = retrieval_concept_activations(
        bounded_anchors[:max_seed_concepts],
        max_concepts=max_seed_concepts,
    )
    neighbors = store.hebbian_neighbors(
        activations,
        artifact_id,
        top_k=max_candidates,
        # Lookup is deliberately seeded by only the first bounded concepts,
        # but every direct anchor is already present in the base membership.
        # Excluding the full base set prevents a lower-ranked anchor from
        # returning as its own "learned" replacement.
        exclude=base_chunk_ids,
        now_turn=now_turn,
        half_life_turns=half_life_turns,
        min_score=min_score,
    )
    candidates = tuple(
        HebbianNeighborCandidateReceipt(
            rank=rank,
            chunk_id=neighbor.chunk_id,
            score=neighbor.score,
            support=neighbor.support,
            anchor_chunk_id=neighbor.anchor_chunk_id,
            coaccess_count=neighbor.coaccess_count,
            last_reinforced_turn=neighbor.last_reinforced_turn,
        )
        for rank, neighbor in enumerate(neighbors, start=1)
    )
    if not neighbors:
        return finish("no_neighbor", bounded_anchors)

    replaceable: list[int] = []
    protected: list[int] = []
    for index in range(len(bounded_anchors) - 1, -1, -1):
        anchor = bounded_anchors[index]
        is_protected = (
            lexical_protection_threshold is not None
            and anchor.lexical_score is not None
            and anchor.lexical_score >= lexical_protection_threshold
        )
        if is_protected:
            protected.append(index)
        else:
            replaceable.append(index)
    protected_chunk_ids = tuple(
        bounded_anchors[index].chunk.chunk_id for index in reversed(protected)
    )
    replaceable_chunk_ids = tuple(
        bounded_anchors[index].chunk.chunk_id for index in replaceable
    )
    if not replaceable:
        return finish("all_protected", bounded_anchors)
    slot_count = min(hebbian_slots, len(replaceable), len(neighbors))
    if slot_count == 0:
        return finish("no_slot", bounded_anchors)

    learned: list[RetrievalResult] = []
    hydration_failures: list[str] = []
    for neighbor in neighbors:
        base = hydrate(neighbor.chunk_id, score=0.0)
        if (
            type(base) is not RetrievalResult
            or base.chunk.chunk_id != neighbor.chunk_id
            or base.chunk.chunk_id in base_chunk_ids
            or any(
                item.chunk.chunk_id == base.chunk.chunk_id for item in learned
            )
        ):
            hydration_failures.append(neighbor.chunk_id)
            continue
        learned.append(
            base.model_copy(
                update={
                    "score": neighbor.score,
                    "route": "hebbian_coaccess",
                    "association_score": neighbor.score,
                    "anchor_chunk_id": neighbor.anchor_chunk_id,
                    "association_hop": 1,
                    "edge_source_chunk_id": neighbor.anchor_chunk_id,
                    "association_path": (
                        neighbor.anchor_chunk_id,
                        neighbor.chunk_id,
                    ),
                    "association_support": neighbor.support,
                }
            )
        )
        if len(learned) >= slot_count:
            break
    hydration_failed_chunk_ids = tuple(hydration_failures)
    if not learned:
        return finish("hydration_failed", bounded_anchors)

    removed_indices = replaceable[: len(learned)]
    removed = set(removed_indices)
    proposed_removed_chunk_ids = tuple(
        bounded_anchors[index].chunk.chunk_id for index in removed_indices
    )
    proposed_added_chunk_ids = tuple(
        result.chunk.chunk_id for result in learned
    )
    composed = [
        result for index, result in enumerate(bounded_anchors) if index not in removed
    ] + learned
    proposed_chunk_token_total = sum(
        result.chunk.token_count for result in composed
    )
    if exceeds_prompt_budget(
        composed,
        # Denominator: this arm spends against all bounded_anchors while
        # associative windows to [:result_cap] (open author decision).
        direct_anchors=bounded_anchors,
        max_prompt_token_increase=max_prompt_token_increase,
    ):
        return finish("token_budget_rollback", bounded_anchors)
    return finish(
        "replaced",
        composed[:result_cap],
        removed_chunk_ids=proposed_removed_chunk_ids,
        added_chunk_ids=proposed_added_chunk_ids,
    )


__all__ = [
    "HebbianExpansionReceipt",
    "HebbianExpansionStatus",
    "HebbianNeighborCandidateReceipt",
    "expand_hebbian_results",
    "expand_hebbian_results_with_receipt",
    "retrieval_concept_activations",
]
