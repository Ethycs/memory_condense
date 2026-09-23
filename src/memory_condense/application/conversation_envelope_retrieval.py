"""Explicit hydration of bounded conversation-envelope retrieval plans."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.domain.sealed import SealedIdentity, reflect_payload
from memory_condense.persistence.conversation_envelope_expansion import (
    ConversationEnvelopeExpansionPlan,
)
from memory_condense.search.packing.packing_contracts import (
    AtomicExpansionContract,
    AtomicExpansionGroup,
)


CONVERSATION_ENVELOPE_RETRIEVAL_FORMAT = (
    "memory-condense-conversation-envelope-retrieval-v2"
)
CONVERSATION_ENVELOPE_MEMBER_ROUTE = "conversation_envelope_member"


def _result_rows_sha256(results: Sequence[RetrievalResult]) -> str:
    def receipt_row(result: RetrievalResult) -> dict[str, Any]:
        turn_payload: dict[str, Any] | None = None
        if result.turn is not None:
            turn_payload = result.turn.model_dump(mode="json", exclude={"text"})
            turn_payload["text_sha256"] = identity_sha256(result.turn.text)
        return {
            "chunk": {
                "chunk_id": result.chunk.chunk_id,
                "turn_id": result.chunk.turn_id,
                "start_char": result.chunk.start_char,
                "end_char": result.chunk.end_char,
                "token_count": result.chunk.token_count,
                "text_sha256": identity_sha256(result.chunk.text),
                "embedding_sha256": (
                    None
                    if result.chunk.embedding is None
                    else identity_sha256(result.chunk.embedding)
                ),
                "lexical_weights": result.chunk.lexical_weights,
            },
            "turn": turn_payload,
            # Seal every scalar/ID used by ranking, grouping, ordering, or
            # prompt packing. Keeping this projection broad also prevents a
            # future RetrievalResult field from silently escaping the receipt.
            "retrieval": result.model_dump(
                mode="json",
                exclude={"chunk", "turn"},
            ),
        }

    return identity_sha256(
        {
            "format": CONVERSATION_ENVELOPE_RETRIEVAL_FORMAT,
            "rows": [receipt_row(result) for result in results],
        }
    )


@dataclass(frozen=True, slots=True)
class ConversationEnvelopeHydrationDiagnostic:
    """Text-free reason why one sealed plan group was left unexpanded."""

    envelope_id: str | None
    anchor_turn_ids: tuple[str, ...]
    reason: str
    chunk_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.envelope_id is not None and not self.envelope_id.strip():
            raise ValueError("diagnostic envelope_id must be non-empty when present")
        object.__setattr__(self, "anchor_turn_ids", tuple(self.anchor_turn_ids))
        object.__setattr__(self, "chunk_ids", tuple(self.chunk_ids))
        if not self.reason.strip():
            raise ValueError("hydration diagnostic reason must be non-empty")
        if any(not value.strip() for value in (*self.anchor_turn_ids, *self.chunk_ids)):
            raise ValueError("hydration diagnostic IDs must be non-empty")

    def identity_payload(self) -> dict[str, Any]:
        return reflect_payload(self)


@dataclass(frozen=True, slots=True)
class ConversationEnvelopeRetrievalExpansion(SealedIdentity):
    """Hydrated results plus a receipt that binds their exact durable IDs."""

    _SEAL_MISMATCH = "conversation envelope retrieval receipt mismatch"
    _PAYLOAD_EXCLUDE = frozenset({"results"})

    format: str
    results: tuple[RetrievalResult, ...]
    original_chunk_ids: tuple[str, ...]
    original_chunk_token_counts: tuple[int, ...]
    companion_chunk_ids: tuple[str, ...]
    companion_token_count: int
    max_companion_chunks: int
    max_companion_tokens: int
    duplicate_input_chunk_ids: tuple[str, ...]
    unhydrated_chunk_ids: tuple[str, ...]
    hydration_diagnostics: tuple[ConversationEnvelopeHydrationDiagnostic, ...]
    result_rows_sha256: str
    plan: ConversationEnvelopeExpansionPlan | None
    planner_failure_kind: str | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != CONVERSATION_ENVELOPE_RETRIEVAL_FORMAT:
            raise ValueError("conversation envelope retrieval format is unsupported")
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(
            self,
            "hydration_diagnostics",
            tuple(self.hydration_diagnostics),
        )
        for name in (
            "original_chunk_ids",
            "companion_chunk_ids",
            "duplicate_input_chunk_ids",
            "unhydrated_chunk_ids",
        ):
            values = tuple(str(value) for value in getattr(self, name))
            object.__setattr__(self, name, values)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
        result_ids = tuple(result.chunk.chunk_id for result in self.results)
        result_by_id = {result.chunk.chunk_id: result for result in self.results}
        if len(set(result_ids)) != len(result_ids):
            raise ValueError("expanded retrieval results must have unique chunk IDs")
        if not set(self.original_chunk_ids).issubset(result_ids):
            raise ValueError("every original chunk must survive expansion")
        if not set(self.companion_chunk_ids).issubset(result_ids):
            raise ValueError("every companion chunk must survive expansion")
        if set(self.original_chunk_ids) & set(self.companion_chunk_ids):
            raise ValueError("original and companion IDs must be disjoint")
        if set(result_ids) != set(
            (*self.original_chunk_ids, *self.companion_chunk_ids)
        ):
            raise ValueError(
                "every expanded row must be original or companion evidence"
            )
        original_token_counts = tuple(self.original_chunk_token_counts)
        if (
            len(original_token_counts) != len(self.original_chunk_ids)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in original_token_counts
            )
        ):
            raise ValueError("original token footprint must align with original IDs")
        expected_original_tokens = tuple(
            result_by_id[chunk_id].chunk.token_count
            for chunk_id in self.original_chunk_ids
        )
        if original_token_counts != expected_original_tokens:
            raise ValueError("original token footprint does not match hydrated rows")
        object.__setattr__(
            self,
            "original_chunk_token_counts",
            original_token_counts,
        )
        for name in ("max_companion_chunks", "max_companion_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        expected_companion_tokens = sum(
            result_by_id[chunk_id].chunk.token_count
            for chunk_id in self.companion_chunk_ids
        )
        if (
            isinstance(self.companion_token_count, bool)
            or not isinstance(self.companion_token_count, int)
            or self.companion_token_count < 0
            or self.companion_token_count != expected_companion_tokens
        ):
            raise ValueError("companion token count does not match hydrated rows")
        if len(self.companion_chunk_ids) > self.max_companion_chunks:
            raise ValueError("hydrated evidence exceeds max_companion_chunks")
        if self.companion_token_count > self.max_companion_tokens:
            raise ValueError("hydrated evidence exceeds max_companion_tokens")
        if any(
            result_by_id[chunk_id].route != CONVERSATION_ENVELOPE_MEMBER_ROUTE
            for chunk_id in self.companion_chunk_ids
        ):
            raise ValueError("every companion must carry the envelope-member route")
        if self.plan is not None:
            if self.plan.original_chunk_token_counts != tuple(
                zip(self.original_chunk_ids, self.original_chunk_token_counts)
            ):
                raise ValueError("sealed plan does not match the original footprint")
            if (
                self.plan.max_companion_chunks != self.max_companion_chunks
                or self.plan.max_companion_tokens != self.max_companion_tokens
            ):
                raise ValueError("sealed plan does not match companion bounds")
            planned_coordinates = {
                chunk_id: (turn_id, token_count, group)
                for group in self.plan.groups
                for chunk_id, turn_id, token_count in zip(
                    group.ordered_chunk_ids,
                    group.ordered_chunk_turn_ids,
                    group.ordered_chunk_token_counts,
                )
            }
            for chunk_id in self.companion_chunk_ids:
                coordinate = planned_coordinates.get(chunk_id)
                if coordinate is None:
                    raise ValueError("every companion must occur in the sealed plan")
                turn_id, token_count, group = coordinate
                result = result_by_id[chunk_id]
                anchor = result_by_id.get(str(result.anchor_chunk_id))
                if (
                    result.chunk.turn_id != turn_id
                    or result.chunk.token_count != token_count
                    or result.turn is None
                    or result.turn.turn_id != turn_id
                    or result.turn.source_id != group.source_id
                    or result.memory_source_id != group.source_id
                    or anchor is None
                    or anchor.chunk.turn_id not in group.anchor_turn_ids
                ):
                    raise ValueError("companion provenance disagrees with sealed plan")
        if _result_rows_sha256(self.results) != self.result_rows_sha256:
            raise ValueError("result row digest does not match hydrated evidence")
        if (self.plan is None) != (self.planner_failure_kind is not None):
            raise ValueError("planner failure state must agree with plan availability")
        if (
            self.planner_failure_kind is not None
            and not self.planner_failure_kind.strip()
        ):
            raise ValueError("planner_failure_kind must be non-empty")
        self._seal()

    def packing_contract(self) -> AtomicExpansionContract:
        """Project the sealed receipt into the packer's text-free contract."""

        result_ids = {result.chunk.chunk_id for result in self.results}
        original_ids = set(self.original_chunk_ids)
        failed_envelope_ids = {
            diagnostic.envelope_id
            for diagnostic in self.hydration_diagnostics
            if diagnostic.envelope_id is not None
        }
        groups: list[AtomicExpansionGroup] = []
        if self.plan is not None:
            for group in self.plan.groups:
                if (
                    group.envelope_id in failed_envelope_ids
                    or not set(group.ordered_chunk_ids).issubset(result_ids)
                ):
                    # Hydration is group-atomic. A planned group with any
                    # unavailable row was deliberately omitted from results.
                    continue
                group_originals = tuple(
                    chunk_id
                    for chunk_id in group.ordered_chunk_ids
                    if chunk_id in original_ids
                )
                groups.append(
                    AtomicExpansionGroup(
                        ordered_chunk_ids=group.ordered_chunk_ids,
                        original_chunk_ids=group_originals,
                    )
                )
        return AtomicExpansionContract(
            original_chunk_ids=self.original_chunk_ids,
            companion_chunk_ids=self.companion_chunk_ids,
            groups=tuple(groups),
            max_companion_chunks=self.max_companion_chunks,
        )


def hydrate_conversation_envelope_plan(
    original_results: Sequence[RetrievalResult],
    *,
    plan: ConversationEnvelopeExpansionPlan | None,
    hydrate_chunk: Callable[..., RetrievalResult | None],
    live_chunk_ids: Callable[[Sequence[str]], frozenset[str]],
    max_companion_chunks: int,
    max_companion_tokens: int,
    planner_failure_kind: str | None = None,
) -> ConversationEnvelopeRetrievalExpansion:
    """Merge one plan after selection; original objects win every collision."""

    originals: list[RetrievalResult] = []
    original_by_id: dict[str, RetrievalResult] = {}
    duplicate_ids: list[str] = []
    for result in original_results:
        if not isinstance(result, RetrievalResult):
            raise TypeError("original_results must contain RetrievalResult values")
        chunk_id = result.chunk.chunk_id
        if chunk_id in original_by_id:
            if chunk_id not in duplicate_ids:
                duplicate_ids.append(chunk_id)
            continue
        original_by_id[chunk_id] = result
        originals.append(result)

    for name, value in (
        ("max_companion_chunks", max_companion_chunks),
        ("max_companion_tokens", max_companion_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    original_ids = tuple(result.chunk.chunk_id for result in originals)
    original_token_counts = tuple(result.chunk.token_count for result in originals)

    def baseline(
        failure_kind: str,
        diagnostics: tuple[ConversationEnvelopeHydrationDiagnostic, ...] = (),
    ) -> ConversationEnvelopeRetrievalExpansion:
        rows = tuple(originals)
        return ConversationEnvelopeRetrievalExpansion(
            format=CONVERSATION_ENVELOPE_RETRIEVAL_FORMAT,
            results=rows,
            original_chunk_ids=original_ids,
            original_chunk_token_counts=original_token_counts,
            companion_chunk_ids=(),
            companion_token_count=0,
            max_companion_chunks=max_companion_chunks,
            max_companion_tokens=max_companion_tokens,
            duplicate_input_chunk_ids=tuple(duplicate_ids),
            unhydrated_chunk_ids=(),
            hydration_diagnostics=diagnostics,
            result_rows_sha256=_result_rows_sha256(rows),
            plan=None,
            planner_failure_kind=failure_kind,
        )

    if plan is None:
        return baseline(planner_failure_kind or "unknown_planner_failure")

    expected_footprint = tuple(zip(original_ids, original_token_counts))
    if plan.original_chunk_token_counts != expected_footprint:
        diagnostic = ConversationEnvelopeHydrationDiagnostic(
            None,
            (),
            "plan_original_footprint_mismatch",
        )
        return baseline("conversation_envelope_plan_footprint_mismatch", (diagnostic,))
    if (
        plan.max_companion_chunks != max_companion_chunks
        or plan.max_companion_tokens != max_companion_tokens
    ):
        diagnostic = ConversationEnvelopeHydrationDiagnostic(
            None,
            (),
            "plan_companion_bounds_mismatch",
        )
        return baseline("conversation_envelope_plan_bounds_mismatch", (diagnostic,))

    turn_indexes: dict[str, list[int]] = {}
    for index, result in enumerate(originals):
        turn_indexes.setdefault(result.chunk.turn_id, []).append(index)
    group_at_index: dict[int, tuple[Any, tuple[RetrievalResult, ...]]] = {}
    companion_ids: list[str] = []
    unhydrated_ids: list[str] = []
    hydration_diagnostics: list[ConversationEnvelopeHydrationDiagnostic] = []
    candidate_groups = plan.groups
    admitted_companion_chunks = 0
    admitted_companion_tokens = 0
    for group in candidate_groups:
        indexes = [
            index
            for turn_id in group.anchor_turn_ids
            for index in turn_indexes.get(turn_id, ())
        ]
        missing_anchor_turns = tuple(
            turn_id
            for turn_id in group.anchor_turn_ids
            if turn_id not in turn_indexes
        )
        if missing_anchor_turns:
            hydration_diagnostics.append(
                ConversationEnvelopeHydrationDiagnostic(
                    group.envelope_id,
                    group.anchor_turn_ids,
                    "absent_original_anchor",
                )
            )
            continue
        planned_ids = set(group.ordered_chunk_ids)
        anchor_chunk_ids = tuple(
            originals[index].chunk.chunk_id for index in indexes
        )
        if any(chunk_id not in planned_ids for chunk_id in anchor_chunk_ids):
            hydration_diagnostics.append(
                ConversationEnvelopeHydrationDiagnostic(
                    group.envelope_id,
                    group.anchor_turn_ids,
                    "anchor_chunk_missing_from_plan",
                    anchor_chunk_ids,
                )
            )
            continue
        try:
            live_before = frozenset(live_chunk_ids(group.ordered_chunk_ids))
        except Exception:
            hydration_diagnostics.append(
                ConversationEnvelopeHydrationDiagnostic(
                    group.envelope_id,
                    group.anchor_turn_ids,
                    "live_chunk_check_failed",
                    group.ordered_chunk_ids,
                )
            )
            continue
        not_live_before = tuple(
            chunk_id
            for chunk_id in group.ordered_chunk_ids
            if chunk_id not in live_before
        )
        if not_live_before:
            for chunk_id in not_live_before:
                if chunk_id not in unhydrated_ids:
                    unhydrated_ids.append(chunk_id)
            hydration_diagnostics.append(
                ConversationEnvelopeHydrationDiagnostic(
                    group.envelope_id,
                    group.anchor_turn_ids,
                    "chunk_not_live",
                    not_live_before,
                )
            )
            continue
        planned_companions = tuple(
            (chunk_id, token_count)
            for chunk_id, token_count in zip(
                group.ordered_chunk_ids,
                group.ordered_chunk_token_counts,
            )
            if chunk_id not in original_by_id
        )
        if (
            len(planned_companions) != group.companion_chunk_count
            or sum(token_count for _chunk_id, token_count in planned_companions)
            != group.companion_token_count
        ):
            hydration_diagnostics.append(
                ConversationEnvelopeHydrationDiagnostic(
                    group.envelope_id,
                    group.anchor_turn_ids,
                    "companion_footprint_mismatch",
                )
            )
            continue
        if (
            admitted_companion_chunks + len(planned_companions)
            > max_companion_chunks
            or admitted_companion_tokens
            + sum(token_count for _chunk_id, token_count in planned_companions)
            > max_companion_tokens
        ):
            hydration_diagnostics.append(
                ConversationEnvelopeHydrationDiagnostic(
                    group.envelope_id,
                    group.anchor_turn_ids,
                    "companion_bound",
                    tuple(chunk_id for chunk_id, _token_count in planned_companions),
                )
            )
            continue
        anchor = originals[min(indexes)]
        prepared: list[RetrievalResult] = []
        prepared_companions: list[str] = []
        failed_ids: list[str] = []
        coordinate_mismatch = False
        for chunk_id, expected_turn_id, expected_token_count in zip(
            group.ordered_chunk_ids,
            group.ordered_chunk_turn_ids,
            group.ordered_chunk_token_counts,
        ):
            selected = original_by_id.get(chunk_id)
            if selected is None:
                try:
                    selected = hydrate_chunk(
                        chunk_id,
                        score=float(anchor.score),
                        route=CONVERSATION_ENVELOPE_MEMBER_ROUTE,
                        anchor_chunk_id=anchor.chunk.chunk_id,
                    )
                except Exception:
                    selected = None
                if selected is None or selected.chunk.chunk_id != chunk_id:
                    failed_ids.append(chunk_id)
                    continue
                if (
                    selected.chunk.turn_id != expected_turn_id
                    or selected.chunk.token_count != expected_token_count
                    or selected.turn is None
                    or selected.turn.turn_id != expected_turn_id
                    or selected.turn.source_id != group.source_id
                    or (
                        expected_turn_id == group.opener_turn_id
                        and selected.turn.role != "user"
                    )
                ):
                    failed_ids.append(chunk_id)
                    coordinate_mismatch = True
                    continue
                selected = selected.model_copy(
                    update={
                        "route": CONVERSATION_ENVELOPE_MEMBER_ROUTE,
                        "anchor_chunk_id": anchor.chunk.chunk_id,
                        "memory_source_id": group.source_id,
                    }
                )
                prepared_companions.append(chunk_id)
            elif (
                selected.chunk.turn_id != expected_turn_id
                or selected.chunk.token_count != expected_token_count
                or selected.turn is None
                or selected.turn.turn_id != expected_turn_id
                or selected.turn.source_id != group.source_id
                or (
                    expected_turn_id == group.opener_turn_id
                    and selected.turn.role != "user"
                )
            ):
                failed_ids.append(chunk_id)
                coordinate_mismatch = True
                continue
            prepared.append(selected)
        if failed_ids:
            for chunk_id in failed_ids:
                if chunk_id not in unhydrated_ids:
                    unhydrated_ids.append(chunk_id)
            hydration_diagnostics.append(
                ConversationEnvelopeHydrationDiagnostic(
                    group.envelope_id,
                    group.anchor_turn_ids,
                    (
                        "evidence_coordinate_mismatch"
                        if coordinate_mismatch
                        else "chunk_unavailable"
                    ),
                    tuple(failed_ids),
                )
            )
            continue
        try:
            live_after = frozenset(live_chunk_ids(group.ordered_chunk_ids))
        except Exception:
            hydration_diagnostics.append(
                ConversationEnvelopeHydrationDiagnostic(
                    group.envelope_id,
                    group.anchor_turn_ids,
                    "live_chunk_check_failed",
                    group.ordered_chunk_ids,
                )
            )
            continue
        retired_during_hydration = tuple(
            chunk_id
            for chunk_id in group.ordered_chunk_ids
            if chunk_id not in live_after
        )
        if retired_during_hydration:
            for chunk_id in retired_during_hydration:
                if chunk_id not in unhydrated_ids:
                    unhydrated_ids.append(chunk_id)
            hydration_diagnostics.append(
                ConversationEnvelopeHydrationDiagnostic(
                    group.envelope_id,
                    group.anchor_turn_ids,
                    "chunk_retired_during_hydration",
                    retired_during_hydration,
                )
            )
            continue
        group_at_index[min(indexes)] = (group, tuple(prepared))
        companion_ids.extend(prepared_companions)
        admitted_companion_chunks += len(prepared_companions)
        admitted_companion_tokens += sum(
            selected.chunk.token_count
            for selected in prepared
            if selected.chunk.chunk_id in prepared_companions
        )

    output: list[RetrievalResult] = []
    emitted: set[str] = set()
    for index, original in enumerate(originals):
        prepared_group = group_at_index.get(index)
        if prepared_group is not None:
            _group, prepared = prepared_group
            for selected in prepared:
                chunk_id = selected.chunk.chunk_id
                if chunk_id in emitted:
                    continue
                output.append(selected)
                emitted.add(chunk_id)
        chunk_id = original.chunk.chunk_id
        if chunk_id not in emitted:
            output.append(original)
            emitted.add(chunk_id)

    rows = tuple(output)
    return ConversationEnvelopeRetrievalExpansion(
        format=CONVERSATION_ENVELOPE_RETRIEVAL_FORMAT,
        results=rows,
        original_chunk_ids=original_ids,
        original_chunk_token_counts=original_token_counts,
        companion_chunk_ids=tuple(companion_ids),
        companion_token_count=sum(
            result.chunk.token_count
            for result in rows
            if result.chunk.chunk_id in companion_ids
        ),
        max_companion_chunks=max_companion_chunks,
        max_companion_tokens=max_companion_tokens,
        duplicate_input_chunk_ids=tuple(duplicate_ids),
        unhydrated_chunk_ids=tuple(unhydrated_ids),
        hydration_diagnostics=tuple(hydration_diagnostics),
        result_rows_sha256=_result_rows_sha256(rows),
        plan=plan,
        planner_failure_kind=None,
    )


__all__ = [
    "CONVERSATION_ENVELOPE_MEMBER_ROUTE",
    "CONVERSATION_ENVELOPE_RETRIEVAL_FORMAT",
    "ConversationEnvelopeHydrationDiagnostic",
    "ConversationEnvelopeRetrievalExpansion",
    "hydrate_conversation_envelope_plan",
]
