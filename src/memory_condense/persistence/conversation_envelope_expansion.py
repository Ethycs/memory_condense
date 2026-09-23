"""Bounded, text-free retrieval plans over durable conversation envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.sealed import SealedIdentity, reflect_payload
from memory_condense.persistence.db import INDEXED_CHUNK_SQL

if TYPE_CHECKING:
    from memory_condense.persistence.conversation_envelope_store import (
        ConversationEnvelopeEvent,
        ConversationEnvelopeStore,
    )


CONVERSATION_ENVELOPE_EXPANSION_FORMAT = (
    "memory-condense-conversation-envelope-expansion-v2"
)
DEFAULT_MAX_EXPANSION_ENVELOPES = 4
DEFAULT_MAX_EXPANSION_TURNS = 8
DEFAULT_MAX_EXPANSION_COMPANION_CHUNKS = 16
DEFAULT_MAX_EXPANSION_COMPANION_TOKENS = 800
HARD_MAX_EXPANSION_ENVELOPES = 64
HARD_MAX_EXPANSION_TURNS = 128
HARD_MAX_EXPANSION_COMPANION_CHUNKS = 512
HARD_MAX_EXPANSION_COMPANION_TOKENS = 1_000_000


def _ordered_strings(values: Sequence[str], label: str) -> tuple[str, ...]:
    rows = tuple(str(value).strip() for value in values)
    if any(not value for value in rows):
        raise ValueError(f"{label} must be non-empty")
    if len(set(rows)) != len(rows):
        raise ValueError(f"{label} must be unique")
    return rows


def _bound(value: int, label: str, hard_maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= hard_maximum
    ):
        raise ValueError(f"{label} must be an integer from 1 through {hard_maximum}")
    return value


def _companion_bound(value: int, label: str, hard_maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= hard_maximum
    ):
        raise ValueError(f"{label} must be an integer from 0 through {hard_maximum}")
    return value


@dataclass(frozen=True, slots=True)
class ConversationEnvelopeExpansionDiagnostic:
    anchor_turn_id: str
    reason: str

    def __post_init__(self) -> None:
        if not self.anchor_turn_id.strip() or not self.reason.strip():
            raise ValueError("expansion diagnostics require an anchor and reason")

    def identity_payload(self) -> dict[str, Any]:
        return reflect_payload(self)


@dataclass(frozen=True, slots=True)
class ConversationEnvelopeExpansionGroup(SealedIdentity):
    """One source-safe envelope group containing identifiers and counts only."""

    _SEAL_MISMATCH = "conversation envelope expansion group receipt mismatch"

    envelope_id: str
    source_id: str
    opener_turn_id: str
    anchor_turn_ids: tuple[str, ...]
    selected_turn_ids: tuple[str, ...]
    selected_turn_ordinals: tuple[int, ...]
    ordered_chunk_ids: tuple[str, ...]
    ordered_chunk_turn_ids: tuple[str, ...]
    ordered_chunk_token_counts: tuple[int, ...]
    selected_token_count: int
    companion_chunk_count: int
    companion_token_count: int
    omitted_turn_count: int
    omitted_chunk_count: int
    truncation_reasons: tuple[str, ...] = ()
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("envelope_id", "source_id", "opener_turn_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        object.__setattr__(
            self,
            "anchor_turn_ids",
            _ordered_strings(self.anchor_turn_ids, "anchor turn IDs"),
        )
        object.__setattr__(
            self,
            "selected_turn_ids",
            _ordered_strings(self.selected_turn_ids, "selected turn IDs"),
        )
        object.__setattr__(
            self,
            "ordered_chunk_ids",
            _ordered_strings(self.ordered_chunk_ids, "ordered chunk IDs"),
        )
        ordinals = tuple(self.selected_turn_ordinals)
        if (
            len(ordinals) != len(self.selected_turn_ids)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in ordinals
            )
            or any(right <= left for left, right in zip(ordinals, ordinals[1:]))
        ):
            raise ValueError("selected turn ordinals must be strictly chronological")
        object.__setattr__(self, "selected_turn_ordinals", ordinals)
        chunk_turn_ids = tuple(
            str(value).strip() for value in self.ordered_chunk_turn_ids
        )
        if any(not value for value in chunk_turn_ids):
            raise ValueError("ordered chunk turn IDs must be non-empty")
        object.__setattr__(self, "ordered_chunk_turn_ids", chunk_turn_ids)
        token_counts = tuple(self.ordered_chunk_token_counts)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in token_counts
        ):
            raise ValueError("ordered chunk token counts must be non-negative integers")
        object.__setattr__(self, "ordered_chunk_token_counts", token_counts)
        object.__setattr__(
            self,
            "truncation_reasons",
            tuple(
                dict.fromkeys(
                    str(value).strip() for value in self.truncation_reasons
                )
            ),
        )
        if any(not value for value in self.truncation_reasons):
            raise ValueError("truncation reasons must be non-empty")
        if (
            not self.anchor_turn_ids
            or not self.selected_turn_ids
            or not self.ordered_chunk_ids
        ):
            raise ValueError("an expansion group requires anchors, turns, and chunks")
        if not (
            len(self.ordered_chunk_ids)
            == len(self.ordered_chunk_turn_ids)
            == len(self.ordered_chunk_token_counts)
        ):
            raise ValueError("ordered chunk coordinates must have equal lengths")
        if tuple(dict.fromkeys(self.ordered_chunk_turn_ids)) != self.selected_turn_ids:
            raise ValueError(
                "ordered chunk coordinates must cover turns chronologically"
            )
        if self.selected_turn_ids[0] != self.opener_turn_id:
            raise ValueError("the user opener must be the first selected turn")
        if not set(self.anchor_turn_ids).issubset(self.selected_turn_ids):
            raise ValueError("every anchor turn must be selected")
        if min(
            self.selected_token_count,
            self.companion_chunk_count,
            self.companion_token_count,
            self.omitted_turn_count,
            self.omitted_chunk_count,
        ) < 0:
            raise ValueError("expansion group counts must be non-negative")
        if sum(self.ordered_chunk_token_counts) != self.selected_token_count:
            raise ValueError(
                "selected token count must match ordered chunk coordinates"
            )
        if self.companion_chunk_count > len(self.ordered_chunk_ids):
            raise ValueError("companion chunk count exceeds selected chunks")
        if self.companion_token_count > self.selected_token_count:
            raise ValueError("companion token count exceeds selected tokens")
        if (
            self.omitted_turn_count or self.omitted_chunk_count
        ) and not self.truncation_reasons:
            raise ValueError("truncated expansion groups require a reason")
        self._seal()


@dataclass(frozen=True, slots=True)
class ConversationEnvelopeExpansionPlan(SealedIdentity):
    """A bounded and replay-verifiable plan that contains no transcript text."""

    _SEAL_MISMATCH = "conversation envelope expansion plan receipt mismatch"

    format: str
    policy_sha256: str
    anchor_turn_ids: tuple[str, ...]
    original_chunk_token_counts: tuple[tuple[str, int], ...]
    groups: tuple[ConversationEnvelopeExpansionGroup, ...]
    diagnostics: tuple[ConversationEnvelopeExpansionDiagnostic, ...]
    max_envelopes: int
    max_turns_per_envelope: int
    max_companion_chunks: int
    max_companion_tokens: int
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != CONVERSATION_ENVELOPE_EXPANSION_FORMAT:
            raise ValueError("conversation envelope expansion format is unsupported")
        if len(self.policy_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.policy_sha256
        ):
            raise ValueError("policy_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(
            self,
            "anchor_turn_ids",
            _ordered_strings(self.anchor_turn_ids, "plan anchor turn IDs")
            if self.anchor_turn_ids
            else (),
        )
        original_rows = tuple(
            (str(chunk_id).strip(), token_count)
            for chunk_id, token_count in self.original_chunk_token_counts
        )
        if any(
            not chunk_id
            or isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
            for chunk_id, token_count in original_rows
        ):
            raise ValueError(
                "original chunk footprints must contain IDs and token counts"
            )
        if len(
            {chunk_id for chunk_id, _token_count in original_rows}
        ) != len(original_rows):
            raise ValueError("original chunk footprints must have unique IDs")
        object.__setattr__(self, "original_chunk_token_counts", original_rows)
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        _bound(self.max_envelopes, "max_envelopes", HARD_MAX_EXPANSION_ENVELOPES)
        _bound(
            self.max_turns_per_envelope,
            "max_turns_per_envelope",
            HARD_MAX_EXPANSION_TURNS,
        )
        _companion_bound(
            self.max_companion_chunks,
            "max_companion_chunks",
            HARD_MAX_EXPANSION_COMPANION_CHUNKS,
        )
        _companion_bound(
            self.max_companion_tokens,
            "max_companion_tokens",
            HARD_MAX_EXPANSION_COMPANION_TOKENS,
        )
        group_anchors = tuple(
            anchor for group in self.groups for anchor in group.anchor_turn_ids
        )
        diagnostic_anchors = tuple(row.anchor_turn_id for row in self.diagnostics)
        if len(set(group_anchors)) != len(group_anchors):
            raise ValueError("an anchor cannot occur in multiple expansion groups")
        if len(set(diagnostic_anchors)) != len(diagnostic_anchors):
            raise ValueError("an anchor cannot have multiple terminal diagnostics")
        if set(group_anchors) & set(diagnostic_anchors):
            raise ValueError("a planned anchor cannot also fail open")
        if set((*group_anchors, *diagnostic_anchors)) != set(self.anchor_turn_ids):
            raise ValueError("every plan anchor requires one terminal outcome")
        if len(self.groups) > self.max_envelopes:
            raise ValueError("expansion plan exceeds max_envelopes")
        if len({group.envelope_id for group in self.groups}) != len(self.groups):
            raise ValueError("an envelope may be expanded only once")
        planned_chunk_ids = tuple(
            chunk_id for group in self.groups for chunk_id in group.ordered_chunk_ids
        )
        if len(set(planned_chunk_ids)) != len(planned_chunk_ids):
            raise ValueError("a chunk cannot occur in multiple expansion groups")
        if any(
            len(group.selected_turn_ids) > self.max_turns_per_envelope
            for group in self.groups
        ):
            raise ValueError("expansion plan exceeds max_turns_per_envelope")
        anchor_positions = {
            turn_id: index for index, turn_id in enumerate(self.anchor_turn_ids)
        }
        group_positions = [
            min(anchor_positions[turn_id] for turn_id in group.anchor_turn_ids)
            for group in self.groups
        ]
        if group_positions != sorted(group_positions):
            raise ValueError("expansion groups must follow earliest anchor order")
        if (
            sum(group.companion_chunk_count for group in self.groups)
            > self.max_companion_chunks
        ):
            raise ValueError("expansion plan exceeds max_companion_chunks")
        if (
            sum(group.companion_token_count for group in self.groups)
            > self.max_companion_tokens
        ):
            raise ValueError("expansion plan exceeds max_companion_tokens")
        self._seal()


@dataclass(frozen=True, slots=True)
class _ChunkCoordinate:
    chunk_id: str
    turn_id: str
    token_count: int
    ordinal: int
    start_char: int
    end_char: int


def _turn_chunks(
    store: ConversationEnvelopeStore,
    turn_ids: Sequence[str],
) -> dict[str, tuple[_ChunkCoordinate, ...]]:
    selected = tuple(dict.fromkeys(turn_ids))
    if not selected:
        return {}
    placeholders = ",".join("?" for _ in selected)
    rows = store._db.execute(
        "SELECT c.chunk_id, c.turn_id, c.token_count, t.ordinal, "
        "c.start_char, c.end_char FROM chunks AS c "
        "JOIN turns AS t ON t.turn_id = c.turn_id "
        f"WHERE c.turn_id IN ({placeholders}) AND {INDEXED_CHUNK_SQL} "
        "ORDER BY t.ordinal, c.start_char, c.end_char, c.chunk_id",
        selected,
    ).fetchall()
    by_turn: dict[str, list[_ChunkCoordinate]] = {turn_id: [] for turn_id in selected}
    for row in rows:
        coordinate = _ChunkCoordinate(
            chunk_id=str(row[0]),
            turn_id=str(row[1]),
            token_count=max(0, int(row[2])),
            ordinal=int(row[3]),
            start_char=int(row[4]),
            end_char=int(row[5]),
        )
        by_turn[coordinate.turn_id].append(coordinate)
    return {turn_id: tuple(values) for turn_id, values in by_turn.items()}


def _missing_event_reason(
    store: ConversationEnvelopeStore,
    turn_id: str,
) -> str:
    assignment = store.assignment(turn_id)
    if assignment is None:
        return "missing_or_pending_assignment"
    if assignment.status == "no_anchor":
        return f"no_anchor:{assignment.terminal_reason or 'unknown'}"
    return "ready_event_missing"


def plan_retrieval_expansion(
    store: ConversationEnvelopeStore,
    anchor_turn_ids: Sequence[str],
    *,
    original_chunk_token_counts: Mapping[str, int] | None = None,
    max_envelopes: int = DEFAULT_MAX_EXPANSION_ENVELOPES,
    max_turns_per_envelope: int = DEFAULT_MAX_EXPANSION_TURNS,
    max_companion_chunks: int = DEFAULT_MAX_EXPANSION_COMPANION_CHUNKS,
    max_companion_tokens: int = DEFAULT_MAX_EXPANSION_COMPANION_TOKENS,
) -> ConversationEnvelopeExpansionPlan:
    """Plan whole-turn envelope hydration without reading transcript text."""

    max_envelopes = _bound(
        max_envelopes, "max_envelopes", HARD_MAX_EXPANSION_ENVELOPES
    )
    max_turns_per_envelope = _bound(
        max_turns_per_envelope,
        "max_turns_per_envelope",
        HARD_MAX_EXPANSION_TURNS,
    )
    max_companion_chunks = _companion_bound(
        max_companion_chunks,
        "max_companion_chunks",
        HARD_MAX_EXPANSION_COMPANION_CHUNKS,
    )
    max_companion_tokens = _companion_bound(
        max_companion_tokens,
        "max_companion_tokens",
        HARD_MAX_EXPANSION_COMPANION_TOKENS,
    )
    original_rows = tuple((original_chunk_token_counts or {}).items())
    original_ids: set[str] = set()
    normalized_original_rows: list[tuple[str, int]] = []
    for chunk_id, token_count in original_rows:
        normalized_id = str(chunk_id).strip()
        if (
            not normalized_id
            or normalized_id in original_ids
            or isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            raise ValueError(
                "original chunk footprints must be unique and non-negative"
            )
        original_ids.add(normalized_id)
        normalized_original_rows.append((normalized_id, token_count))
    anchors = tuple(
        dict.fromkeys(str(turn_id).strip() for turn_id in anchor_turn_ids)
    )
    if any(not turn_id for turn_id in anchors):
        raise ValueError("anchor turn IDs must be non-empty")

    diagnostics: list[ConversationEnvelopeExpansionDiagnostic] = []
    events: dict[str, ConversationEnvelopeEvent] = {}
    envelope_order: list[str] = []
    anchors_by_envelope: dict[str, list[str]] = {}
    for turn_id in anchors:
        event = store.event_for_turn(turn_id)
        if event is None:
            diagnostics.append(
                ConversationEnvelopeExpansionDiagnostic(
                    turn_id,
                    _missing_event_reason(store, turn_id),
                )
            )
            continue
        events[turn_id] = event
        if event.envelope_id not in anchors_by_envelope:
            envelope_order.append(event.envelope_id)
            anchors_by_envelope[event.envelope_id] = []
        anchors_by_envelope[event.envelope_id].append(turn_id)

    admitted_envelopes = set(envelope_order[:max_envelopes])
    for envelope_id in envelope_order[max_envelopes:]:
        diagnostics.extend(
            ConversationEnvelopeExpansionDiagnostic(turn_id, "max_envelopes")
            for turn_id in anchors_by_envelope[envelope_id]
        )

    groups: list[ConversationEnvelopeExpansionGroup] = []
    used_companion_chunks = 0
    used_companion_tokens = 0
    for envelope_id in envelope_order:
        if envelope_id not in admitted_envelopes:
            continue
        group_anchor_ids = tuple(anchors_by_envelope[envelope_id])
        anchor_events = tuple(events[turn_id] for turn_id in group_anchor_ids)
        first = anchor_events[0]
        opener = store.event_for_turn(first.opener_turn_id)
        if any(
            event.envelope_id != envelope_id
            or event.source_id != first.source_id
            or event.opener_turn_id != first.opener_turn_id
            for event in anchor_events
        ) or (
            opener is None
            or opener.envelope_id != envelope_id
            or opener.source_id != first.source_id
            or opener.opener_turn_id != first.opener_turn_id
            or opener.event_kind != "open"
            or opener.actor_kind != "user"
        ):
            diagnostics.extend(
                ConversationEnvelopeExpansionDiagnostic(
                    turn_id,
                    "inconsistent_envelope",
                )
                for turn_id in group_anchor_ids
            )
            continue

        required_turn_ids = tuple(
            dict.fromkeys((first.opener_turn_id, *group_anchor_ids))
        )
        if len(required_turn_ids) > max_turns_per_envelope:
            diagnostics.extend(
                ConversationEnvelopeExpansionDiagnostic(turn_id, "mandatory_turn_bound")
                for turn_id in group_anchor_ids
            )
            continue
        required_chunks = _turn_chunks(store, required_turn_ids)
        if any(not required_chunks.get(turn_id) for turn_id in required_turn_ids):
            diagnostics.extend(
                ConversationEnvelopeExpansionDiagnostic(
                    turn_id,
                    "mandatory_turn_has_no_chunks",
                )
                for turn_id in group_anchor_ids
            )
            continue
        required_rows = tuple(
            row for turn_id in required_turn_ids for row in required_chunks[turn_id]
        )
        required_companions = tuple(
            row for row in required_rows if row.chunk_id not in original_ids
        )
        required_companion_tokens = sum(
            row.token_count for row in required_companions
        )
        if (
            used_companion_chunks + len(required_companions) > max_companion_chunks
            or used_companion_tokens + required_companion_tokens
            > max_companion_tokens
        ):
            diagnostics.extend(
                ConversationEnvelopeExpansionDiagnostic(
                    turn_id,
                    "mandatory_companion_chunk_or_token_bound",
                )
                for turn_id in group_anchor_ids
            )
            continue

        total_turns, total_chunks = store._db.execute(
            "SELECT COUNT(DISTINCT e.turn_id), COUNT(c.chunk_id) "
            "FROM conversation_envelope_events AS e "
            "LEFT JOIN chunks AS c ON c.turn_id = e.turn_id "
            f"AND {INDEXED_CHUNK_SQL} "
            "WHERE e.policy_sha256 = ? AND e.envelope_id = ?",
            (store.policy_sha256, envelope_id),
        ).fetchone()
        remaining_slots = max_turns_per_envelope - len(required_turn_ids)
        candidate_events: list[tuple[str, int]] = []
        candidate_scan_capped = False
        if remaining_slots:
            exclusions = ",".join("?" for _ in required_turn_ids)
            anchor_ordinals = tuple(event.turn_ordinal for event in anchor_events)
            distances = ["ABS(turn_ordinal - ?)" for _ in anchor_ordinals]
            distance_sql = (
                distances[0]
                if len(distances) == 1
                else "MIN(" + ",".join(distances) + ")"
            )
            scan_limit = min(HARD_MAX_EXPANSION_TURNS * 4, max_turns_per_envelope * 4)
            rows = store._db.execute(
                "SELECT turn_id, turn_ordinal, source_id, opener_turn_id "
                "FROM conversation_envelope_events WHERE policy_sha256 = ? "
                "AND envelope_id = ? "
                f"AND turn_id NOT IN ({exclusions}) "
                f"ORDER BY {distance_sql}, turn_ordinal, turn_id LIMIT ?",
                (
                    store.policy_sha256,
                    envelope_id,
                    *required_turn_ids,
                    *anchor_ordinals,
                    scan_limit,
                ),
            ).fetchall()
            if any(
                str(row[2]) != first.source_id
                or str(row[3]) != first.opener_turn_id
                for row in rows
            ):
                diagnostics.extend(
                    ConversationEnvelopeExpansionDiagnostic(
                        turn_id,
                        "inconsistent_envelope",
                    )
                    for turn_id in group_anchor_ids
                )
                continue
            candidate_events = [(str(row[0]), int(row[1])) for row in rows]
            candidate_scan_capped = len(rows) < max(
                0,
                int(total_turns) - len(required_turn_ids),
            )

        candidate_chunks = _turn_chunks(
            store,
            tuple(turn_id for turn_id, _ordinal in candidate_events),
        )
        selected_candidates: list[tuple[str, int]] = []
        skipped_for_chunks = False
        skipped_for_tokens = False
        local_companion_chunks = len(required_companions)
        local_companion_tokens = required_companion_tokens
        for turn_id, ordinal in candidate_events:
            if len(selected_candidates) >= remaining_slots:
                break
            rows = candidate_chunks.get(turn_id, ())
            if not rows:
                continue
            companion_rows = tuple(
                row for row in rows if row.chunk_id not in original_ids
            )
            row_tokens = sum(row.token_count for row in companion_rows)
            if (
                used_companion_chunks
                + local_companion_chunks
                + len(companion_rows)
                > max_companion_chunks
            ):
                skipped_for_chunks = True
                continue
            if (
                used_companion_tokens
                + local_companion_tokens
                + row_tokens
                > max_companion_tokens
            ):
                skipped_for_tokens = True
                continue
            selected_candidates.append((turn_id, ordinal))
            local_companion_chunks += len(companion_rows)
            local_companion_tokens += row_tokens

        selected_turn_coordinates = sorted(
            {
                opener.turn_id: opener.turn_ordinal,
                **{
                    event.turn_id: event.turn_ordinal for event in anchor_events
                },
                **dict(selected_candidates),
            }.items(),
            key=lambda item: (item[1], item[0]),
        )
        selected_turn_ids = tuple(
            turn_id for turn_id, _ordinal in selected_turn_coordinates
        )
        chunks_by_turn = {**required_chunks, **candidate_chunks}
        ordered_rows = tuple(
            row for turn_id in selected_turn_ids for row in chunks_by_turn[turn_id]
        )
        truncation_reasons: list[str] = []
        omitted_turns = max(0, int(total_turns) - len(selected_turn_ids))
        omitted_chunks = max(0, int(total_chunks) - len(ordered_rows))
        if omitted_turns:
            truncation_reasons.append("turn_bound")
        if skipped_for_chunks:
            truncation_reasons.append("chunk_bound")
        if skipped_for_tokens:
            truncation_reasons.append("token_bound")
        if candidate_scan_capped:
            truncation_reasons.append("candidate_scan_bound")
        groups.append(
            ConversationEnvelopeExpansionGroup(
                envelope_id=envelope_id,
                source_id=first.source_id,
                opener_turn_id=first.opener_turn_id,
                anchor_turn_ids=group_anchor_ids,
                selected_turn_ids=tuple(selected_turn_ids),
                selected_turn_ordinals=tuple(
                    ordinal for _turn_id, ordinal in selected_turn_coordinates
                ),
                ordered_chunk_ids=tuple(row.chunk_id for row in ordered_rows),
                ordered_chunk_turn_ids=tuple(row.turn_id for row in ordered_rows),
                ordered_chunk_token_counts=tuple(
                    row.token_count for row in ordered_rows
                ),
                selected_token_count=sum(row.token_count for row in ordered_rows),
                companion_chunk_count=sum(
                    row.chunk_id not in original_ids for row in ordered_rows
                ),
                companion_token_count=sum(
                    row.token_count
                    for row in ordered_rows
                    if row.chunk_id not in original_ids
                ),
                omitted_turn_count=omitted_turns,
                omitted_chunk_count=omitted_chunks,
                truncation_reasons=tuple(truncation_reasons),
            )
        )
        used_companion_chunks += sum(
            row.chunk_id not in original_ids for row in ordered_rows
        )
        used_companion_tokens += sum(
            row.token_count
            for row in ordered_rows
            if row.chunk_id not in original_ids
        )

    return ConversationEnvelopeExpansionPlan(
        format=CONVERSATION_ENVELOPE_EXPANSION_FORMAT,
        policy_sha256=store.policy_sha256,
        anchor_turn_ids=anchors,
        original_chunk_token_counts=tuple(normalized_original_rows),
        groups=tuple(groups),
        diagnostics=tuple(diagnostics),
        max_envelopes=max_envelopes,
        max_turns_per_envelope=max_turns_per_envelope,
        max_companion_chunks=max_companion_chunks,
        max_companion_tokens=max_companion_tokens,
    )


__all__ = [
    "CONVERSATION_ENVELOPE_EXPANSION_FORMAT",
    "DEFAULT_MAX_EXPANSION_COMPANION_CHUNKS",
    "DEFAULT_MAX_EXPANSION_COMPANION_TOKENS",
    "DEFAULT_MAX_EXPANSION_ENVELOPES",
    "DEFAULT_MAX_EXPANSION_TURNS",
    "ConversationEnvelopeExpansionDiagnostic",
    "ConversationEnvelopeExpansionGroup",
    "ConversationEnvelopeExpansionPlan",
    "plan_retrieval_expansion",
]
