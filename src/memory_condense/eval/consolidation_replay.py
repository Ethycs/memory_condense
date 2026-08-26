"""Chronological rank-vs-Qwen consolidation replay over a compiled corpus.

This is a development experiment, not the 95% answer-stage benchmark.  It
reuses already computed chunk embeddings, reveals turns in causal order, and
learns only from bounded direct retrievals made by later user prompts.  Three
isolated final stores then answer the same held-out probes at the same token
budget: no consolidation, rank-weighted consolidation, and Qwen QK/OV-weighted
consolidation.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableSequence, Sequence

import numpy as np

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.integrity import file_sha256 as _sha256
from memory_condense.application.condenser import MemoryCondenser
from memory_condense.search.packing.context_packer import ContextBudget
from memory_condense.modeling.embedding import EmbeddingService
from memory_condense.eval.answer_value_coverage import contains_answer
from memory_condense.tooling.qwen_consolidation import load_qwen_linker
from memory_condense.domain.schemas import Chunk, PackedContext


RETRIEVAL_ACCESS_EVENT_FORMAT = "memory-condense.retrieval-access-event.v1"


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    """One bounded slice of a causally completed interaction."""

    event_id: str
    now_turn: int
    user_text: str
    chunk_ids: tuple[str, ...]
    causal_chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalAccessEvent:
    """Immutable, text-free receipt for one historical direct retrieval."""

    event_id: str
    now_turn: int
    chunk_ids: tuple[str, ...]
    event_sha256: str = field(init=False)

    def identity_payload(self) -> dict[str, object]:
        """Return the canonical ordered-ID projection bound by the event seal."""

        return {
            "format": RETRIEVAL_ACCESS_EVENT_FORMAT,
            "event_id": self.event_id,
            "now_turn": self.now_turn,
            "chunk_ids": self.chunk_ids,
        }

    def __post_init__(self) -> None:
        if (
            type(self.event_id) is not str
            or not self.event_id
            or self.event_id != self.event_id.strip()
        ):
            raise ValueError("event_id must be an exact non-empty string")
        if type(self.now_turn) is not int or self.now_turn < 0:
            raise ValueError("now_turn must be an exact non-negative integer")
        if type(self.chunk_ids) is not tuple:
            raise TypeError("chunk_ids must be an exact tuple")
        if any(
            type(chunk_id) is not str
            or not chunk_id
            or chunk_id != chunk_id.strip()
            for chunk_id in self.chunk_ids
        ):
            raise ValueError("chunk_ids must contain only exact non-empty IDs")
        event_id = self.event_id
        now_turn = self.now_turn
        chunk_ids = self.chunk_ids
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "now_turn", now_turn)
        object.__setattr__(self, "chunk_ids", chunk_ids)
        object.__setattr__(
            self,
            "event_sha256",
            identity_sha256(self.identity_payload()),
        )


RETRIEVAL_ACCESS_CAPTURE_FORMAT = (
    "memory-condense.retrieval-access-capture.v1"
)
_RETRIEVAL_ACCESS_CAPTURE_AUTHORITY = secrets.token_bytes(32)


class RetrievalAccessCaptureValidationError(ValueError):
    """Raised when a staging-issued direct-retrieval capture is invalid."""


@dataclass(frozen=True, slots=True)
class RetrievalAccessCapture:
    """Text-free, transiently authorized projection of direct retrievals.

    ``capture_sha256`` is deterministic and safe to persist. The private
    authorization tag is process-local and exists only to prevent a caller
    from feeding an independently re-sealed event list to the history sealer.
    It is intentionally excluded from equality, repr, and ``payload``.
    """

    format: str
    source_database_sha256: str
    capture_policy_sha256: str
    retrieval_k: int
    expansion_tokens: int
    max_prompt_tokens: int
    events: tuple[RetrievalAccessEvent, ...]
    retained_request_token_state_bytes: int
    capture_sha256: str
    _authorization_tag: str = field(repr=False, compare=False)

    def payload(self) -> dict[str, object]:
        """Return the deterministic, text-free projection (never its tag)."""

        return _retrieval_access_capture_payload(
            source_database_sha256=self.source_database_sha256,
            capture_policy_sha256=self.capture_policy_sha256,
            retrieval_k=self.retrieval_k,
            expansion_tokens=self.expansion_tokens,
            max_prompt_tokens=self.max_prompt_tokens,
            events=self.events,
        )


@dataclass(slots=True)
class RetrievalAccessCaptureSink:
    """Single-use output slot populated only by ``stage_causal_store``."""

    _capture: RetrievalAccessCapture | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def capture(self) -> RetrievalAccessCapture:
        """Return the completed capture or fail if staging did not finish."""

        if self._capture is None:
            raise RetrievalAccessCaptureValidationError(
                "retrieval access capture is unavailable before staging completes"
            )
        return verify_retrieval_access_capture(self._capture)


def _require_capture_digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RetrievalAccessCaptureValidationError(
            f"{label} must be an exact lowercase SHA-256 digest"
        )
    return value


def _retrieval_access_capture_payload(
    *,
    source_database_sha256: str,
    capture_policy_sha256: str,
    retrieval_k: int,
    expansion_tokens: int,
    max_prompt_tokens: int,
    events: Sequence[RetrievalAccessEvent],
) -> dict[str, object]:
    return {
        "format": RETRIEVAL_ACCESS_CAPTURE_FORMAT,
        "source_database_sha256": source_database_sha256,
        "capture_policy_sha256": capture_policy_sha256,
        "retrieval_k": retrieval_k,
        "expansion_tokens": expansion_tokens,
        "max_prompt_tokens": max_prompt_tokens,
        "direct_expansion_only": True,
        "event_id_scheme": "causal-user:{ordinal}",
        "capture_point": (
            "after_direct_context_pack_before_current_user_append"
        ),
        "exclude_current_and_future_turns": True,
        "ordered_events": [
            {
                "event_id": event.event_id,
                "now_turn": event.now_turn,
                "event_sha256": event.event_sha256,
            }
            for event in events
        ],
        "event_count": len(events),
        "retained_request_token_state_bytes": 0,
    }


def _validated_capture_events(
    events: object,
) -> tuple[RetrievalAccessEvent, ...]:
    if type(events) is not tuple:
        raise RetrievalAccessCaptureValidationError(
            "capture events must be an exact tuple"
        )
    for index, event in enumerate(events):
        label = f"capture events[{index}]"
        if type(event) is not RetrievalAccessEvent:
            raise RetrievalAccessCaptureValidationError(
                f"{label} must be an exact RetrievalAccessEvent"
            )
        if (
            type(event.event_id) is not str
            or not event.event_id
            or event.event_id != event.event_id.strip()
        ):
            raise RetrievalAccessCaptureValidationError(
                f"{label}.event_id must be an exact non-empty string"
            )
        if type(event.now_turn) is not int or event.now_turn < 0:
            raise RetrievalAccessCaptureValidationError(
                f"{label}.now_turn must be an exact non-negative integer"
            )
        if type(event.chunk_ids) is not tuple or any(
            type(chunk_id) is not str
            or not chunk_id
            or chunk_id != chunk_id.strip()
            for chunk_id in event.chunk_ids
        ):
            raise RetrievalAccessCaptureValidationError(
                f"{label}.chunk_ids must be an exact tuple of non-empty IDs"
            )
        event_sha256 = _require_capture_digest(
            event.event_sha256,
            f"{label}.event_sha256",
        )
        if event_sha256 != identity_sha256(event.identity_payload()):
            raise RetrievalAccessCaptureValidationError(
                f"{label}.event_sha256 does not match its ordered ID payload"
            )
    return events


def retrieval_access_capture_sha256(
    *,
    source_database_sha256: str,
    capture_policy_sha256: str,
    retrieval_k: int,
    expansion_tokens: int,
    max_prompt_tokens: int,
    events: Sequence[RetrievalAccessEvent],
) -> str:
    """Return the deterministic persisted seal for one capture projection."""

    return identity_sha256(
        _retrieval_access_capture_payload(
            source_database_sha256=source_database_sha256,
            capture_policy_sha256=capture_policy_sha256,
            retrieval_k=retrieval_k,
            expansion_tokens=expansion_tokens,
            max_prompt_tokens=max_prompt_tokens,
            events=events,
        )
    )


def _mint_retrieval_access_capture(
    *,
    source_database_sha256: str,
    capture_policy_sha256: str,
    retrieval_k: int,
    expansion_tokens: int,
    max_prompt_tokens: int,
    events: Sequence[RetrievalAccessEvent],
) -> RetrievalAccessCapture:
    """Mint the process-local capability; called only at the staging boundary."""

    source_digest = _require_capture_digest(
        source_database_sha256,
        "capture source_database_sha256",
    )
    policy_digest = _require_capture_digest(
        capture_policy_sha256,
        "capture capture_policy_sha256",
    )
    for value, label in (
        (retrieval_k, "retrieval_k"),
        (expansion_tokens, "expansion_tokens"),
        (max_prompt_tokens, "max_prompt_tokens"),
    ):
        if type(value) is not int or value < 1:
            raise RetrievalAccessCaptureValidationError(
                f"capture {label} must be an exact positive integer"
            )
    normalized_events = _validated_capture_events(tuple(events))
    capture_sha256 = retrieval_access_capture_sha256(
        source_database_sha256=source_digest,
        capture_policy_sha256=policy_digest,
        retrieval_k=retrieval_k,
        expansion_tokens=expansion_tokens,
        max_prompt_tokens=max_prompt_tokens,
        events=normalized_events,
    )
    authorization_tag = hmac.new(
        _RETRIEVAL_ACCESS_CAPTURE_AUTHORITY,
        capture_sha256.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return RetrievalAccessCapture(
        format=RETRIEVAL_ACCESS_CAPTURE_FORMAT,
        source_database_sha256=source_digest,
        capture_policy_sha256=policy_digest,
        retrieval_k=retrieval_k,
        expansion_tokens=expansion_tokens,
        max_prompt_tokens=max_prompt_tokens,
        events=normalized_events,
        retained_request_token_state_bytes=0,
        capture_sha256=capture_sha256,
        _authorization_tag=authorization_tag,
    )


def verify_retrieval_access_capture(
    capture: object,
) -> RetrievalAccessCapture:
    """Verify deterministic content and transient staging authorization."""

    if type(capture) is not RetrievalAccessCapture:
        raise RetrievalAccessCaptureValidationError(
            "history sealing requires a staging-issued RetrievalAccessCapture"
        )
    if capture.format != RETRIEVAL_ACCESS_CAPTURE_FORMAT:
        raise RetrievalAccessCaptureValidationError(
            "unsupported retrieval access capture"
        )
    source_digest = _require_capture_digest(
        capture.source_database_sha256,
        "capture source_database_sha256",
    )
    policy_digest = _require_capture_digest(
        capture.capture_policy_sha256,
        "capture capture_policy_sha256",
    )
    for value, label in (
        (capture.retrieval_k, "retrieval_k"),
        (capture.expansion_tokens, "expansion_tokens"),
        (capture.max_prompt_tokens, "max_prompt_tokens"),
    ):
        if type(value) is not int or value < 1:
            raise RetrievalAccessCaptureValidationError(
                f"capture {label} must be an exact positive integer"
            )
    events = _validated_capture_events(capture.events)
    if (
        type(capture.retained_request_token_state_bytes) is not int
        or capture.retained_request_token_state_bytes != 0
    ):
        raise RetrievalAccessCaptureValidationError(
            "retrieval access capture may not retain request token state"
        )
    expected_sha256 = retrieval_access_capture_sha256(
        source_database_sha256=source_digest,
        capture_policy_sha256=policy_digest,
        retrieval_k=capture.retrieval_k,
        expansion_tokens=capture.expansion_tokens,
        max_prompt_tokens=capture.max_prompt_tokens,
        events=events,
    )
    capture_sha256 = _require_capture_digest(
        capture.capture_sha256,
        "capture capture_sha256",
    )
    if capture_sha256 != expected_sha256:
        raise RetrievalAccessCaptureValidationError(
            "retrieval access capture seal mismatch"
        )
    expected_tag = hmac.new(
        _RETRIEVAL_ACCESS_CAPTURE_AUTHORITY,
        expected_sha256.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    if (
        type(capture._authorization_tag) is not str
        or not hmac.compare_digest(capture._authorization_tag, expected_tag)
    ):
        raise RetrievalAccessCaptureValidationError(
            "retrieval access capture was not issued by this staging process"
        )
    return capture


@dataclass(frozen=True, slots=True)
class _SourceReplayRow:
    """Exact source-store turn identity plus its already embedded chunks."""

    ordinal: int
    turn_id: str
    role: str
    text: str
    source_id: str | None
    created_at: str | None
    chunks: tuple[Chunk, ...]


class FrozenQueryEmbedder:
    """Read-only lookup for query vectors batched before Qwen is loaded."""

    def __init__(self, vectors: Mapping[str, Sequence[float]]) -> None:
        if not vectors:
            raise ValueError("at least one frozen query vector is required")
        self._vectors = {
            str(text): np.asarray(vector, dtype=np.float32)
            for text, vector in vectors.items()
        }
        dimensions = {int(vector.shape[0]) for vector in self._vectors.values()}
        if len(dimensions) != 1:
            raise ValueError("all frozen query vectors must have one dimension")
        self._dim = dimensions.pop()

    @property
    def dim(self) -> int:
        return self._dim

    def embed_query(self, query: str) -> np.ndarray:
        try:
            return self._vectors[query].copy()
        except KeyError as exc:
            raise KeyError("query was not included in the frozen batch") from exc

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        return np.stack([self.embed_query(query) for query in queries])

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        if chunks:
            raise RuntimeError("compiled replay must not re-embed source chunks")
        return []


def _open_source_database(source_db: str | Path) -> sqlite3.Connection:
    """Open a historical store without creating SQLite WAL/SHM sidecars."""

    try:
        path = Path(source_db).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(f"source database not found: {source_db}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"source database not found: {path}")
    connection = sqlite3.connect(
        f"{path.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    connection.execute("PRAGMA query_only=ON")
    return connection


def _source_replay_rows(source_db: Path) -> list[_SourceReplayRow]:
    """Read exact turn IDs/timestamps without changing historical row order."""

    connection = _open_source_database(source_db)
    try:
        turn_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(turns)").fetchall()
        }
        ordinal_expression = "ordinal" if "ordinal" in turn_columns else "rowid"
        source_expression = "source_id" if "source_id" in turn_columns else "NULL"
        created_expression = (
            "created_at" if "created_at" in turn_columns else "NULL"
        )
        turns = connection.execute(
            f"SELECT {ordinal_expression}, turn_id, role, text, "
            f"{source_expression}, {created_expression} FROM turns "
            f"ORDER BY {ordinal_expression}"
        ).fetchall()
        chunk_rows = connection.execute(
            "SELECT chunk_id, turn_id, text, start_char, end_char, token_count, "
            "embedding, lexical_weights FROM chunks ORDER BY rowid"
        ).fetchall()
    finally:
        connection.close()
    chunks_by_turn: dict[str, list[Chunk]] = {}
    for row in chunk_rows:
        if row[6] is None:
            continue
        lexical_weights = json.loads(row[7]) if row[7] else None
        chunk = Chunk(
            chunk_id=row[0],
            turn_id=row[1],
            text=row[2],
            start_char=int(row[3]),
            end_char=int(row[4]),
            token_count=int(row[5]),
            embedding=np.frombuffer(row[6], dtype=np.float32).tolist(),
            lexical_weights=lexical_weights,
        )
        chunks_by_turn.setdefault(row[1], []).append(chunk)
    return [
        _SourceReplayRow(
            ordinal=int(ordinal),
            turn_id=str(turn_id),
            role=str(role),
            text=str(text),
            source_id=None if source_id is None else str(source_id),
            created_at=None if created_at is None else str(created_at),
            chunks=tuple(chunks_by_turn.get(turn_id, ())),
        )
        for ordinal, turn_id, role, text, source_id, created_at in turns
    ]


def _source_rows(
    source_db: Path,
) -> list[tuple[int, str, str, str | None, list[Chunk]]]:
    """Historical public projection retained for replay scripts and tests."""

    return [
        (
            row.ordinal,
            row.role,
            row.text,
            row.source_id,
            list(row.chunks),
        )
        for row in _source_replay_rows(source_db)
    ]


def _validated_replay_timestamps(
    rows: Sequence[_SourceReplayRow],
) -> tuple[datetime | None, ...]:
    """Parse all source timestamps before a replay target can be created.

    The tolerant historical ``_source_rows`` projection remains available for
    inspecting old databases.  A revision-4 causal replay, however, cannot
    certify chronology after silently replacing an invalid source timestamp.
    """

    parsed: list[datetime | None] = []
    for row in rows:
        if row.created_at is None:
            parsed.append(None)
            continue
        try:
            parsed.append(datetime.fromisoformat(row.created_at))
        except ValueError as exc:
            raise ValueError(
                "causal replay source turn has an invalid created_at: "
                f"{row.turn_id}"
            ) from exc
    return tuple(parsed)


def _source_user_queries(source_db: str | Path) -> list[str]:
    """Read only user text, avoiding a second materialization of embeddings."""

    connection = _open_source_database(source_db)
    try:
        turn_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(turns)").fetchall()
        }
        ordinal_expression = "ordinal" if "ordinal" in turn_columns else "rowid"
        return [
            str(row[0])
            for row in connection.execute(
                "SELECT text FROM turns WHERE role = 'user' "
                f"ORDER BY {ordinal_expression}"
            ).fetchall()
            if str(row[0]).strip()
        ]
    finally:
        connection.close()


def stage_causal_store(
    source_db: str | Path,
    target_dir: str | Path,
    embedder: object,
    *,
    expansion_tokens: int = 1600,
    retrieval_k: int = 10,
    max_event_nodes: int = 9,
    new_event_nodes: int = 5,
    max_prompt_tokens: int = 128,
    retrieval_access_events: MutableSequence[RetrievalAccessEvent] | None = None,
    retrieval_access_capture_sink: RetrievalAccessCaptureSink | None = None,
    retrieval_access_capture_policy_sha256: str | None = None,
) -> tuple[list[ReplayEvent], dict[str, int]]:
    """Reveal a corpus and bind each response episode to its retrieved context.

    A user prompt first retrieves only earlier chunks. The prompt and following
    assistant/system turns are then ingested. At the next user boundary, the
    event joins the stored prompt and prior anchors with every newly experienced
    response/tool chunk through as many fixed-size slices as necessary. This is
    the fast episodic-binding half of consolidation; later prompts provide
    repeated support. When ``retrieval_access_events`` is supplied, the exact
    ordered direct-result IDs are captured before the current prompt is
    appended. Those receipts contain no query text or token state. Nothing from
    a future episode can enter an earlier event, and slice count changes compute
    rather than transformer workspace memory. A capture sink additionally
    receives a transiently authorized, deterministic projection of those exact
    direct packs. Only that projection is accepted by the history sealer.
    """

    if max_event_nodes < 2:
        raise ValueError("max_event_nodes must be at least two")
    if not 1 <= new_event_nodes < max_event_nodes:
        raise ValueError("new_event_nodes must be in [1, max_event_nodes)")
    source = Path(source_db)
    target = Path(target_dir)
    if target.exists():
        raise FileExistsError(f"refusing to replace replay store: {target}")
    if retrieval_access_capture_sink is not None:
        if type(retrieval_access_capture_sink) is not RetrievalAccessCaptureSink:
            raise TypeError(
                "retrieval_access_capture_sink must be an exact "
                "RetrievalAccessCaptureSink"
            )
        if retrieval_access_capture_sink._capture is not None:
            raise ValueError("retrieval access capture sink is single-use")
        if retrieval_access_capture_policy_sha256 is None:
            raise ValueError(
                "capture policy digest is required with a retrieval access sink"
            )
        capture_policy_sha256 = _require_capture_digest(
            retrieval_access_capture_policy_sha256,
            "retrieval_access_capture_policy_sha256",
        )
        source_database_sha256_before = _sha256(source)
    else:
        if retrieval_access_capture_policy_sha256 is not None:
            raise ValueError(
                "capture policy digest requires a retrieval access capture sink"
            )
        capture_policy_sha256 = None
        source_database_sha256_before = None
    rows = _source_replay_rows(source)
    replay_timestamps = _validated_replay_timestamps(rows)
    source_chunk_count = sum(len(row.chunks) for row in rows)
    budget = ContextBudget(
        recent_window_tokens=0,
        memory_header_tokens=0,
        expansion_tokens=expansion_tokens,
        max_expansions=retrieval_k,
    )
    events: list[ReplayEvent] = []
    skipped_large = 0
    skipped_empty = 0
    chunks_seen = 0
    pending: dict[str, object] | None = None
    completed_episodes = 0
    outcome_chunks_bound = 0
    captured_access_events: list[RetrievalAccessEvent] = []

    def interleave(old_ids: Sequence[str], new_ids: Sequence[str]) -> tuple[str, ...]:
        members: list[str] = []
        for index in range(max(len(old_ids), len(new_ids))):
            if index < len(old_ids):
                members.append(old_ids[index])
            if index < len(new_ids):
                members.append(new_ids[index])
        return tuple(dict.fromkeys(members))[:max_event_nodes]

    def finalize_pending(now_turn: int) -> None:
        nonlocal completed_episodes, outcome_chunks_bound, skipped_empty, pending
        if pending is None:
            return
        prompt_ids = list(dict.fromkeys(pending["prompt_ids"]))
        prior_ids = list(dict.fromkeys(pending["anchor_ids"]))
        anchors = list(dict.fromkeys([*prompt_ids, *prior_ids]))
        outcome_ids = list(dict.fromkeys(pending["outcome_ids"]))
        if not anchors or not outcome_ids:
            skipped_empty += 1
            pending = None
            return
        completed_episodes += 1
        outcome_chunks_bound += len(outcome_ids)
        for part, offset in enumerate(range(0, len(outcome_ids), new_event_nodes)):
            causal_ids = outcome_ids[offset : offset + new_event_nodes]
            anchor_limit = max_event_nodes - len(causal_ids)
            members = interleave(anchors[:anchor_limit], causal_ids)
            causal_members = tuple(
                chunk_id for chunk_id in causal_ids if chunk_id in members
            )
            if len(members) < 2 or not causal_members:
                skipped_empty += 1
                continue
            events.append(
                ReplayEvent(
                    event_id=f'{pending["event_id"]}:part:{part}',
                    now_turn=now_turn,
                    user_text=str(pending["user_text"]),
                    chunk_ids=members,
                    causal_chunk_ids=causal_members,
                )
            )
        pending = None

    with MemoryCondenser(
        data_dir=target,
        embedder=embedder,
        auto_extract=False,
        budget=budget,
        retriever_max_elements=max(1, source_chunk_count),
    ) as condenser:
        for row, created_at in zip(rows, replay_timestamps, strict=True):
            ordinal = row.ordinal
            role = row.role
            text = row.text
            source_id = row.source_id
            source_chunks = row.chunks
            # LongMemEval inserts a dated system turn at the start of each
            # source session. Close the previous episode before that turn is
            # visible; otherwise its timestamp is incorrectly treated as an
            # outcome of the previous conversation.
            source_changed = (
                pending is not None
                and source_id != pending.get("source_id")
            )
            if role == "user" or source_changed:
                finalize_pending(condenser.transcript.current_turn())

            if role == "user" and chunks_seen > 0:
                if count_tokens(text) > max_prompt_tokens:
                    skipped_large += 1
                else:
                    access_event_id = f"causal-user:{ordinal}"
                    access_turn = condenser.transcript.current_turn()
                    packed = condenser.build_context(
                        text,
                        recent_turns=0,
                        k_memories=0,
                        k_expansions=retrieval_k,
                        hybrid=True,
                        reheat_memories=False,
                        use_consolidation=False,
                        learn_consolidation=False,
                        access_event_id=access_event_id,
                    )
                    access_event = RetrievalAccessEvent(
                        event_id=access_event_id,
                        now_turn=access_turn,
                        chunk_ids=tuple(packed.direct_expansion_chunk_ids),
                    )
                    if retrieval_access_events is not None:
                        retrieval_access_events.append(access_event)
                    if retrieval_access_capture_sink is not None:
                        captured_access_events.append(access_event)
                    pending = {
                        "event_id": access_event_id,
                        "user_text": text,
                        "source_id": source_id,
                        "anchor_ids": tuple(packed.direct_expansion_chunk_ids),
                        "prompt_ids": [],
                        "outcome_ids": [],
                    }

            turn = condenser.transcript.append(
                role,
                text,
                source_id=source_id,
                created_at=created_at,
                turn_id=row.turn_id,
            )
            copied = list(source_chunks)
            if any(chunk.turn_id != turn.turn_id for chunk in copied):
                raise RuntimeError(
                    "causal replay source chunk changed its original turn ID"
                )
            condenser.retriever.add_chunks(copied)
            chunks_seen += len(copied)
            if pending is not None:
                member_key = "prompt_ids" if role == "user" else "outcome_ids"
                pending_ids = pending[member_key]
                if not isinstance(pending_ids, list):
                    raise TypeError(f"pending {member_key} must be a list")
                pending_ids.extend(chunk.chunk_id for chunk in copied)

        finalize_pending(condenser.transcript.current_turn())
    if retrieval_access_capture_sink is not None:
        source_database_sha256_after = _sha256(source)
        if source_database_sha256_before != source_database_sha256_after:
            raise RuntimeError(
                "source database changed while retrieval access was staged"
            )
        if capture_policy_sha256 is None:
            raise AssertionError("validated capture policy digest was lost")
        retrieval_access_capture_sink._capture = _mint_retrieval_access_capture(
            source_database_sha256=source_database_sha256_after,
            capture_policy_sha256=capture_policy_sha256,
            retrieval_k=retrieval_k,
            expansion_tokens=expansion_tokens,
            max_prompt_tokens=max_prompt_tokens,
            events=captured_access_events,
        )

    return events, {
        "source_turns": len(rows),
        "events": len(events),
        "completed_episodes": completed_episodes,
        "outcome_chunks_bound": outcome_chunks_bound,
        "skipped_large_prompt": skipped_large,
        "skipped_insufficient_candidates": skipped_empty,
    }


def apply_rank_learning(
    store_dir: str | Path,
    embedder: object,
    events: Sequence[ReplayEvent],
) -> dict[str, object]:
    """Persist provider-free causal/co-access scalars for a staged store."""

    applied = 0
    started = time.perf_counter()
    with MemoryCondenser(
        data_dir=store_dir,
        embedder=embedder,
        auto_extract=False,
        persist_index_on_close=False,
    ) as condenser:
        for event in events:
            update = condenser.observe_context_access(
                [],
                event.chunk_ids,
                access_event_id=event.event_id,
                now_turn=event.now_turn,
                causal_chunk_ids=event.causal_chunk_ids,
            )
            applied += int(update.created)
        graph = condenser.consolidation.stats()
    return {
        "events_offered": len(events),
        "events_applied": applied,
        "elapsed_s": time.perf_counter() - started,
        "graph": graph,
    }


def _copy_store(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to replace arm store: {destination}")
    shutil.copytree(source, destination)


def apply_matched_learning(
    rank_store: str | Path,
    qwen_store: str | Path,
    embedder: FrozenQueryEmbedder,
    events: Sequence[ReplayEvent],
    linker,
) -> dict[str, object]:
    """Apply only events that the bounded Qwen workspace successfully inspects."""

    applied = 0
    failed: list[dict[str, object]] = []
    qwen_elapsed = 0.0
    with (
        MemoryCondenser(
            data_dir=rank_store,
            embedder=embedder,
            auto_extract=False,
            persist_index_on_close=False,
        ) as rank_condenser,
        MemoryCondenser(
            data_dir=qwen_store,
            embedder=embedder,
            auto_extract=False,
            persist_index_on_close=False,
        ) as qwen_condenser,
    ):
        for event in events:
            packed = PackedContext(
                direct_expansion_chunk_ids=list(event.chunk_ids),
                consolidation_event_id=event.event_id,
            )
            started = time.perf_counter()
            try:
                _result, qwen_update = qwen_condenser.consolidate_context_with_qwen(
                    event.user_text,
                    packed,
                    linker,
                    access_event_id=event.event_id,
                    now_turn=event.now_turn,
                    causal_chunk_ids=event.causal_chunk_ids,
                )
            except (MemoryError, RuntimeError, ValueError) as exc:
                failed.append(
                    {
                        "event_id": event.event_id,
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            qwen_elapsed += time.perf_counter() - started
            rank_update = rank_condenser.observe_context_access(
                [],
                event.chunk_ids,
                access_event_id=event.event_id,
                now_turn=event.now_turn,
                causal_chunk_ids=event.causal_chunk_ids,
            )
            if qwen_update.created != rank_update.created:
                raise RuntimeError("rank and Qwen event idempotency diverged")
            applied += int(qwen_update.created)
        rank_stats = rank_condenser.consolidation.stats()
        qwen_stats = qwen_condenser.consolidation.stats()
    return {
        "events_offered": len(events),
        "events_applied_to_both": applied,
        "events_failed_qwen": failed,
        "qwen_update_elapsed_s": qwen_elapsed,
        "qwen_mean_event_s": qwen_elapsed / applied if applied else 0.0,
        "rank_graph": rank_stats,
        "qwen_graph": qwen_stats,
    }


def evaluate_arm(
    store_dir: str | Path,
    embedder: FrozenQueryEmbedder,
    probes: Sequence[tuple[str, str, str, int]],
    *,
    use_consolidation: bool,
    expansion_tokens: int = 1600,
    retrieval_k: int = 10,
    budget_aware_packing: bool = False,
    consolidation_read_slots: int = 1,
    consolidation_hops: int = 1,
    consolidation_candidates: int = 32,
    consolidation_diffusion_width: int = 32,
) -> dict[str, object]:
    """Measure literal reachability and prompt cost without teaching on probes."""

    budget = ContextBudget(
        recent_window_tokens=0,
        memory_header_tokens=0,
        expansion_tokens=expansion_tokens,
        max_expansions=retrieval_k,
        max_consolidation_expansions=consolidation_read_slots,
        budget_aware_expansions=budget_aware_packing,
    )
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    with MemoryCondenser(
        data_dir=store_dir,
        embedder=embedder,
        auto_extract=False,
        budget=budget,
        persist_index_on_close=False,
    ) as condenser:
        for index, (question, answer, category, _source_count) in enumerate(probes):
            packed = condenser.build_context(
                question,
                recent_turns=0,
                k_memories=0,
                k_expansions=retrieval_k,
                hybrid=True,
                reheat_memories=False,
                use_consolidation=use_consolidation,
                learn_consolidation=False,
                consolidation_memory_slots=0,
                consolidation_chunk_slots=consolidation_read_slots,
                consolidation_min_count=2,
                consolidation_hops=consolidation_hops,
                consolidation_candidates=consolidation_candidates,
                consolidation_diffusion_width=consolidation_diffusion_width,
            )
            direct = set(packed.direct_expansion_chunk_ids)
            rows.append(
                {
                    "question_id": f"q{index}",
                    "category": category,
                    "hit": contains_answer(packed.expansions, answer),
                    "context_tokens": packed.token_counts.get("expansions", 0),
                    "chunk_ids": packed.expansion_chunk_ids,
                    "consolidation_chunk_ids": [
                        chunk_id
                        for chunk_id in packed.expansion_chunk_ids
                        if chunk_id not in direct
                    ],
                }
            )
        graph_stats = condenser.consolidation.stats()
    hits = sum(bool(row["hit"]) for row in rows)
    token_values = [int(row["context_tokens"]) for row in rows]
    return {
        "questions": len(rows),
        "literal_recall": hits / len(rows) if rows else 0.0,
        "mean_context_tokens": sum(token_values) / len(rows) if rows else 0.0,
        "max_context_tokens": max(token_values, default=0),
        "questions_with_consolidation_read": sum(
            bool(row["consolidation_chunk_ids"]) for row in rows
        ),
        "elapsed_s": time.perf_counter() - started,
        "graph": graph_stats,
        "rows": rows,
    }


def _comparison(treatment: dict, baseline: dict) -> dict[str, object]:
    treatment_rows = {row["question_id"]: row for row in treatment["rows"]}
    baseline_rows = {row["question_id"]: row for row in baseline["rows"]}
    gained = [
        question_id
        for question_id in baseline_rows
        if treatment_rows[question_id]["hit"] and not baseline_rows[question_id]["hit"]
    ]
    lost = [
        question_id
        for question_id in baseline_rows
        if baseline_rows[question_id]["hit"] and not treatment_rows[question_id]["hit"]
    ]
    return {
        "literal_recall_delta": treatment["literal_recall"]
        - baseline["literal_recall"],
        "mean_context_token_delta": treatment["mean_context_tokens"]
        - baseline["mean_context_tokens"],
        "gained_question_ids": gained,
        "lost_question_ids": lost,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-store", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/models/Qwen3-8B"))
    parser.add_argument(
        "--cav-report",
        type=Path,
        default=Path("eval_results/qwen3_prefix_cav_probe.json"),
    )
    parser.add_argument(
        "--cav-vectors",
        type=Path,
        default=Path("eval_results/qwen3_prefix_cav_probe.safetensors"),
    )
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--qwen-device", default="cuda")
    parser.add_argument("--expansion-tokens", type=int, default=1600)
    parser.add_argument("--retrieval-k", type=int, default=10)
    parser.add_argument("--max-event-nodes", type=int, default=9)
    parser.add_argument("--new-event-nodes", type=int, default=5)
    parser.add_argument("--qwen-group-candidates", type=int, default=3)
    parser.add_argument("--max-prompt-tokens", type=int, default=128)
    parser.add_argument("--max-workspace-tokens", type=int, default=1024)
    parser.add_argument("--consolidation-read-slots", type=int, default=3)
    parser.add_argument("--consolidation-hops", type=int, default=2)
    parser.add_argument("--consolidation-candidates", type=int, default=128)
    parser.add_argument("--consolidation-diffusion-width", type=int, default=32)
    parser.add_argument(
        "--budget-aware-packing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_db = args.source_store / "memory.db"
    if not source_db.is_file():
        raise FileNotFoundError(f"compiled source database not found: {source_db}")
    probes = [tuple(item) for item in json.loads(args.probe.read_text(encoding="utf-8"))]
    source_rows = _source_rows(source_db)
    training_prompts = [
        text
        for _ordinal, role, text, _source_id, _chunks in source_rows
        if role == "user" and count_tokens(text) <= args.max_prompt_tokens
    ]
    query_texts = list(dict.fromkeys([*training_prompts, *(row[0] for row in probes)]))

    embedding_started = time.perf_counter()
    live_embedder = EmbeddingService(device=args.embedding_device)
    vectors = live_embedder.embed_queries(query_texts)
    frozen_embedder = FrozenQueryEmbedder(dict(zip(query_texts, vectors, strict=True)))
    live_embedder.close()
    embedding_elapsed = time.perf_counter() - embedding_started

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_root / f"consolidation-replay-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    base_store = run_dir / "no-consolidation.store"
    rank_store = run_dir / "rank-consolidation.store"
    qwen_store = run_dir / "qwen-consolidation.store"

    staging_started = time.perf_counter()
    events, staging = stage_causal_store(
        source_db,
        base_store,
        frozen_embedder,
        expansion_tokens=args.expansion_tokens,
        retrieval_k=args.retrieval_k,
        max_event_nodes=args.max_event_nodes,
        new_event_nodes=args.new_event_nodes,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    staging_elapsed = time.perf_counter() - staging_started
    _copy_store(base_store, rank_store)
    _copy_store(base_store, qwen_store)

    linker_started = time.perf_counter()
    linker = load_qwen_linker(
        args.model_dir,
        prefix_layers=7,
        attention_layer=1,
        cav_report=args.cav_report,
        cav_vectors=args.cav_vectors,
        cav_layer=5,
        device=args.qwen_device,
        dtype="bfloat16",
        max_candidates=args.qwen_group_candidates,
        max_workspace_tokens=args.max_workspace_tokens,
    )
    linker_load_elapsed = time.perf_counter() - linker_started
    learning = apply_matched_learning(
        rank_store,
        qwen_store,
        frozen_embedder,
        events,
        linker,
    )

    arms = {
        "no_consolidation": evaluate_arm(
            base_store,
            frozen_embedder,
            probes,
            use_consolidation=False,
            expansion_tokens=args.expansion_tokens,
            retrieval_k=args.retrieval_k,
        ),
        "budget_aware_no_consolidation": evaluate_arm(
            base_store,
            frozen_embedder,
            probes,
            use_consolidation=False,
            expansion_tokens=args.expansion_tokens,
            retrieval_k=args.retrieval_k,
            budget_aware_packing=args.budget_aware_packing,
            consolidation_read_slots=args.consolidation_read_slots,
        ),
        "rank_consolidation": evaluate_arm(
            rank_store,
            frozen_embedder,
            probes,
            use_consolidation=True,
            expansion_tokens=args.expansion_tokens,
            retrieval_k=args.retrieval_k,
            budget_aware_packing=args.budget_aware_packing,
            consolidation_read_slots=args.consolidation_read_slots,
            consolidation_hops=args.consolidation_hops,
            consolidation_candidates=args.consolidation_candidates,
            consolidation_diffusion_width=args.consolidation_diffusion_width,
        ),
        "qwen_consolidation": evaluate_arm(
            qwen_store,
            frozen_embedder,
            probes,
            use_consolidation=True,
            expansion_tokens=args.expansion_tokens,
            retrieval_k=args.retrieval_k,
            budget_aware_packing=args.budget_aware_packing,
            consolidation_read_slots=args.consolidation_read_slots,
            consolidation_hops=args.consolidation_hops,
            consolidation_candidates=args.consolidation_candidates,
            consolidation_diffusion_width=args.consolidation_diffusion_width,
        ),
    }
    report = {
        "format": "memory-condense-chronological-consolidation-replay-v2",
        "status": "development_literal_recall_not_answer_accuracy",
        "source_store": str(args.source_store),
        "source_db_sha256": _sha256(source_db),
        "probe_sha256": _sha256(args.probe),
        "config": {
            "expansion_tokens": args.expansion_tokens,
            "retrieval_k": args.retrieval_k,
            "max_event_nodes": args.max_event_nodes,
            "new_event_nodes": args.new_event_nodes,
            "qwen_group_candidates": args.qwen_group_candidates,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_workspace_tokens": args.max_workspace_tokens,
            "consolidation_read_slots": args.consolidation_read_slots,
            "consolidation_hops": args.consolidation_hops,
            "consolidation_candidates": args.consolidation_candidates,
            "consolidation_diffusion_width": args.consolidation_diffusion_width,
            "consolidation_min_count": 2,
            "causal_min_count": 1,
            "budget_aware_packing": args.budget_aware_packing,
        },
        "timing": {
            "query_embedding_s": embedding_elapsed,
            "causal_staging_s": staging_elapsed,
            "qwen_linker_load_s": linker_load_elapsed,
        },
        "staging": staging,
        "learning": learning,
        "arms": arms,
        "comparisons": {
            "budget_vs_none": _comparison(
                arms["budget_aware_no_consolidation"],
                arms["no_consolidation"],
            ),
            "rank_vs_none": _comparison(arms["rank_consolidation"], arms["no_consolidation"]),
            "qwen_vs_none": _comparison(arms["qwen_consolidation"], arms["no_consolidation"]),
            "rank_vs_budget": _comparison(
                arms["rank_consolidation"],
                arms["budget_aware_no_consolidation"],
            ),
            "qwen_vs_budget": _comparison(
                arms["qwen_consolidation"],
                arms["budget_aware_no_consolidation"],
            ),
            "qwen_vs_rank": _comparison(arms["qwen_consolidation"], arms["rank_consolidation"]),
        },
        "retained_prompt_state_bytes": 0,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), **report["comparisons"]}, indent=2))
    for name, arm in arms.items():
        print(
            f"{name}: recall={arm['literal_recall']:.3f}, "
            f"mean_tokens={arm['mean_context_tokens']:.1f}, "
            f"graph_reads={arm['questions_with_consolidation_read']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
