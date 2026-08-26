"""Sealed, text-free history receipts for Hebbian co-retrieval learning.

The artifact in this module records only stable event and chunk identifiers.
It deliberately has no field for a query, prompt, answer, or transcript text.
Callers may optionally re-open the bound source database as an immutable
SQLite snapshot to prove that every retrieved chunk was available at the
event's causal turn.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain._discourse_identity import identity_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval.consolidation_replay import (
    RetrievalAccessCapture,
    RetrievalAccessCaptureValidationError,
    RetrievalAccessEvent,
    retrieval_access_capture_sha256,
    verify_retrieval_access_capture,
)
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
)


HISTORY_ARTIFACT_FORMAT = "memory-condense.hebbian-history-artifact.v1"
HISTORY_RECEIPT_FORMAT = "memory-condense.hebbian-history-receipt.v1"
EVENT_POPULATION_FORMAT = "memory-condense.hebbian-history-population.v1"
CAPTURE_POLICY_FORMAT = "memory-condense.hebbian-capture-policy.v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_CAPTURE_POLICY_FIELDS = frozenset(
    {
        "format",
        "retrieval_k",
        "expansion_tokens",
        "max_prompt_tokens",
        "direct_expansion_only",
        "event_id_scheme",
        "capture_point",
        "exclude_current_and_future_turns",
        "query_embedding_model_id",
        "query_embedding_model_revision",
        "query_embedding_checkpoint_sha256",
        "query_embedding_execution_sha256",
    }
)
_CAPTURE_POINT = "after_direct_context_pack_before_current_user_append"
_ARTIFACT_PAYLOAD_FIELDS = frozenset(
    {
        "format",
        "capture_policy_payload",
        "events",
        "receipt",
        "artifact_sha256",
    }
)
_EVENT_PAYLOAD_FIELDS = frozenset(
    {"event_id", "now_turn", "chunk_ids", "event_sha256"}
)
_RECEIPT_PAYLOAD_FIELDS = frozenset(
    {
        "format",
        "source_database_sha256",
        "source_store_receipt_sha256",
        "implementation_sha256",
        "environment_lock_sha256",
        "capture_policy_sha256",
        "direct_capture_sha256",
        "ordered_event_sha256s",
        "event_population_sha256",
        "event_count",
        "empty_event_count",
        "retained_request_token_state_bytes",
        "receipt_sha256",
    }
)


class HebbianHistoryValidationError(ValueError):
    """Raised when a Hebbian history artifact fails closed validation."""


@dataclass(frozen=True, slots=True)
class HebbianHistoryReceipt:
    """Content and provenance seal for one ordered retrieval history."""

    format: str
    source_database_sha256: str
    source_store_receipt_sha256: str
    implementation_sha256: str
    environment_lock_sha256: str
    capture_policy_sha256: str
    direct_capture_sha256: str
    ordered_event_sha256s: tuple[str, ...]
    event_population_sha256: str
    event_count: int
    empty_event_count: int
    retained_request_token_state_bytes: int
    receipt_sha256: str

    def payload(self) -> dict[str, object]:
        """Return a detached JSON-compatible representation of the receipt."""

        return _receipt_payload(self, include_seal=True)


@dataclass(frozen=True, slots=True)
class HebbianHistoryArtifact:
    """Immutable, text-free events plus their provenance receipt."""

    format: str
    capture_policy_payload: Mapping[str, object]
    events: tuple[RetrievalAccessEvent, ...]
    receipt: HebbianHistoryReceipt
    artifact_sha256: str

    def payload(self) -> dict[str, object]:
        """Return a detached JSON-compatible representation of the artifact."""

        return _artifact_payload(self, include_seal=True)


def _require_digest(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise HebbianHistoryValidationError(
            f"{label} must be an exact lowercase SHA-256 digest"
        )
    return value


def _require_exact_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise HebbianHistoryValidationError(
            f"{label} must be an exact non-negative integer"
        )
    return value


def _validated_capture_policy(value: object) -> dict[str, object]:
    """Accept only the bounded, text-free policy used by causal replay."""

    if type(value) is not dict or set(value) != _CAPTURE_POLICY_FIELDS:
        raise HebbianHistoryValidationError(
            "capture_policy_payload has a noncanonical shape"
        )
    policy = dict(value)
    if policy["format"] != CAPTURE_POLICY_FORMAT:
        raise HebbianHistoryValidationError("unsupported Hebbian capture policy")
    for name, maximum in (
        ("retrieval_k", 64),
        ("expansion_tokens", 8_000),
        ("max_prompt_tokens", 512),
    ):
        amount = _require_exact_int(policy[name], f"capture policy {name}")
        if not 1 <= amount <= maximum:
            raise HebbianHistoryValidationError(
                f"capture policy {name} must lie in [1, {maximum}]"
            )
    required_values = {
        "direct_expansion_only": True,
        "event_id_scheme": "causal-user:{ordinal}",
        "capture_point": _CAPTURE_POINT,
        "exclude_current_and_future_turns": True,
        "query_embedding_model_id": DEFAULT_MODEL_NAME,
        "query_embedding_model_revision": DEFAULT_MODEL_REVISION,
        "query_embedding_checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
    }
    for name, expected in required_values.items():
        if type(policy[name]) is not type(expected) or policy[name] != expected:
            raise HebbianHistoryValidationError(
                f"capture policy {name} changed from its locked value"
            )
    _require_digest(
        policy["query_embedding_execution_sha256"],
        "capture policy query_embedding_execution_sha256",
    )
    return policy


def _freeze_capture_policy(value: object) -> Mapping[str, object]:
    return MappingProxyType(_validated_capture_policy(value))


def _plain_capture_policy(value: object) -> dict[str, object]:
    if type(value) is not _MAPPING_PROXY_TYPE:
        raise HebbianHistoryValidationError(
            "capture_policy_payload must be an immutable exact JSON object"
        )
    return _validated_capture_policy(dict(value))


def _event_payload(event: RetrievalAccessEvent) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "now_turn": event.now_turn,
        "chunk_ids": list(event.chunk_ids),
        "event_sha256": event.event_sha256,
    }


def _validated_events(value: object) -> tuple[RetrievalAccessEvent, ...]:
    if type(value) is not tuple:
        raise HebbianHistoryValidationError("events must be an exact tuple")

    events: tuple[RetrievalAccessEvent, ...] = value
    seen_event_ids: set[str] = set()
    prior_turn = -1
    for index, event in enumerate(events):
        label = f"events[{index}]"
        if type(event) is not RetrievalAccessEvent:
            raise HebbianHistoryValidationError(
                f"{label} must be an exact RetrievalAccessEvent"
            )
        if (
            type(event.event_id) is not str
            or not event.event_id
            or event.event_id != event.event_id.strip()
        ):
            raise HebbianHistoryValidationError(
                f"{label}.event_id must be an exact non-empty string"
            )
        now_turn = _require_exact_int(event.now_turn, f"{label}.now_turn")
        if now_turn < prior_turn:
            raise HebbianHistoryValidationError(
                "event now_turn values must be nondecreasing"
            )
        prior_turn = now_turn
        if event.event_id in seen_event_ids:
            raise HebbianHistoryValidationError("event IDs must be unique")
        seen_event_ids.add(event.event_id)

        if type(event.chunk_ids) is not tuple:
            raise HebbianHistoryValidationError(
                f"{label}.chunk_ids must be an exact tuple"
            )
        seen_chunk_ids: set[str] = set()
        for chunk_index, chunk_id in enumerate(event.chunk_ids):
            if (
                type(chunk_id) is not str
                or not chunk_id
                or chunk_id != chunk_id.strip()
            ):
                raise HebbianHistoryValidationError(
                    f"{label}.chunk_ids[{chunk_index}] must be an exact "
                    "non-empty string"
                )
            if chunk_id in seen_chunk_ids:
                raise HebbianHistoryValidationError(
                    f"{label}.chunk_ids must be unique within the event"
                )
            seen_chunk_ids.add(chunk_id)

        event_sha256 = _require_digest(
            event.event_sha256,
            f"{label}.event_sha256",
        )
        if identity_sha256(event.identity_payload()) != event_sha256:
            raise HebbianHistoryValidationError(
                f"{label}.event_sha256 does not match its ordered ID payload"
            )
    return events


def _population_sha256(events: Sequence[RetrievalAccessEvent]) -> str:
    return identity_sha256(
        {
            "format": EVENT_POPULATION_FORMAT,
            "events": [
                {
                    "event_id": event.event_id,
                    "now_turn": event.now_turn,
                    "event_sha256": event.event_sha256,
                }
                for event in events
            ],
        }
    )


def _receipt_payload(
    receipt: HebbianHistoryReceipt,
    *,
    include_seal: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "format": receipt.format,
        "source_database_sha256": receipt.source_database_sha256,
        "source_store_receipt_sha256": receipt.source_store_receipt_sha256,
        "implementation_sha256": receipt.implementation_sha256,
        "environment_lock_sha256": receipt.environment_lock_sha256,
        "capture_policy_sha256": receipt.capture_policy_sha256,
        "direct_capture_sha256": receipt.direct_capture_sha256,
        "ordered_event_sha256s": list(receipt.ordered_event_sha256s),
        "event_population_sha256": receipt.event_population_sha256,
        "event_count": receipt.event_count,
        "empty_event_count": receipt.empty_event_count,
        "retained_request_token_state_bytes": (
            receipt.retained_request_token_state_bytes
        ),
    }
    if include_seal:
        payload["receipt_sha256"] = receipt.receipt_sha256
    return payload


def _artifact_payload(
    artifact: HebbianHistoryArtifact,
    *,
    include_seal: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "format": artifact.format,
        "capture_policy_payload": _plain_capture_policy(
            artifact.capture_policy_payload
        ),
        "events": [_event_payload(event) for event in artifact.events],
        "receipt": _receipt_payload(artifact.receipt, include_seal=True),
    }
    if include_seal:
        payload["artifact_sha256"] = artifact.artifact_sha256
    return payload


def _validated_receipt(
    receipt: object,
    *,
    events: tuple[RetrievalAccessEvent, ...],
    capture_policy: Mapping[str, object],
) -> HebbianHistoryReceipt:
    if type(receipt) is not HebbianHistoryReceipt:
        raise HebbianHistoryValidationError(
            "receipt must be an exact HebbianHistoryReceipt"
        )
    if type(receipt.format) is not str or receipt.format != HISTORY_RECEIPT_FORMAT:
        raise HebbianHistoryValidationError("unsupported Hebbian history receipt")
    for field_name in (
        "source_database_sha256",
        "source_store_receipt_sha256",
        "implementation_sha256",
        "environment_lock_sha256",
        "capture_policy_sha256",
        "direct_capture_sha256",
        "event_population_sha256",
        "receipt_sha256",
    ):
        _require_digest(getattr(receipt, field_name), f"receipt.{field_name}")
    capture_policy_sha256 = identity_sha256(dict(capture_policy))
    if receipt.capture_policy_sha256 != capture_policy_sha256:
        raise HebbianHistoryValidationError(
            "receipt does not bind the exact capture policy payload"
        )
    expected_direct_capture_sha256 = retrieval_access_capture_sha256(
        source_database_sha256=receipt.source_database_sha256,
        capture_policy_sha256=capture_policy_sha256,
        retrieval_k=int(capture_policy["retrieval_k"]),
        expansion_tokens=int(capture_policy["expansion_tokens"]),
        max_prompt_tokens=int(capture_policy["max_prompt_tokens"]),
        events=events,
    )
    if receipt.direct_capture_sha256 != expected_direct_capture_sha256:
        raise HebbianHistoryValidationError(
            "receipt does not bind the staging-issued direct capture"
        )

    if type(receipt.ordered_event_sha256s) is not tuple:
        raise HebbianHistoryValidationError(
            "receipt.ordered_event_sha256s must be an exact tuple"
        )
    for index, digest in enumerate(receipt.ordered_event_sha256s):
        _require_digest(digest, f"receipt.ordered_event_sha256s[{index}]")
    expected_order = tuple(event.event_sha256 for event in events)
    if receipt.ordered_event_sha256s != expected_order:
        raise HebbianHistoryValidationError(
            "receipt does not bind the ordered event seals"
        )

    event_count = _require_exact_int(receipt.event_count, "receipt.event_count")
    empty_count = _require_exact_int(
        receipt.empty_event_count,
        "receipt.empty_event_count",
    )
    retained = _require_exact_int(
        receipt.retained_request_token_state_bytes,
        "receipt.retained_request_token_state_bytes",
    )
    if event_count != len(events):
        raise HebbianHistoryValidationError("receipt event_count mismatch")
    if empty_count != sum(not event.chunk_ids for event in events):
        raise HebbianHistoryValidationError("receipt empty_event_count mismatch")
    if retained != 0:
        raise HebbianHistoryValidationError(
            "Hebbian history may not retain request token state"
        )
    if receipt.event_population_sha256 != _population_sha256(events):
        raise HebbianHistoryValidationError("event population seal mismatch")
    if receipt.receipt_sha256 != identity_sha256(
        _receipt_payload(receipt, include_seal=False)
    ):
        raise HebbianHistoryValidationError("Hebbian history receipt seal mismatch")
    return receipt


def _resolved_database_path(value: str | Path) -> Path:
    if type(value) is not str and not isinstance(value, Path):
        raise HebbianHistoryValidationError(
            "source_database_path must be an exact string or Path"
        )
    try:
        path = Path(value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HebbianHistoryValidationError(
            "source database does not exist"
        ) from exc
    if not path.is_file():
        raise HebbianHistoryValidationError("source database must be a file")
    return path


def _require_no_sqlite_sidecars(database_path: Path) -> None:
    if any(
        database_path.with_name(database_path.name + suffix).exists()
        for suffix in ("-wal", "-shm")
    ):
        raise HebbianHistoryValidationError(
            "source database retained SQLite WAL/SHM sidecars"
        )


def _verify_causal_history(
    database_path: Path,
    events: Sequence[RetrievalAccessEvent],
    capture_policy: Mapping[str, object],
) -> None:
    expected_ids = {
        chunk_id for event in events for chunk_id in event.chunk_ids
    }
    found: dict[str, int] = {}
    target = f"{database_path.as_uri()}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(target, uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            turn_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(turns)")
            }
            chunk_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(chunks)")
            }
            if not {"ordinal", "role", "text"}.issubset(turn_columns) or not {
                "chunk_id",
                "turn_id",
            }.issubset(chunk_columns):
                raise HebbianHistoryValidationError(
                    "source database lacks causal turn/chunk coordinates"
                )
            turn_rows = connection.execute(
                "SELECT ordinal, role, text FROM turns ORDER BY ordinal"
            ).fetchall()
            ordinals = [row[0] for row in turn_rows]
            if (
                not turn_rows
                or any(type(value) is not int for value in ordinals)
                or ordinals != list(range(1, len(turn_rows) + 1))
            ):
                raise HebbianHistoryValidationError(
                    "source turn ordinals must be contiguous and one-based"
                )
            embedded_only = " AND chunks.embedding IS NOT NULL" if (
                "embedding" in chunk_columns
            ) else ""
            chunk_counts = {
                int(row[0]): int(row[1])
                for row in connection.execute(
                    "SELECT turns.ordinal, COUNT(chunks.chunk_id) FROM turns "
                    "LEFT JOIN chunks ON chunks.turn_id = turns.turn_id"
                    + embedded_only
                    + " GROUP BY turns.ordinal"
                )
            }
            max_prompt_tokens = int(capture_policy["max_prompt_tokens"])
            expected_event_coordinates: list[tuple[str, int]] = []
            chunks_seen = 0
            for raw_ordinal, raw_role, raw_text in turn_rows:
                ordinal = int(raw_ordinal)
                role = str(raw_role)
                text = str(raw_text)
                if (
                    role == "user"
                    and chunks_seen > 0
                    and count_tokens(text) <= max_prompt_tokens
                ):
                    expected_event_coordinates.append(
                        (f"causal-user:{ordinal}", ordinal - 1)
                    )
                chunks_seen += chunk_counts.get(ordinal, 0)
            observed_event_coordinates = [
                (event.event_id, event.now_turn) for event in events
            ]
            if observed_event_coordinates != expected_event_coordinates:
                raise HebbianHistoryValidationError(
                    "event population does not match eligible historical user turns"
                )
            retrieval_k = int(capture_policy["retrieval_k"])
            if any(len(event.chunk_ids) > retrieval_k for event in events):
                raise HebbianHistoryValidationError(
                    "event membership exceeds the locked retrieval_k"
                )
            ordered_ids = sorted(expected_ids)
            for start in range(0, len(ordered_ids), 900):
                batch = ordered_ids[start : start + 900]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    "SELECT chunks.chunk_id, turns.ordinal FROM chunks "
                    "JOIN turns ON turns.turn_id = chunks.turn_id "
                    f"WHERE chunks.chunk_id IN ({placeholders})",
                    batch,
                )
                for raw_chunk_id, raw_ordinal in rows:
                    if (
                        type(raw_chunk_id) is not str
                        or type(raw_ordinal) is not int
                        or raw_ordinal < 0
                    ):
                        raise HebbianHistoryValidationError(
                            "source database returned a noncanonical chunk ordinal"
                        )
                    found[raw_chunk_id] = raw_ordinal
        finally:
            connection.close()
    except HebbianHistoryValidationError:
        raise
    except sqlite3.Error as exc:
        raise HebbianHistoryValidationError(
            "source database cannot prove causal chunk ordinals"
        ) from exc

    missing = expected_ids.difference(found)
    if missing:
        raise HebbianHistoryValidationError(
            "source database is missing an event chunk ID"
        )
    for event in events:
        for chunk_id in event.chunk_ids:
            if found[chunk_id] > event.now_turn:
                raise HebbianHistoryValidationError(
                    "event references a chunk from a future turn"
                )


def _verify_database_binding(
    database_path: str | Path,
    events: tuple[RetrievalAccessEvent, ...],
    capture_policy: Mapping[str, object],
    expected_sha256: str | None,
) -> str:
    path = _resolved_database_path(database_path)
    _require_no_sqlite_sidecars(path)
    before = file_sha256(path)
    _verify_causal_history(path, events, capture_policy)
    after = file_sha256(path)
    _require_no_sqlite_sidecars(path)
    if before != after:
        raise HebbianHistoryValidationError(
            "source database changed while its history was being verified"
        )
    if expected_sha256 is not None and before != expected_sha256:
        raise HebbianHistoryValidationError(
            "source database does not match the sealed database digest"
        )
    return before


def seal_hebbian_history_artifact(
    capture: RetrievalAccessCapture,
    *,
    source_database_path: str | Path,
    source_store_receipt_sha256: str,
    implementation_sha256: str,
    environment_lock_sha256: str,
    capture_policy_payload: Mapping[str, object],
) -> HebbianHistoryArtifact:
    """Seal only the exact direct packs certified by causal staging."""

    try:
        validated_capture = verify_retrieval_access_capture(capture)
    except RetrievalAccessCaptureValidationError as exc:
        raise HebbianHistoryValidationError(str(exc)) from exc
    normalized_events = _validated_events(validated_capture.events)
    store_receipt_sha256 = _require_digest(
        source_store_receipt_sha256,
        "source_store_receipt_sha256",
    )
    implementation = _require_digest(
        implementation_sha256,
        "implementation_sha256",
    )
    environment = _require_digest(
        environment_lock_sha256,
        "environment_lock_sha256",
    )
    policy = _freeze_capture_policy(capture_policy_payload)
    policy_sha256 = identity_sha256(_plain_capture_policy(policy))
    if validated_capture.capture_policy_sha256 != policy_sha256:
        raise HebbianHistoryValidationError(
            "staging capture does not bind the exact capture policy"
        )
    for field_name in ("retrieval_k", "expansion_tokens", "max_prompt_tokens"):
        if getattr(validated_capture, field_name) != policy[field_name]:
            raise HebbianHistoryValidationError(
                f"staging capture {field_name} does not match capture policy"
            )
    database_sha256 = _verify_database_binding(
        source_database_path,
        normalized_events,
        policy,
        None,
    )
    if validated_capture.source_database_sha256 != database_sha256:
        raise HebbianHistoryValidationError(
            "staging capture does not bind the exact source database"
        )
    ordered_event_sha256s = tuple(
        event.event_sha256 for event in normalized_events
    )

    unsigned_receipt = HebbianHistoryReceipt(
        format=HISTORY_RECEIPT_FORMAT,
        source_database_sha256=database_sha256,
        source_store_receipt_sha256=store_receipt_sha256,
        implementation_sha256=implementation,
        environment_lock_sha256=environment,
        capture_policy_sha256=policy_sha256,
        direct_capture_sha256=validated_capture.capture_sha256,
        ordered_event_sha256s=ordered_event_sha256s,
        event_population_sha256=_population_sha256(normalized_events),
        event_count=len(normalized_events),
        empty_event_count=sum(not event.chunk_ids for event in normalized_events),
        retained_request_token_state_bytes=0,
        receipt_sha256="0" * 64,
    )
    receipt = HebbianHistoryReceipt(
        format=unsigned_receipt.format,
        source_database_sha256=unsigned_receipt.source_database_sha256,
        source_store_receipt_sha256=unsigned_receipt.source_store_receipt_sha256,
        implementation_sha256=unsigned_receipt.implementation_sha256,
        environment_lock_sha256=unsigned_receipt.environment_lock_sha256,
        capture_policy_sha256=unsigned_receipt.capture_policy_sha256,
        direct_capture_sha256=unsigned_receipt.direct_capture_sha256,
        ordered_event_sha256s=unsigned_receipt.ordered_event_sha256s,
        event_population_sha256=unsigned_receipt.event_population_sha256,
        event_count=unsigned_receipt.event_count,
        empty_event_count=unsigned_receipt.empty_event_count,
        retained_request_token_state_bytes=0,
        receipt_sha256=identity_sha256(
            _receipt_payload(unsigned_receipt, include_seal=False)
        ),
    )
    unsigned_artifact = HebbianHistoryArtifact(
        format=HISTORY_ARTIFACT_FORMAT,
        capture_policy_payload=policy,
        events=normalized_events,
        receipt=receipt,
        artifact_sha256="0" * 64,
    )
    artifact = HebbianHistoryArtifact(
        format=unsigned_artifact.format,
        capture_policy_payload=unsigned_artifact.capture_policy_payload,
        events=unsigned_artifact.events,
        receipt=unsigned_artifact.receipt,
        artifact_sha256=identity_sha256(
            _artifact_payload(unsigned_artifact, include_seal=False)
        ),
    )
    # The external binding and causal ordinals were already checked above.
    # Re-run only the cheap internal seals here; callers can independently
    # re-prove the source database later by passing its path to ``verify``.
    return verify_hebbian_history_artifact(artifact)


def load_hebbian_history_artifact(
    payload: Mapping[str, object],
) -> HebbianHistoryArtifact:
    """Reconstruct and verify an artifact from its exact JSON payload."""

    if type(payload) is not dict or set(payload) != _ARTIFACT_PAYLOAD_FIELDS:
        raise HebbianHistoryValidationError(
            "Hebbian history artifact payload has a noncanonical shape"
        )

    raw_policy = payload["capture_policy_payload"]
    policy = _freeze_capture_policy(raw_policy)

    raw_events = payload["events"]
    if type(raw_events) is not list:
        raise HebbianHistoryValidationError(
            "artifact events must be an exact JSON array"
        )
    events: list[RetrievalAccessEvent] = []
    for index, raw_event in enumerate(raw_events):
        label = f"artifact events[{index}]"
        if type(raw_event) is not dict or set(raw_event) != _EVENT_PAYLOAD_FIELDS:
            raise HebbianHistoryValidationError(f"{label} has a noncanonical shape")
        event_id = raw_event["event_id"]
        if (
            type(event_id) is not str
            or not event_id
            or event_id != event_id.strip()
        ):
            raise HebbianHistoryValidationError(
                f"{label}.event_id must be an exact non-empty string"
            )
        now_turn = _require_exact_int(raw_event["now_turn"], f"{label}.now_turn")
        raw_chunk_ids = raw_event["chunk_ids"]
        if type(raw_chunk_ids) is not list:
            raise HebbianHistoryValidationError(
                f"{label}.chunk_ids must be an exact JSON array"
            )
        chunk_ids: list[str] = []
        for chunk_index, chunk_id in enumerate(raw_chunk_ids):
            if (
                type(chunk_id) is not str
                or not chunk_id
                or chunk_id != chunk_id.strip()
            ):
                raise HebbianHistoryValidationError(
                    f"{label}.chunk_ids[{chunk_index}] must be an exact "
                    "non-empty string"
                )
            chunk_ids.append(chunk_id)
        declared_event_sha256 = _require_digest(
            raw_event["event_sha256"],
            f"{label}.event_sha256",
        )
        event = RetrievalAccessEvent(
            event_id=event_id,
            now_turn=now_turn,
            chunk_ids=tuple(chunk_ids),
        )
        if event.event_sha256 != declared_event_sha256:
            raise HebbianHistoryValidationError(
                f"{label}.event_sha256 does not match its ordered ID payload"
            )
        events.append(event)

    raw_receipt = payload["receipt"]
    if (
        type(raw_receipt) is not dict
        or set(raw_receipt) != _RECEIPT_PAYLOAD_FIELDS
    ):
        raise HebbianHistoryValidationError(
            "Hebbian history receipt payload has a noncanonical shape"
        )
    raw_ordered_seals = raw_receipt["ordered_event_sha256s"]
    if type(raw_ordered_seals) is not list:
        raise HebbianHistoryValidationError(
            "receipt.ordered_event_sha256s must be an exact JSON array"
        )
    receipt = HebbianHistoryReceipt(
        format=raw_receipt["format"],
        source_database_sha256=raw_receipt["source_database_sha256"],
        source_store_receipt_sha256=raw_receipt[
            "source_store_receipt_sha256"
        ],
        implementation_sha256=raw_receipt["implementation_sha256"],
        environment_lock_sha256=raw_receipt["environment_lock_sha256"],
        capture_policy_sha256=raw_receipt["capture_policy_sha256"],
        direct_capture_sha256=raw_receipt["direct_capture_sha256"],
        ordered_event_sha256s=tuple(raw_ordered_seals),
        event_population_sha256=raw_receipt["event_population_sha256"],
        event_count=raw_receipt["event_count"],
        empty_event_count=raw_receipt["empty_event_count"],
        retained_request_token_state_bytes=raw_receipt[
            "retained_request_token_state_bytes"
        ],
        receipt_sha256=raw_receipt["receipt_sha256"],
    )
    artifact = HebbianHistoryArtifact(
        format=payload["format"],
        capture_policy_payload=policy,
        events=tuple(events),
        receipt=receipt,
        artifact_sha256=payload["artifact_sha256"],
    )
    return verify_hebbian_history_artifact(artifact)


def verify_hebbian_history_artifact(
    artifact: object,
    *,
    source_database_path: str | Path | None = None,
) -> HebbianHistoryArtifact:
    """Fail closed on tampering and optionally re-prove source causality."""

    if type(artifact) is not HebbianHistoryArtifact:
        raise HebbianHistoryValidationError(
            "artifact must be an exact HebbianHistoryArtifact"
        )
    if type(artifact.format) is not str or artifact.format != HISTORY_ARTIFACT_FORMAT:
        raise HebbianHistoryValidationError("unsupported Hebbian history artifact")
    events = _validated_events(artifact.events)
    policy = _plain_capture_policy(artifact.capture_policy_payload)
    policy_sha256 = identity_sha256(policy)
    receipt = _validated_receipt(
        artifact.receipt,
        events=events,
        capture_policy=policy,
    )
    artifact_sha256 = _require_digest(
        artifact.artifact_sha256,
        "artifact.artifact_sha256",
    )
    if artifact_sha256 != identity_sha256(
        _artifact_payload(artifact, include_seal=False)
    ):
        raise HebbianHistoryValidationError("Hebbian history artifact seal mismatch")
    if source_database_path is not None:
        _verify_database_binding(
            source_database_path,
            events,
            artifact.capture_policy_payload,
            receipt.source_database_sha256,
        )
    return artifact


__all__ = [
    "EVENT_POPULATION_FORMAT",
    "HISTORY_ARTIFACT_FORMAT",
    "HISTORY_RECEIPT_FORMAT",
    "HebbianHistoryArtifact",
    "HebbianHistoryReceipt",
    "HebbianHistoryValidationError",
    "load_hebbian_history_artifact",
    "seal_hebbian_history_artifact",
    "verify_hebbian_history_artifact",
]
