"""Apply a sealed Hebbian history to an isolated causal replay store.

The source store is never opened writable.  Callers first use
``stage_causal_store`` to create a fresh chronological store, then this module
registers one deterministic co-access namespace and applies every frozen
retrieval event.  Publication retains only scalar graph state and ID receipts;
no query text or transformer-shaped state enters the manifest or graph tables.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from memory_condense.associations.association_models import AssociationArtifact
from memory_condense.associations.association_store import AssociationStore
from memory_condense.associations.coaccess_graph import rank_discount
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval.hebbian_history import (
    HebbianHistoryArtifact,
    verify_hebbian_history_artifact,
)
from memory_condense.persistence.db import Database


DERIVED_STORE_FORMAT = "memory-condense.hebbian-derived-store.v1"
LEARNING_POLICY_FORMAT = "memory-condense.hebbian-learning-policy.v1"
ARTIFACT_MODEL_ID = "memory-condense/hebbian-rank-coaccess-v1"
ARTIFACT_CHECKPOINT_ID = "external-scalar-coaccess-v1"
DETERMINISTIC_CREATED_AT = "1970-01-01T00:00:00+00:00"
MANIFEST_NAME = "hebbian-derived-store.json"

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_STORE_INPUT_FILES = frozenset({"memory.db", "hnsw_index.bin"})
_EMPTY_GRAPH_TABLES = (
    "chunk_cav_signatures",
    "chunk_head_edges",
    "hebbian_access_events",
    "hebbian_chunk_nodes",
    "hebbian_chunk_edges",
    "consolidation_access_events",
    "consolidation_nodes",
    "consolidation_edges",
)
_POLICY_PAYLOAD_FIELDS = frozenset(
    {
        "format",
        "learning_rate",
        "half_life_turns",
        "max_concepts_per_event",
        "max_degree",
        "min_edge_score",
        "retain_all_event_receipts",
    }
)
_RECEIPT_PAYLOAD_FIELDS = frozenset(
    {
        "format",
        "source_database_sha256",
        "source_index_sha256",
        "source_store_receipt_sha256",
        "source_turn_sequence_sha256",
        "source_chunk_sequence_sha256",
        "derived_database_sha256",
        "derived_index_sha256",
        "derived_turn_sequence_sha256",
        "derived_chunk_sequence_sha256",
        "history_artifact_sha256",
        "history_receipt_sha256",
        "implementation_sha256",
        "environment_lock_sha256",
        "learning_policy",
        "learning_policy_sha256",
        "association_artifact_id",
        "association_artifact_sha256",
        "events_offered",
        "events_applied",
        "graph_nodes",
        "graph_edges",
        "graph_event_receipts",
        "retained_request_token_state_bytes",
        "receipt_sha256",
    }
)


class HebbianDerivedStoreError(ValueError):
    """Raised when a derived store cannot prove its isolation or identity."""


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise HebbianDerivedStoreError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise HebbianDerivedStoreError(
            f"{label} must be an exact integer of at least {minimum}"
        )
    return value


def _finite(value: object, label: str, *, minimum: float) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise HebbianDerivedStoreError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise HebbianDerivedStoreError(
            f"{label} must be finite and at least {minimum}"
        )
    return result


@dataclass(frozen=True, slots=True)
class HebbianLearningPolicy:
    """Frozen scalar learning policy for the H1 history graph."""

    format: str = LEARNING_POLICY_FORMAT
    learning_rate: float = 1.0
    half_life_turns: float = 200.0
    max_concepts_per_event: int = 12
    max_degree: int = 32
    min_edge_score: float = 0.0
    retain_all_event_receipts: bool = True

    def __post_init__(self) -> None:
        if type(self.format) is not str or self.format != LEARNING_POLICY_FORMAT:
            raise HebbianDerivedStoreError("unsupported Hebbian learning policy")
        object.__setattr__(
            self,
            "learning_rate",
            _finite(self.learning_rate, "learning_rate", minimum=0.0),
        )
        if self.learning_rate <= 0.0:
            raise HebbianDerivedStoreError("learning_rate must be positive")
        object.__setattr__(
            self,
            "half_life_turns",
            _finite(self.half_life_turns, "half_life_turns", minimum=0.0),
        )
        if self.half_life_turns <= 0.0:
            raise HebbianDerivedStoreError("half_life_turns must be positive")
        _exact_int(
            self.max_concepts_per_event,
            "max_concepts_per_event",
            minimum=1,
        )
        _exact_int(self.max_degree, "max_degree", minimum=0)
        min_edge = _finite(self.min_edge_score, "min_edge_score", minimum=0.0)
        if min_edge > 1.0:
            raise HebbianDerivedStoreError("min_edge_score must not exceed one")
        object.__setattr__(self, "min_edge_score", min_edge)
        if type(self.retain_all_event_receipts) is not bool or not (
            self.retain_all_event_receipts
        ):
            raise HebbianDerivedStoreError(
                "derived-store policy must retain every frozen event receipt"
            )

    def payload(self) -> dict[str, object]:
        return {
            "format": self.format,
            "learning_rate": self.learning_rate,
            "half_life_turns": self.half_life_turns,
            "max_concepts_per_event": self.max_concepts_per_event,
            "max_degree": self.max_degree,
            "min_edge_score": self.min_edge_score,
            "retain_all_event_receipts": self.retain_all_event_receipts,
        }

    @property
    def policy_sha256(self) -> str:
        return identity_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class HebbianDerivedStoreReceipt:
    """Exact provenance and graph counts for one published H1 store."""

    format: str
    source_database_sha256: str
    source_index_sha256: str
    source_store_receipt_sha256: str
    source_turn_sequence_sha256: str
    source_chunk_sequence_sha256: str
    derived_database_sha256: str
    derived_index_sha256: str
    derived_turn_sequence_sha256: str
    derived_chunk_sequence_sha256: str
    history_artifact_sha256: str
    history_receipt_sha256: str
    implementation_sha256: str
    environment_lock_sha256: str
    learning_policy: HebbianLearningPolicy
    learning_policy_sha256: str
    association_artifact_id: str
    association_artifact_sha256: str
    events_offered: int
    events_applied: int
    graph_nodes: int
    graph_edges: int
    graph_event_receipts: int
    retained_request_token_state_bytes: int
    receipt_sha256: str

    def payload(self, *, include_seal: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "format": self.format,
            "source_database_sha256": self.source_database_sha256,
            "source_index_sha256": self.source_index_sha256,
            "source_store_receipt_sha256": self.source_store_receipt_sha256,
            "source_turn_sequence_sha256": self.source_turn_sequence_sha256,
            "source_chunk_sequence_sha256": self.source_chunk_sequence_sha256,
            "derived_database_sha256": self.derived_database_sha256,
            "derived_index_sha256": self.derived_index_sha256,
            "derived_turn_sequence_sha256": self.derived_turn_sequence_sha256,
            "derived_chunk_sequence_sha256": self.derived_chunk_sequence_sha256,
            "history_artifact_sha256": self.history_artifact_sha256,
            "history_receipt_sha256": self.history_receipt_sha256,
            "implementation_sha256": self.implementation_sha256,
            "environment_lock_sha256": self.environment_lock_sha256,
            "learning_policy": self.learning_policy.payload(),
            "learning_policy_sha256": self.learning_policy_sha256,
            "association_artifact_id": self.association_artifact_id,
            "association_artifact_sha256": self.association_artifact_sha256,
            "events_offered": self.events_offered,
            "events_applied": self.events_applied,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "graph_event_receipts": self.graph_event_receipts,
            "retained_request_token_state_bytes": (
                self.retained_request_token_state_bytes
            ),
        }
        if include_seal:
            result["receipt_sha256"] = self.receipt_sha256
        return result


def _resolved_file(value: str | Path, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_symlink():
        raise HebbianDerivedStoreError(f"{label} must not be a symbolic link")
    try:
        path = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HebbianDerivedStoreError(f"{label} does not exist") from exc
    if not path.is_file() or path.is_symlink():
        raise HebbianDerivedStoreError(f"{label} must be a regular file")
    return path


def _resolved_store(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_symlink():
        raise HebbianDerivedStoreError(
            "staged store must not be a symbolic link"
        )
    try:
        path = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HebbianDerivedStoreError("staged store does not exist") from exc
    if not path.is_dir() or path.is_symlink():
        raise HebbianDerivedStoreError("staged store must be a regular directory")
    return path


def _open_immutable(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _lexical_weights_sha256(value: object) -> str | None:
    if value is None:
        return None
    parsed = json.loads(str(value))
    if type(parsed) is not dict:
        raise HebbianDerivedStoreError(
            "chunk lexical_weights must be a JSON object"
        )
    return identity_sha256(
        {str(term): float(weight) for term, weight in parsed.items()}
    )


def _sequence_hashes(database_path: Path) -> tuple[str, str]:
    connection = _open_immutable(database_path)
    try:
        turns = [
            {
                "ordinal": int(row[0]),
                "turn_id": str(row[1]),
                "role": str(row[2]),
                "source_id": None if row[3] is None else str(row[3]),
                "created_at": None if row[4] is None else str(row[4]),
                "text_sha256": quote_sha256(str(row[5])),
            }
            for row in connection.execute(
                "SELECT ordinal, turn_id, role, source_id, created_at, text "
                "FROM turns ORDER BY ordinal"
            )
        ]
        chunks = [
            {
                "chunk_id": str(row[0]),
                "turn_id": str(row[1]),
                "start_char": int(row[2]),
                "end_char": int(row[3]),
                "token_count": int(row[4]),
                "text_sha256": quote_sha256(str(row[5])),
                "embedding_sha256": (
                    None
                    if row[6] is None
                    else hashlib.sha256(bytes(row[6])).hexdigest()
                ),
                "lexical_weights_sha256": _lexical_weights_sha256(row[7]),
                "hnsw_label": None if row[8] is None else int(row[8]),
                "term_count": None if row[9] is None else int(row[9]),
            }
            for row in connection.execute(
                "SELECT chunk_id, turn_id, start_char, end_char, token_count, text, "
                "embedding, lexical_weights, hnsw_label, term_count "
                "FROM chunks ORDER BY rowid"
            )
        ]
        chunk_terms = [
            {
                "chunk_id": str(row[0]),
                "term": str(row[1]),
                "tf": int(row[2]),
            }
            for row in connection.execute(
                "SELECT chunk_id, term, tf FROM chunk_terms "
                "ORDER BY chunk_id, term"
            )
        ]
    finally:
        connection.close()
    return (
        identity_sha256({"format": "memory-condense.turn-sequence.v1", "rows": turns}),
        identity_sha256(
            {
                "format": "memory-condense.chunk-sequence.v1",
                "rows": chunks,
                "chunk_terms": chunk_terms,
            }
        ),
    )


def _graph_counts(database_path: Path) -> dict[str, int]:
    connection = _open_immutable(database_path)
    try:
        counts = {
            "association_artifacts": int(
                connection.execute(
                    "SELECT COUNT(*) FROM association_artifacts"
                ).fetchone()[0]
            )
        }
        for table in _EMPTY_GRAPH_TABLES:
            counts[table] = int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
        return counts
    except sqlite3.Error as exc:
        raise HebbianDerivedStoreError(
            "staged database cannot prove a clean graph namespace"
        ) from exc
    finally:
        connection.close()


def _require_clean_graph_namespace(database_path: Path) -> None:
    contaminated = {
        table: count
        for table, count in _graph_counts(database_path).items()
        if count != 0
    }
    if contaminated:
        raise HebbianDerivedStoreError(
            "staged database contains pre-existing graph state: "
            f"{contaminated!r}"
        )


def _artifact_payload(artifact: AssociationArtifact) -> dict[str, object]:
    return {
        "artifact_id": artifact.artifact_id,
        "model_id": artifact.model_id,
        "checkpoint_id": artifact.checkpoint_id,
        "prefix_layers": artifact.prefix_layers,
        "head_layer": artifact.head_layer,
        "cav_layer": artifact.cav_layer,
        "concept_names": list(artifact.concept_names),
        "head_count": artifact.head_count,
        "created_at": artifact.created_at,
        "metadata": dict(artifact.metadata),
    }


def _association_artifact_from_bindings(
    *,
    history_artifact_sha256: str,
    history_receipt_sha256: str,
    source_store_receipt_sha256: str,
    learning_policy_sha256: str,
) -> AssociationArtifact:
    artifact = AssociationArtifact.create(
        model_id=ARTIFACT_MODEL_ID,
        checkpoint_id=ARTIFACT_CHECKPOINT_ID,
        prefix_layers=1,
        head_layer=0,
        cav_layer=None,
        concept_names=(),
        head_count=1,
        metadata={
            "format": "memory-condense.hebbian-association-namespace.v1",
            "history_artifact_sha256": history_artifact_sha256,
            "history_receipt_sha256": history_receipt_sha256,
            "source_store_receipt_sha256": source_store_receipt_sha256,
            "learning_policy_sha256": learning_policy_sha256,
        },
    )
    return replace(artifact, created_at=DETERMINISTIC_CREATED_AT)


def _association_artifact(
    history: HebbianHistoryArtifact,
    policy: HebbianLearningPolicy,
) -> AssociationArtifact:
    return _association_artifact_from_bindings(
        history_artifact_sha256=history.artifact_sha256,
        history_receipt_sha256=history.receipt.receipt_sha256,
        source_store_receipt_sha256=(
            history.receipt.source_store_receipt_sha256
        ),
        learning_policy_sha256=policy.policy_sha256,
    )


def _remove_checkpoint_sidecars(database_path: Path) -> None:
    wal = database_path.with_name(database_path.name + "-wal")
    shm = database_path.with_name(database_path.name + "-shm")
    if wal.exists() and wal.stat().st_size != 0:
        raise HebbianDerivedStoreError(
            "derived database retained non-empty WAL state after checkpoint"
        )
    for path in (wal, shm):
        if path.exists():
            path.unlink()


def _require_no_sqlite_sidecars(database_path: Path, label: str) -> None:
    if any(
        database_path.with_name(database_path.name + suffix).exists()
        for suffix in ("-wal", "-shm")
    ):
        raise HebbianDerivedStoreError(f"{label} retained SQLite sidecars")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _publish_manifest(path: Path, receipt: HebbianDerivedStoreReceipt) -> None:
    payload = _canonical_bytes(receipt.payload())
    if path.exists():
        raise FileExistsError(f"refusing to replace derived-store manifest: {path}")
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_receipt(receipt: HebbianDerivedStoreReceipt) -> None:
    if type(receipt) is not HebbianDerivedStoreReceipt:
        raise HebbianDerivedStoreError(
            "receipt must be an exact HebbianDerivedStoreReceipt"
        )
    if receipt.format != DERIVED_STORE_FORMAT:
        raise HebbianDerivedStoreError("unsupported derived-store receipt")
    if type(receipt.learning_policy) is not HebbianLearningPolicy:
        raise HebbianDerivedStoreError(
            "learning_policy must be an exact HebbianLearningPolicy"
        )
    for name in (
        "source_database_sha256",
        "source_index_sha256",
        "source_store_receipt_sha256",
        "source_turn_sequence_sha256",
        "source_chunk_sequence_sha256",
        "derived_database_sha256",
        "derived_index_sha256",
        "derived_turn_sequence_sha256",
        "derived_chunk_sequence_sha256",
        "history_artifact_sha256",
        "history_receipt_sha256",
        "implementation_sha256",
        "environment_lock_sha256",
        "learning_policy_sha256",
        "association_artifact_sha256",
        "receipt_sha256",
    ):
        _digest(getattr(receipt, name), name)
    if receipt.learning_policy_sha256 != receipt.learning_policy.policy_sha256:
        raise HebbianDerivedStoreError("learning-policy seal mismatch")
    if (
        type(receipt.association_artifact_id) is not str
        or not receipt.association_artifact_id
        or receipt.association_artifact_id != receipt.association_artifact_id.strip()
    ):
        raise HebbianDerivedStoreError(
            "association_artifact_id must be an exact non-empty string"
        )
    expected_artifact = _association_artifact_from_bindings(
        history_artifact_sha256=receipt.history_artifact_sha256,
        history_receipt_sha256=receipt.history_receipt_sha256,
        source_store_receipt_sha256=receipt.source_store_receipt_sha256,
        learning_policy_sha256=receipt.learning_policy_sha256,
    )
    if receipt.association_artifact_id != expected_artifact.artifact_id:
        raise HebbianDerivedStoreError("association-artifact ID mismatch")
    if receipt.association_artifact_sha256 != identity_sha256(
        _artifact_payload(expected_artifact)
    ):
        raise HebbianDerivedStoreError("association-artifact seal mismatch")
    for name in (
        "events_offered",
        "events_applied",
        "graph_nodes",
        "graph_edges",
        "graph_event_receipts",
        "retained_request_token_state_bytes",
    ):
        _exact_int(getattr(receipt, name), name)
    if receipt.events_applied != receipt.events_offered:
        raise HebbianDerivedStoreError("not every frozen history event was applied")
    if receipt.graph_event_receipts != receipt.events_offered:
        raise HebbianDerivedStoreError("not every frozen event receipt was retained")
    if receipt.retained_request_token_state_bytes != 0:
        raise HebbianDerivedStoreError("derived store retained request token state")
    if receipt.source_turn_sequence_sha256 != receipt.derived_turn_sequence_sha256:
        raise HebbianDerivedStoreError("derived store changed the source turn sequence")
    if receipt.source_chunk_sequence_sha256 != receipt.derived_chunk_sequence_sha256:
        raise HebbianDerivedStoreError(
            "derived store changed the source chunk sequence"
        )
    if receipt.receipt_sha256 != identity_sha256(receipt.payload(include_seal=False)):
        raise HebbianDerivedStoreError("derived-store receipt seal mismatch")


def load_hebbian_derived_store_receipt(
    payload: Mapping[str, object],
) -> HebbianDerivedStoreReceipt:
    """Load one exact JSON receipt, failing closed on shape or seal drift."""

    if type(payload) is not dict or set(payload) != _RECEIPT_PAYLOAD_FIELDS:
        raise HebbianDerivedStoreError(
            "derived-store receipt payload has a noncanonical shape"
        )
    raw_policy = payload["learning_policy"]
    if type(raw_policy) is not dict or set(raw_policy) != _POLICY_PAYLOAD_FIELDS:
        raise HebbianDerivedStoreError(
            "Hebbian learning-policy payload has a noncanonical shape"
        )
    for name in ("learning_rate", "half_life_turns", "min_edge_score"):
        if type(raw_policy[name]) is not float:
            raise HebbianDerivedStoreError(
                f"learning_policy.{name} must be an exact JSON float"
            )
    for name in ("max_concepts_per_event", "max_degree"):
        if type(raw_policy[name]) is not int:
            raise HebbianDerivedStoreError(
                f"learning_policy.{name} must be an exact JSON integer"
            )
    if type(raw_policy["retain_all_event_receipts"]) is not bool:
        raise HebbianDerivedStoreError(
            "learning_policy.retain_all_event_receipts must be an exact boolean"
        )
    policy = HebbianLearningPolicy(
        format=raw_policy["format"],
        learning_rate=raw_policy["learning_rate"],
        half_life_turns=raw_policy["half_life_turns"],
        max_concepts_per_event=raw_policy["max_concepts_per_event"],
        max_degree=raw_policy["max_degree"],
        min_edge_score=raw_policy["min_edge_score"],
        retain_all_event_receipts=raw_policy["retain_all_event_receipts"],
    )
    receipt = HebbianDerivedStoreReceipt(
        format=payload["format"],
        source_database_sha256=payload["source_database_sha256"],
        source_index_sha256=payload["source_index_sha256"],
        source_store_receipt_sha256=payload["source_store_receipt_sha256"],
        source_turn_sequence_sha256=payload["source_turn_sequence_sha256"],
        source_chunk_sequence_sha256=payload["source_chunk_sequence_sha256"],
        derived_database_sha256=payload["derived_database_sha256"],
        derived_index_sha256=payload["derived_index_sha256"],
        derived_turn_sequence_sha256=payload["derived_turn_sequence_sha256"],
        derived_chunk_sequence_sha256=payload["derived_chunk_sequence_sha256"],
        history_artifact_sha256=payload["history_artifact_sha256"],
        history_receipt_sha256=payload["history_receipt_sha256"],
        implementation_sha256=payload["implementation_sha256"],
        environment_lock_sha256=payload["environment_lock_sha256"],
        learning_policy=policy,
        learning_policy_sha256=payload["learning_policy_sha256"],
        association_artifact_id=payload["association_artifact_id"],
        association_artifact_sha256=payload["association_artifact_sha256"],
        events_offered=payload["events_offered"],
        events_applied=payload["events_applied"],
        graph_nodes=payload["graph_nodes"],
        graph_edges=payload["graph_edges"],
        graph_event_receipts=payload["graph_event_receipts"],
        retained_request_token_state_bytes=payload[
            "retained_request_token_state_bytes"
        ],
        receipt_sha256=payload["receipt_sha256"],
    )
    _validate_receipt(receipt)
    return receipt


def apply_hebbian_history_to_staged_store(
    staged_store_dir: str | Path,
    *,
    source_database_path: str | Path,
    source_index_path: str | Path,
    history: HebbianHistoryArtifact,
    policy: HebbianLearningPolicy | None = None,
) -> HebbianDerivedStoreReceipt:
    """Apply every sealed event once and publish an immutable graph receipt."""

    active_policy = policy or HebbianLearningPolicy()
    if type(active_policy) is not HebbianLearningPolicy:
        raise TypeError("policy must be an exact HebbianLearningPolicy")
    store_dir = _resolved_store(staged_store_dir)
    source_database = _resolved_file(source_database_path, "source database")
    source_index = _resolved_file(source_index_path, "source index")
    database_path = _resolved_file(store_dir / "memory.db", "staged database")
    index_path = _resolved_file(store_dir / "hnsw_index.bin", "staged index")
    _require_no_sqlite_sidecars(source_database, "source database")
    if os.path.samefile(source_database, source_index):
        raise HebbianDerivedStoreError("source database and index must not alias")
    if os.path.samefile(database_path, index_path):
        raise HebbianDerivedStoreError("staged database and index must not alias")
    if any(
        os.path.samefile(derived, source)
        for derived in (database_path, index_path)
        for source in (source_database, source_index)
    ):
        raise HebbianDerivedStoreError("derived store must not alias the source store")
    children = {item.name for item in store_dir.iterdir()}
    if children != _STORE_INPUT_FILES:
        raise HebbianDerivedStoreError(
            f"staged store has unexpected or missing inputs: {sorted(children)!r}"
        )
    verified_history = verify_hebbian_history_artifact(
        history,
        source_database_path=source_database,
    )
    if any(len(event.event_id) > 256 for event in verified_history.events):
        raise HebbianDerivedStoreError(
            "Hebbian history event IDs must be at most 256 characters"
        )

    source_turn_sha, source_chunk_sha = _sequence_hashes(source_database)
    staged_turn_sha, staged_chunk_sha = _sequence_hashes(database_path)
    if (staged_turn_sha, staged_chunk_sha) != (source_turn_sha, source_chunk_sha):
        raise HebbianDerivedStoreError(
            "staged causal store changed the source turn or chunk sequence"
        )
    _require_clean_graph_namespace(database_path)

    artifact = _association_artifact(verified_history, active_policy)
    applied = 0
    max_history = max(1, len(verified_history.events))
    with Database(database_path) as database:
        associations = AssociationStore(database)
        associations.register_artifact(artifact)
        for event in verified_history.events:
            activations = {
                chunk_id: rank_discount(rank)
                for rank, chunk_id in enumerate(
                    event.chunk_ids[: active_policy.max_concepts_per_event],
                    start=1,
                )
            }
            update = associations.reinforce_retrieval_coaccess(
                artifact.artifact_id,
                event.event_id,
                activations,
                now_turn=event.now_turn,
                learning_rate=active_policy.learning_rate,
                half_life_turns=active_policy.half_life_turns,
                max_concepts_per_event=active_policy.max_concepts_per_event,
                max_degree=active_policy.max_degree,
                min_edge_score=active_policy.min_edge_score,
                max_event_history=max_history,
            )
            applied += int(update.created)
        stats = associations.hebbian_stats(artifact.artifact_id)
        database.commit()
        checkpoint = database.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is None or int(checkpoint[0]) != 0:
            raise HebbianDerivedStoreError("derived database WAL checkpoint failed")
    _remove_checkpoint_sidecars(database_path)

    derived_turn_sha, derived_chunk_sha = _sequence_hashes(database_path)
    unsigned = HebbianDerivedStoreReceipt(
        format=DERIVED_STORE_FORMAT,
        source_database_sha256=verified_history.receipt.source_database_sha256,
        source_index_sha256=file_sha256(source_index),
        source_store_receipt_sha256=(
            verified_history.receipt.source_store_receipt_sha256
        ),
        source_turn_sequence_sha256=source_turn_sha,
        source_chunk_sequence_sha256=source_chunk_sha,
        derived_database_sha256=file_sha256(database_path),
        derived_index_sha256=file_sha256(index_path),
        derived_turn_sequence_sha256=derived_turn_sha,
        derived_chunk_sequence_sha256=derived_chunk_sha,
        history_artifact_sha256=verified_history.artifact_sha256,
        history_receipt_sha256=verified_history.receipt.receipt_sha256,
        implementation_sha256=verified_history.receipt.implementation_sha256,
        environment_lock_sha256=verified_history.receipt.environment_lock_sha256,
        learning_policy=active_policy,
        learning_policy_sha256=active_policy.policy_sha256,
        association_artifact_id=artifact.artifact_id,
        association_artifact_sha256=identity_sha256(_artifact_payload(artifact)),
        events_offered=len(verified_history.events),
        events_applied=applied,
        graph_nodes=int(stats["nodes"]),
        graph_edges=int(stats["edges"]),
        graph_event_receipts=int(stats["event_receipts"]),
        retained_request_token_state_bytes=int(
            stats["retained_request_token_state_bytes"]
        ),
        receipt_sha256="0" * 64,
    )
    receipt = replace(
        unsigned,
        receipt_sha256=identity_sha256(unsigned.payload(include_seal=False)),
    )
    _validate_receipt(receipt)
    _publish_manifest(store_dir / MANIFEST_NAME, receipt)
    return verify_hebbian_derived_store(store_dir, expected=receipt)


def verify_hebbian_derived_store(
    store_dir: str | Path,
    *,
    expected: HebbianDerivedStoreReceipt,
) -> HebbianDerivedStoreReceipt:
    """Reopen a published store immutably and re-prove all durable bindings."""

    _validate_receipt(expected)
    root = _resolved_store(store_dir)
    expected_children = _STORE_INPUT_FILES | {MANIFEST_NAME}
    children = {item.name for item in root.iterdir()}
    if children != expected_children:
        raise HebbianDerivedStoreError(
            f"derived store has unexpected or missing files: {sorted(children)!r}"
        )
    database_path = _resolved_file(root / "memory.db", "derived database")
    index_path = _resolved_file(root / "hnsw_index.bin", "derived index")
    _require_no_sqlite_sidecars(database_path, "derived store")
    manifest_path = _resolved_file(root / MANIFEST_NAME, "derived manifest")
    raw_manifest = manifest_path.read_bytes()
    try:
        payload = json.loads(raw_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HebbianDerivedStoreError("derived manifest is not valid JSON") from exc
    try:
        canonical_manifest = (
            _canonical_bytes(payload) if type(payload) is dict else b""
        )
    except (TypeError, ValueError) as exc:
        raise HebbianDerivedStoreError(
            "derived manifest is not canonical JSON"
        ) from exc
    if type(payload) is not dict or raw_manifest != canonical_manifest:
        raise HebbianDerivedStoreError("derived manifest is not canonical JSON")
    if raw_manifest != _canonical_bytes(expected.payload()):
        raise HebbianDerivedStoreError("derived manifest differs from its receipt")
    if file_sha256(database_path) != expected.derived_database_sha256:
        raise HebbianDerivedStoreError("derived database digest changed")
    if file_sha256(index_path) != expected.derived_index_sha256:
        raise HebbianDerivedStoreError("derived index digest changed")
    turn_sha, chunk_sha = _sequence_hashes(database_path)
    if (turn_sha, chunk_sha) != (
        expected.derived_turn_sequence_sha256,
        expected.derived_chunk_sequence_sha256,
    ):
        raise HebbianDerivedStoreError("derived source coordinates changed")
    connection = _open_immutable(database_path)
    try:
        artifact_count_row = connection.execute(
            "SELECT COUNT(*) FROM association_artifacts"
        ).fetchone()
        artifact_row = connection.execute(
            "SELECT artifact_id, model_id, checkpoint_id, prefix_layers, "
            "head_layer, cav_layer, concept_names, head_count, created_at, "
            "metadata FROM association_artifacts WHERE artifact_id = ?",
            (expected.association_artifact_id,),
        ).fetchone()
        event_row = connection.execute(
            "SELECT COUNT(*) FROM hebbian_access_events WHERE artifact_id = ?",
            (expected.association_artifact_id,),
        ).fetchone()
        node_row = connection.execute(
            "SELECT COUNT(*) FROM hebbian_chunk_nodes WHERE artifact_id = ?",
            (expected.association_artifact_id,),
        ).fetchone()
        edge_row = connection.execute(
            "SELECT COUNT(*) FROM hebbian_chunk_edges WHERE artifact_id = ?",
            (expected.association_artifact_id,),
        ).fetchone()
        unrelated_counts = {
            table: int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in (
                "chunk_cav_signatures",
                "chunk_head_edges",
                "consolidation_access_events",
                "consolidation_nodes",
                "consolidation_edges",
            )
        }
        total_hebbian_counts = {
            table: int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in (
                "hebbian_access_events",
                "hebbian_chunk_nodes",
                "hebbian_chunk_edges",
            )
        }
    finally:
        connection.close()
    if artifact_count_row != (1,):
        raise HebbianDerivedStoreError(
            "derived store contains an unexpected association namespace"
        )
    if any(unrelated_counts.values()):
        raise HebbianDerivedStoreError("derived store contains unrelated graph state")
    if artifact_row is None:
        raise HebbianDerivedStoreError("derived association artifact is missing")
    try:
        stored_artifact = AssociationArtifact(
            artifact_id=artifact_row[0],
            model_id=artifact_row[1],
            checkpoint_id=artifact_row[2],
            prefix_layers=int(artifact_row[3]),
            head_layer=int(artifact_row[4]),
            cav_layer=(
                None if artifact_row[5] is None else int(artifact_row[5])
            ),
            concept_names=tuple(json.loads(artifact_row[6])),
            head_count=int(artifact_row[7]),
            created_at=artifact_row[8],
            metadata=json.loads(artifact_row[9]),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HebbianDerivedStoreError(
            "derived association artifact is invalid"
        ) from exc
    if identity_sha256(_artifact_payload(stored_artifact)) != (
        expected.association_artifact_sha256
    ):
        raise HebbianDerivedStoreError("derived association artifact changed")
    if event_row != (expected.graph_event_receipts,):
        raise HebbianDerivedStoreError("derived event receipt count changed")
    if node_row != (expected.graph_nodes,) or edge_row != (expected.graph_edges,):
        raise HebbianDerivedStoreError("derived graph counts changed")
    if total_hebbian_counts != {
        "hebbian_access_events": expected.graph_event_receipts,
        "hebbian_chunk_nodes": expected.graph_nodes,
        "hebbian_chunk_edges": expected.graph_edges,
    }:
        raise HebbianDerivedStoreError(
            "derived store contains an unexpected Hebbian namespace"
        )
    return expected


__all__ = [
    "ARTIFACT_CHECKPOINT_ID",
    "ARTIFACT_MODEL_ID",
    "DERIVED_STORE_FORMAT",
    "DETERMINISTIC_CREATED_AT",
    "HebbianDerivedStoreError",
    "HebbianDerivedStoreReceipt",
    "HebbianLearningPolicy",
    "LEARNING_POLICY_FORMAT",
    "MANIFEST_NAME",
    "apply_hebbian_history_to_staged_store",
    "load_hebbian_derived_store_receipt",
    "verify_hebbian_derived_store",
]
