"""Closed-world validation of frozen compiled and causal cache artifacts."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from .canonical import (
    AuditError,
    FileSnapshot,
    canonical_sha256,
    file_sha256,
    length_prefixed_digest,
    parse_json_object,
    read_file_snapshot,
    require_int,
    require_mapping,
    require_number,
    require_sha256,
    require_text,
)
from .population import Sample


COMPILED_RECEIPT_FIELDS = {
    "manifest_sha256",
    "cache_key",
    "sample_sha256",
    "database_sha256",
    "index_sha256",
    "embedding_execution_sha256",
    "implementation_sha256",
    "environment_lock_sha256",
    "turn_count",
    "chunk_count",
}
CAUSAL_RECEIPT_FIELDS = {
    "manifest_sha256",
    "cache_key",
    "sample_sha256",
    "compiled_cache_key",
    "database_sha256",
    "index_sha256",
    "build_protocol_sha256",
    "embedding_execution_sha256",
    "implementation_sha256",
    "environment_lock_sha256",
}

_COMPILED_MANIFEST_FIELDS = {
    "format",
    "cache_revision",
    "cache_key",
    "sample_id",
    "sample_sha256",
    "chunker_min_tokens",
    "chunker_max_tokens",
    "embedding_model",
    "embedding_dim",
    "embedding_execution",
    "implementation_sha256",
    "environment_lock_sha256",
    "schema_version",
    "turn_count",
    "chunk_count",
    "database_sha256",
    "index_sha256",
}
_CAUSAL_MANIFEST_FIELDS = {
    "format",
    "cache_revision",
    "build_protocol",
    "cache_key",
    "sample_id",
    "sample_sha256",
    "embedding_execution",
    "implementation_sha256",
    "environment_lock_sha256",
    "compiled_cache_key",
    "database_sha256",
    "index_sha256",
    "stats",
}

_TABLE_COLUMNS = {
    "turns": ("turn_id", "role", "text", "source_id", "created_at", "ordinal"),
    "chunks": (
        "chunk_id",
        "turn_id",
        "text",
        "start_char",
        "end_char",
        "token_count",
        "embedding",
        "lexical_weights",
        "hnsw_label",
        "term_count",
    ),
    "chunk_terms": ("term", "chunk_id", "tf"),
    "memory_items": (
        "mem_id",
        "type",
        "content",
        "details",
        "status",
        "supersedes",
        "pin",
        "energy",
        "half_life_s",
        "half_life_turns",
        "importance",
        "created_at",
        "last_access_at",
        "last_access_turn",
        "embedding",
        "content_hash",
    ),
    "memory_provenance": ("mem_id", "turn_id", "chunk_id", "quote"),
    "meta": ("key", "value"),
    "association_artifacts": (
        "artifact_id",
        "model_id",
        "checkpoint_id",
        "prefix_layers",
        "head_layer",
        "cav_layer",
        "concept_names",
        "head_count",
        "created_at",
        "metadata",
    ),
    "chunk_cav_signatures": (
        "chunk_id",
        "artifact_id",
        "signature",
        "created_turn",
        "access_count",
        "last_access_turn",
    ),
    "chunk_head_edges": (
        "source_chunk_id",
        "destination_chunk_id",
        "artifact_id",
        "head_weights",
        "qk_score",
        "ov_transport",
        "evidence_count",
        "traversal_count",
        "last_access_turn",
        "temporal_forward",
    ),
    "hebbian_access_events": (
        "artifact_id",
        "event_id",
        "observed_turn",
        "event_fingerprint",
        "member_count",
    ),
    "hebbian_chunk_nodes": (
        "artifact_id",
        "chunk_id",
        "access_mass",
        "access_count",
        "last_access_turn",
    ),
    "hebbian_chunk_edges": (
        "artifact_id",
        "chunk_low",
        "chunk_high",
        "coaccess_mass",
        "coaccess_count",
        "last_reinforced_turn",
    ),
    "consolidation_access_events": (
        "event_id",
        "observed_turn",
        "event_fingerprint",
        "member_count",
    ),
    "consolidation_nodes": (
        "node_key",
        "node_kind",
        "memory_id",
        "chunk_id",
        "access_mass",
        "access_count",
        "last_access_turn",
    ),
    "consolidation_edges": (
        "node_low",
        "node_high",
        "coactivation_mass",
        "coactivation_count",
        "last_reinforced_turn",
        "causal_count",
    ),
}

_ALLOWED_BLOBS = {
    ("chunks", "embedding"),
    ("memory_items", "embedding"),
    ("chunk_cav_signatures", "signature"),
    ("chunk_head_edges", "head_weights"),
}

_HEX_32 = re.compile(r"[0-9a-f]{32}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_CAUSAL_EVENT_ID = re.compile(r"causal-user:[1-9][0-9]*(?::part:[1-9][0-9]*)?")


@dataclass(frozen=True, slots=True)
class TurnRecord:
    turn_id: str
    role: str
    text: str
    source_id: str | None
    created_at: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: str
    turn_id: str
    text: str
    start_char: int
    end_char: int
    token_count: int
    role: str
    turn_text: str
    source_id: str
    source_timestamp: str | None


@dataclass(frozen=True, slots=True)
class CacheArtifact:
    kind: str
    directory: Path
    manifest: dict[str, Any]
    manifest_snapshot: FileSnapshot
    receipt: dict[str, Any]

    @property
    def database(self) -> Path:
        return self.directory / "memory.db"


@dataclass(frozen=True, slots=True)
class CachePair:
    compiled: CacheArtifact
    causal: CacheArtifact


@contextmanager
def immutable_database(path: str | Path) -> Iterator[sqlite3.Connection]:
    target = Path(path).resolve()
    uri = target.as_uri() + "?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only = ON")
        databases = connection.execute("PRAGMA database_list").fetchall()
        if len(databases) != 1 or databases[0][1] != "main":
            raise AuditError(f"database has unexpected attachments: {target}")
        yield connection
    except sqlite3.Error as exc:
        raise AuditError(f"cannot inspect immutable SQLite cache {target}: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()


def validate_cache_receipts(
    value: Any,
    *,
    sample_sha256: str,
    implementation_sha256: str,
    environment_sha256: str,
) -> dict[str, dict[str, Any]]:
    mapping = require_mapping(value, "cache receipts")
    if set(mapping) != {"compiled", "causal"}:
        raise AuditError("cache receipts must contain exactly compiled and causal")
    result: dict[str, dict[str, Any]] = {}
    for kind, fields in (
        ("compiled", COMPILED_RECEIPT_FIELDS),
        ("causal", CAUSAL_RECEIPT_FIELDS),
    ):
        entries = mapping[kind]
        if not isinstance(entries, list) or len(entries) != 1:
            raise AuditError(f"{kind} cache receipts must contain exactly one row")
        receipt = require_mapping(entries[0], f"{kind} cache receipt")
        if set(receipt) != fields:
            raise AuditError(f"{kind} cache receipt has an unexpected shape")
        for field, raw in receipt.items():
            if field in {"turn_count", "chunk_count"}:
                require_int(raw, f"{kind} receipt {field}")
            else:
                require_sha256(raw, f"{kind} receipt {field}")
        result[kind] = dict(receipt)
    compiled = result["compiled"]
    causal = result["causal"]
    if causal["compiled_cache_key"] != compiled["cache_key"]:
        raise AuditError("causal receipt does not bind the compiled cache key")
    for field, expected in (
        ("sample_sha256", sample_sha256),
        ("implementation_sha256", implementation_sha256),
        ("environment_lock_sha256", environment_sha256),
    ):
        if compiled[field] != expected or causal[field] != expected:
            raise AuditError(f"cache receipt mismatch for {field}")
    return result


def cache_receipts_sha256(receipts: dict[str, dict[str, Any]]) -> str:
    return canonical_sha256(
        {
            "compiled": [receipts["compiled"]],
            "causal": [receipts["causal"]],
        }
    )


def _load_manifest(path: Path) -> tuple[dict[str, Any], FileSnapshot]:
    snapshot = read_file_snapshot(path, "cache manifest")
    return (
        parse_json_object(snapshot.payload, f"cache manifest {path}"),
        snapshot,
    )


def _bad_constant(value: str) -> None:
    raise ValueError(f"non-finite number {value!r}")


def _validate_scalar_stats(stats: Any, label: str) -> None:
    root = require_mapping(stats, label)
    if set(root) != {"staging", "learning"}:
        raise AuditError(f"{label} must contain exactly staging and learning")
    staging = require_mapping(root["staging"], f"{label}.staging")
    expected_staging = {
        "source_turns",
        "events",
        "completed_episodes",
        "outcome_chunks_bound",
        "skipped_large_prompt",
        "skipped_insufficient_candidates",
        "elapsed_s",
    }
    if set(staging) != expected_staging:
        raise AuditError(f"{label}.staging has an unexpected shape")
    learning = require_mapping(root["learning"], f"{label}.learning")
    if set(learning) != {"events_offered", "events_applied", "elapsed_s", "graph"}:
        raise AuditError(f"{label}.learning has an unexpected shape")
    graph = require_mapping(learning["graph"], f"{label}.learning.graph")
    if set(graph) != {"nodes", "edges", "event_receipts", "retained_prompt_state_bytes"}:
        raise AuditError(f"{label}.learning.graph has an unexpected shape")
    if graph["retained_prompt_state_bytes"] != 0:
        raise AuditError("causal manifest reports retained prompt state")
    for key in expected_staging - {"elapsed_s"}:
        require_int(staging[key], f"{label}.staging.{key}")
    require_number(staging["elapsed_s"], f"{label}.staging.elapsed_s", minimum=0)
    for key in {"events_offered", "events_applied"}:
        require_int(learning[key], f"{label}.learning.{key}")
    require_number(learning["elapsed_s"], f"{label}.learning.elapsed_s", minimum=0)
    for key in {"nodes", "edges", "event_receipts", "retained_prompt_state_bytes"}:
        require_int(graph[key], f"{label}.learning.graph.{key}")


def _validate_manifest(
    artifact: CacheArtifact,
    *,
    sample: Sample,
    implementation_sha256: str,
    environment_sha256: str,
) -> None:
    manifest = artifact.manifest
    receipt = artifact.receipt
    expected_files = {
        "compiled": {"compiled-store.json", "memory.db", "hnsw_index.bin"},
        "causal": {"causal-store.json", "memory.db", "hnsw_index.bin"},
    }[artifact.kind]
    try:
        actual_files = {path.name for path in artifact.directory.iterdir() if path.is_file()}
        actual_dirs = [path for path in artifact.directory.iterdir() if path.is_dir()]
    except OSError as exc:
        raise AuditError(f"cannot inspect cache directory {artifact.directory}: {exc}") from exc
    if actual_dirs or actual_files != expected_files:
        raise AuditError(f"cache entry is not closed-world: {artifact.directory}")
    manifest_name = "compiled-store.json" if artifact.kind == "compiled" else "causal-store.json"
    if artifact.manifest_snapshot.sha256 != receipt["manifest_sha256"]:
        raise AuditError(f"{artifact.kind} manifest hash disagrees with report receipt")
    if file_sha256(artifact.database) != receipt["database_sha256"]:
        raise AuditError(f"{artifact.kind} database hash disagrees with report receipt")
    if file_sha256(artifact.directory / "hnsw_index.bin") != receipt["index_sha256"]:
        raise AuditError(f"{artifact.kind} ANN hash disagrees with report receipt")
    if manifest.get("cache_key") != receipt["cache_key"]:
        raise AuditError(f"{artifact.kind} cache key mismatch")
    for field, expected in (
        ("sample_sha256", sample.sha256),
        ("implementation_sha256", implementation_sha256),
        ("environment_lock_sha256", environment_sha256),
        ("database_sha256", receipt["database_sha256"]),
        ("index_sha256", receipt["index_sha256"]),
    ):
        if manifest.get(field) != expected:
            raise AuditError(f"{artifact.kind} manifest mismatch for {field}")
    if manifest.get("sample_id") != sample.sample_id:
        raise AuditError(f"{artifact.kind} manifest sample ID mismatch")
    embedding_execution = require_mapping(
        manifest.get("embedding_execution"),
        f"{artifact.kind} manifest embedding_execution",
    )
    if canonical_sha256(embedding_execution) != receipt["embedding_execution_sha256"]:
        raise AuditError(f"{artifact.kind} embedding execution receipt mismatch")
    if artifact.kind == "compiled":
        if set(manifest) != _COMPILED_MANIFEST_FIELDS:
            raise AuditError("compiled manifest has an unexpected shape")
        if manifest.get("format") != "memory-condense-compiled-benchmark-store-v1":
            raise AuditError("compiled cache format mismatch")
        if manifest.get("cache_revision") != 3 or manifest.get("schema_version") != 9:
            raise AuditError("compiled cache revision/schema mismatch")
        minimum = require_int(
            manifest.get("chunker_min_tokens"),
            "compiled manifest chunker_min_tokens",
            minimum=1,
        )
        maximum = require_int(
            manifest.get("chunker_max_tokens"),
            "compiled manifest chunker_max_tokens",
            minimum=minimum,
        )
        if maximum < minimum:
            raise AuditError("compiled chunker bounds are reversed")
        require_text(manifest.get("embedding_model"), "compiled manifest embedding_model")
        require_int(
            manifest.get("embedding_dim"),
            "compiled manifest embedding_dim",
            minimum=1,
        )
        if manifest.get("turn_count") != receipt["turn_count"]:
            raise AuditError("compiled turn-count receipt mismatch")
        if manifest.get("chunk_count") != receipt["chunk_count"]:
            raise AuditError("compiled chunk-count receipt mismatch")
    else:
        if set(manifest) != _CAUSAL_MANIFEST_FIELDS:
            raise AuditError("causal manifest has an unexpected shape")
        if manifest.get("format") != "memory-condense-causal-benchmark-store-v1":
            raise AuditError("causal cache format mismatch")
        if manifest.get("cache_revision") != 3:
            raise AuditError("causal cache revision mismatch")
        if manifest.get("build_protocol") != "causal-training-query-only-v1":
            raise AuditError("causal cache build protocol mismatch")
        if manifest.get("compiled_cache_key") != receipt["compiled_cache_key"]:
            raise AuditError("causal manifest compiled-cache link mismatch")
        protocol_hash = __import__("hashlib").sha256(
            str(manifest["build_protocol"]).encode("utf-8")
        ).hexdigest()
        if protocol_hash != receipt["build_protocol_sha256"]:
            raise AuditError("causal build-protocol receipt mismatch")
        _validate_scalar_stats(manifest.get("stats"), "causal manifest stats")


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _normalized_sql(value: Any) -> str | None:
    return None if value is None else " ".join(str(value).split()).casefold()


def _schema_contract(connection: sqlite3.Connection) -> dict[str, Any]:
    definitions = tuple(
        (str(row[0]), str(row[1]), str(row[2]), _normalized_sql(row[3]))
        for row in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE type IN ('table', 'index', 'trigger', 'view') "
            "ORDER BY type, name"
        ).fetchall()
    )
    table_info: dict[str, tuple[tuple[Any, ...], ...]] = {}
    foreign_keys: dict[str, tuple[tuple[Any, ...], ...]] = {}
    indexes: dict[str, tuple[tuple[Any, ...], ...]] = {}
    for table in _TABLE_COLUMNS:
        quoted = _quote_identifier(table)
        table_info[table] = tuple(
            (
                str(row[1]),
                str(row[2]).strip().upper(),
                int(row[3]),
                None if row[4] is None else str(row[4]),
                int(row[5]),
            )
            for row in connection.execute(f"PRAGMA table_info({quoted})").fetchall()
        )
        foreign_keys[table] = tuple(
            tuple(row) for row in connection.execute(f"PRAGMA foreign_key_list({quoted})")
        )
        index_rows: list[tuple[Any, ...]] = []
        for row in connection.execute(f"PRAGMA index_list({quoted})").fetchall():
            name = str(row[1])
            index_info = tuple(
                tuple(value) for value in connection.execute(
                    f"PRAGMA index_xinfo({_quote_identifier(name)})"
                ).fetchall()
            )
            index_rows.append(
                (name, int(row[2]), str(row[3]), int(row[4]), index_info)
            )
        indexes[table] = tuple(sorted(index_rows, key=lambda value: value[0]))
    triggers = tuple(
        (str(row[0]), str(row[1]), _normalized_sql(row[2]))
        for row in connection.execute(
            "SELECT name, tbl_name, sql FROM sqlite_master "
            "WHERE type='trigger' ORDER BY name"
        ).fetchall()
    )
    return {
        "definitions": definitions,
        "table_info": table_info,
        "foreign_keys": foreign_keys,
        "indexes": indexes,
        "triggers": triggers,
    }


@lru_cache(maxsize=4)
def _expected_schema_contract(schema_sql: str) -> dict[str, Any]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(schema_sql)
        return _schema_contract(connection)
    except sqlite3.Error as exc:
        raise AuditError(f"cannot materialize frozen schema-v9 contract: {exc}") from exc
    finally:
        connection.close()


def _validate_runtime_storage_types(
    connection: sqlite3.Connection,
    contract: dict[str, Any],
    path: Path,
) -> None:
    for table, columns in contract["table_info"].items():
        for column, declared_type, not_null, _default, primary_key in columns:
            observed = {
                str(row[0])
                for row in connection.execute(
                    f"SELECT DISTINCT typeof({_quote_identifier(column)}) "
                    f"FROM {_quote_identifier(table)}"
                ).fetchall()
            }
            allowed = set()
            if not not_null and not primary_key:
                allowed.add("null")
            if declared_type == "TEXT":
                allowed.add("text")
            elif declared_type == "BLOB":
                if (table, column) not in _ALLOWED_BLOBS:
                    raise AuditError(f"unapproved BLOB state column {table}.{column}")
                allowed.add("blob")
            elif declared_type == "INTEGER":
                allowed.add("integer")
            elif declared_type == "REAL":
                allowed.update({"integer", "real"})
            else:
                raise AuditError(f"unrecognized frozen SQLite affinity {table}.{column}")
            if not observed.issubset(allowed):
                raise AuditError(
                    f"SQLite runtime storage class mismatch for {table}.{column} "
                    f"in {path}: {sorted(observed)}"
                )


def _validate_chunk_payloads(
    connection: sqlite3.Connection,
    path: Path,
    embedding_dim: int,
) -> None:
    expected_bytes = embedding_dim * 4
    for chunk_id, embedding, lexical_weights, term_count, hnsw_label in connection.execute(
        "SELECT chunk_id, embedding, lexical_weights, term_count, hnsw_label FROM chunks"
    ):
        if not isinstance(chunk_id, str) or _HEX_32.fullmatch(chunk_id) is None:
            raise AuditError(f"chunk ID is not a lowercase UUID hex value in {path}")
        if not isinstance(embedding, bytes) or len(embedding) != expected_bytes:
            raise AuditError(f"chunk embedding has the wrong fixed width in {path}: {chunk_id}")
        if any(not math.isfinite(value[0]) for value in struct.iter_unpack("<f", embedding)):
            raise AuditError(f"chunk embedding contains a non-finite value: {chunk_id}")
        if isinstance(hnsw_label, bool) or not isinstance(hnsw_label, int) or hnsw_label < 0:
            raise AuditError(f"chunk HNSW label is invalid: {chunk_id}")
        try:
            weights = (
                {}
                if lexical_weights is None
                else json.loads(lexical_weights, parse_constant=_bad_constant)
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AuditError(f"chunk lexical weights are invalid: {chunk_id}") from exc
        if not isinstance(weights, dict) or any(
            not isinstance(term, str)
            or not term
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) <= 0
            or float(weight) != int(float(weight))
            for term, weight in weights.items()
        ):
            raise AuditError(f"chunk lexical weights have an unsafe shape: {chunk_id}")
        if isinstance(term_count, bool) or not isinstance(term_count, int):
            raise AuditError(f"chunk lexical term count is invalid: {chunk_id}")
        if term_count != sum(int(float(weight)) for weight in weights.values()):
            raise AuditError(f"chunk lexical term count disagrees with weights: {chunk_id}")


def _validate_scalar_graph(connection: sqlite3.Connection, path: Path) -> None:
    chunk_ids = {
        str(row[0]) for row in connection.execute("SELECT chunk_id FROM chunks").fetchall()
    }
    for event_id, observed_turn, fingerprint, member_count in connection.execute(
        "SELECT event_id, observed_turn, event_fingerprint, member_count "
        "FROM consolidation_access_events"
    ):
        if not isinstance(event_id, str) or _CAUSAL_EVENT_ID.fullmatch(event_id) is None:
            raise AuditError(f"consolidation event ID is outside the frozen protocol: {path}")
        if not isinstance(fingerprint, str) or _HEX_64.fullmatch(fingerprint) is None:
            raise AuditError(f"consolidation event fingerprint is not SHA-256: {path}")
        if not isinstance(observed_turn, int) or observed_turn < 0:
            raise AuditError(f"consolidation event turn is invalid: {path}")
        if not isinstance(member_count, int) or member_count < 0:
            raise AuditError(f"consolidation event member count is invalid: {path}")
    node_keys: set[str] = set()
    for node_key, kind, memory_id, chunk_id, mass, count, turn in connection.execute(
        "SELECT node_key, node_kind, memory_id, chunk_id, access_mass, "
        "access_count, last_access_turn FROM consolidation_nodes"
    ):
        if (
            kind != "chunk"
            or memory_id is not None
            or not isinstance(chunk_id, str)
            or chunk_id not in chunk_ids
            or node_key != f"chunk:{chunk_id}"
        ):
            raise AuditError(f"consolidation node is not an exact chunk reference: {path}")
        if not math.isfinite(float(mass)) or float(mass) < 0:
            raise AuditError(f"consolidation node mass is invalid: {path}")
        if not isinstance(count, int) or count < 0 or not isinstance(turn, int) or turn < 0:
            raise AuditError(f"consolidation node counters are invalid: {path}")
        node_keys.add(str(node_key))
    for low, high, mass, count, turn, causal_count in connection.execute(
        "SELECT node_low, node_high, coactivation_mass, coactivation_count, "
        "last_reinforced_turn, causal_count FROM consolidation_edges"
    ):
        if low not in node_keys or high not in node_keys or not str(low) < str(high):
            raise AuditError(f"consolidation edge endpoints are invalid: {path}")
        if not math.isfinite(float(mass)) or float(mass) < 0:
            raise AuditError(f"consolidation edge mass is invalid: {path}")
        if any(
            not isinstance(value, int) or value < 0
            for value in (count, turn, causal_count)
        ):
            raise AuditError(f"consolidation edge counters are invalid: {path}")


def _validate_database_schema(
    path: Path,
    *,
    expected_schema_sql: str,
    embedding_dim: int,
) -> dict[str, int]:
    expected_contract = _expected_schema_contract(expected_schema_sql)
    with immutable_database(path) as connection:
        quick = connection.execute("PRAGMA quick_check").fetchall()
        if quick != [("ok",)]:
            raise AuditError(f"SQLite quick_check failed: {path}")
        actual_contract = _schema_contract(connection)
        if actual_contract != expected_contract:
            raise AuditError(f"SQLite schema objects differ from frozen schema v9: {path}")
        _validate_runtime_storage_types(connection, expected_contract, path)
        meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
        if set(meta) - {"schema_version", "next_hnsw_label"}:
            raise AuditError(f"SQLite meta contains unapproved state keys: {path}")
        if meta.get("schema_version") != "9":
            raise AuditError(f"SQLite schema_version is not 9: {path}")
        if "next_hnsw_label" in meta:
            try:
                next_label = int(meta["next_hnsw_label"])
            except (TypeError, ValueError) as exc:
                raise AuditError(f"SQLite next_hnsw_label is invalid: {path}") from exc
            if next_label < 0 or str(next_label) != meta["next_hnsw_label"]:
                raise AuditError(f"SQLite next_hnsw_label is not canonical: {path}")
        counts = {
            table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in _TABLE_COLUMNS
        }
        forbidden_persisted_partitions = {
            "memory_items",
            "memory_provenance",
            "association_artifacts",
            "chunk_cav_signatures",
            "chunk_head_edges",
            "hebbian_access_events",
            "hebbian_chunk_nodes",
            "hebbian_chunk_edges",
        }
        populated = {
            table: counts[table]
            for table in forbidden_persisted_partitions
            if counts[table] != 0
        }
        if populated:
            raise AuditError(
                "frozen treatment cache contains an unapproved persisted "
                f"memory/transformer-state partition: {populated}"
            )
        _validate_chunk_payloads(connection, path, embedding_dim)
        _validate_scalar_graph(connection, path)
    return counts


def _discover_root(
    root: str | Path,
    *,
    kind: str,
    receipts: dict[str, dict[str, Any]],
) -> dict[str, CacheArtifact]:
    base = Path(root).resolve()
    if not base.is_dir():
        raise AuditError(f"{kind} cache root is not a directory: {base}")
    manifest_name = "compiled-store.json" if kind == "compiled" else "causal-store.json"
    by_key: dict[str, CacheArtifact] = {}
    try:
        children = list(base.iterdir())
    except OSError as exc:
        raise AuditError(f"cannot enumerate {kind} cache root {base}: {exc}") from exc
    if any(not child.is_dir() or child.is_symlink() for child in children):
        raise AuditError(f"{kind} cache root may contain only ordinary entry directories")
    for child in children:
        manifest_path = child / manifest_name
        if not manifest_path.is_file():
            raise AuditError(f"cache entry lacks {manifest_name}: {child}")
        manifest, manifest_snapshot = _load_manifest(manifest_path)
        key = require_sha256(manifest.get("cache_key"), f"{kind} manifest cache_key")
        if key in by_key:
            raise AuditError(f"duplicate {kind} cache key {key}")
        receipt = receipts.get(key)
        if receipt is None:
            raise AuditError(f"{kind} cache root contains an unreported cache entry")
        by_key[key] = CacheArtifact(
            kind,
            child,
            manifest,
            manifest_snapshot,
            receipt,
        )
    if set(by_key) != set(receipts):
        raise AuditError(f"{kind} cache root does not exactly match report receipts")
    return by_key


def build_cache_pairs(
    compiled_root: str | Path,
    causal_root: str | Path,
    *,
    samples: dict[str, Sample],
    sample_receipts: dict[str, dict[str, dict[str, Any]]],
    implementation_sha256: str,
    environment_sha256: str,
    database_schema_sql: str,
) -> dict[str, CachePair]:
    compiled_receipts = {
        value["compiled"]["cache_key"]: value["compiled"]
        for value in sample_receipts.values()
    }
    causal_receipts = {
        value["causal"]["cache_key"]: value["causal"]
        for value in sample_receipts.values()
    }
    if len(compiled_receipts) != len(sample_receipts) or len(causal_receipts) != len(sample_receipts):
        raise AuditError("validation samples reuse a cache key")
    compiled = _discover_root(compiled_root, kind="compiled", receipts=compiled_receipts)
    causal = _discover_root(causal_root, kind="causal", receipts=causal_receipts)
    pairs: dict[str, CachePair] = {}
    for sample_sha, receipts in sample_receipts.items():
        sample = samples[sample_sha]
        compiled_artifact = compiled[receipts["compiled"]["cache_key"]]
        causal_artifact = causal[receipts["causal"]["cache_key"]]
        _validate_manifest(
            compiled_artifact,
            sample=sample,
            implementation_sha256=implementation_sha256,
            environment_sha256=environment_sha256,
        )
        _validate_manifest(
            causal_artifact,
            sample=sample,
            implementation_sha256=implementation_sha256,
            environment_sha256=environment_sha256,
        )
        compiled_embedding = require_mapping(
            compiled_artifact.manifest.get("embedding_execution"),
            "compiled embedding execution",
        )
        causal_embedding = require_mapping(
            causal_artifact.manifest.get("embedding_execution"),
            "causal embedding execution",
        )
        if canonical_sha256(compiled_embedding) != canonical_sha256(causal_embedding):
            raise AuditError("compiled and causal embedding executions differ")
        embedding_dim = require_int(
            compiled_artifact.manifest.get("embedding_dim"),
            "compiled embedding dimension",
            minimum=1,
        )
        compiled_counts = _validate_database_schema(
            compiled_artifact.database,
            expected_schema_sql=database_schema_sql,
            embedding_dim=embedding_dim,
        )
        causal_counts = _validate_database_schema(
            causal_artifact.database,
            expected_schema_sql=database_schema_sql,
            embedding_dim=embedding_dim,
        )
        if compiled_counts["turns"] != receipts["compiled"]["turn_count"]:
            raise AuditError("compiled database turn count mismatch")
        if compiled_counts["chunks"] != receipts["compiled"]["chunk_count"]:
            raise AuditError("compiled database chunk count mismatch")
        if compiled_counts != causal_counts:
            # Learned tables intentionally differ. Only source tables must agree.
            for table in ("turns", "chunks", "chunk_terms", "memory_items", "memory_provenance"):
                if compiled_counts[table] != causal_counts[table]:
                    raise AuditError(f"compiled/causal source-table count mismatch: {table}")
        graph = require_mapping(
            require_mapping(
                require_mapping(
                    causal_artifact.manifest.get("stats"),
                    "causal manifest stats",
                ).get("learning"),
                "causal manifest stats.learning",
            ).get("graph"),
            "causal manifest stats.learning.graph",
        )
        for field, table in (
            ("nodes", "consolidation_nodes"),
            ("edges", "consolidation_edges"),
            ("event_receipts", "consolidation_access_events"),
        ):
            if graph.get(field) != causal_counts[table]:
                raise AuditError(f"causal graph statistic disagrees with {table}")
        pairs[sample_sha] = CachePair(compiled_artifact, causal_artifact)
    return pairs


def source_surface_sha256(path: Path) -> str:
    with immutable_database(path) as connection:
        turns = connection.execute(
            "SELECT role, text, source_id, ordinal FROM turns ORDER BY ordinal"
        )
        chunks = connection.execute(
            "SELECT c.chunk_id, t.ordinal, c.text, c.start_char, c.end_char, "
            "c.token_count FROM chunks c JOIN turns t ON t.turn_id = c.turn_id "
            "ORDER BY c.rowid"
        )
        return canonical_sha256(
            {
                "turns": length_prefixed_digest(turns),
                "chunks": length_prefixed_digest(chunks),
            }
        )


def load_source_records(path: Path, sample: Sample) -> tuple[list[TurnRecord], list[ChunkRecord]]:
    with immutable_database(path) as connection:
        raw_turns = connection.execute(
            "SELECT turn_id, role, text, source_id, created_at, ordinal "
            "FROM turns ORDER BY ordinal"
        ).fetchall()
        if len(raw_turns) != len(sample.turns):
            raise AuditError("cache transcript length does not match reconstructed sample")
        turns: list[TurnRecord] = []
        by_id: dict[str, TurnRecord] = {}
        for index, row in enumerate(raw_turns):
            expected_role, expected_text = sample.turns[index]
            expected_source = sample.turn_source_ids[index]
            actual = TurnRecord(
                turn_id=str(row[0]),
                role=str(row[1]),
                text=str(row[2]),
                source_id=(None if row[3] is None else str(row[3])),
                created_at=str(row[4]),
                ordinal=int(row[5]),
            )
            if _HEX_32.fullmatch(actual.turn_id) is None:
                raise AuditError("cache turn ID is not a lowercase UUID hex value")
            try:
                parsed_created_at = datetime.fromisoformat(actual.created_at)
            except ValueError as exc:
                raise AuditError("cache turn creation time is not ISO-8601") from exc
            if parsed_created_at.tzinfo is None or len(actual.created_at) > 48:
                raise AuditError("cache turn creation time lacks a bounded timezone")
            if actual.ordinal != index + 1:
                raise AuditError("cache turn ordinals are not exact and contiguous")
            if (actual.role, actual.text, actual.source_id) != (
                expected_role,
                expected_text,
                expected_source,
            ):
                raise AuditError("cache transcript differs from reconstructed dataset")
            if actual.turn_id in by_id:
                raise AuditError("cache contains duplicate turn IDs")
            turns.append(actual)
            by_id[actual.turn_id] = actual
        timestamps: dict[str, str] = {}
        for turn in turns:
            if turn.role != "system" or turn.source_id is None:
                continue
            match = __import__("re").fullmatch(
                r"\[(?P<source>.+?) took place at (?P<timestamp>.+?)\]\s*",
                turn.text.strip(),
            )
            if match is not None:
                timestamps.setdefault(turn.source_id, match.group("timestamp").strip())
        raw_chunks = connection.execute(
            "SELECT chunk_id, turn_id, text, start_char, end_char, token_count "
            "FROM chunks ORDER BY rowid"
        ).fetchall()
    chunks: list[ChunkRecord] = []
    for row in raw_chunks:
        turn = by_id.get(str(row[1]))
        if turn is None:
            raise AuditError("chunk refers to an absent turn")
        start = int(row[3])
        end = int(row[4])
        if start < 0 or end < start or end > len(turn.text):
            raise AuditError("chunk has invalid turn-relative source coordinates")
        source_id = turn.source_id or turn.turn_id
        chunks.append(
            ChunkRecord(
                chunk_id=str(row[0]),
                turn_id=turn.turn_id,
                text=str(row[2]),
                start_char=start,
                end_char=end,
                token_count=int(row[5]),
                role=turn.role,
                turn_text=turn.text,
                source_id=source_id,
                source_timestamp=timestamps.get(source_id),
            )
        )
    return turns, chunks
