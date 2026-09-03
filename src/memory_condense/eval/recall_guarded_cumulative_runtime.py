"""Build one query-ready causal-plus-discourse store for cumulative retrieval.

The ordinary causal cache is immutable after its manifest hashes are written,
so diffuse compilation must not be appended to a published cache hit.  This
module provides the explicit combined-build seam: stage exact corpus
identities, learn the causal overlay, compile discourse before publication,
then reopen the finished target read-only with the frozen causal-graph budget.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from memory_condense.application.condenser import MemoryCondenser, query_facets
from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import (
    DiscourseArtifact,
    DiscourseSnapshot,
    identity_sha256,
    quote_sha256,
)
from memory_condense.domain.integrity import file_sha256
from memory_condense.domain.sealed import SealedIdentity
from memory_condense.eval._identity import exact_int, sha256_digest
from memory_condense.eval.consolidation_replay import (
    FrozenQueryEmbedder,
    _source_user_queries,
    apply_rank_learning,
    stage_causal_store,
)
from memory_condense.eval.diffuse_compilation import (
    DiffuseCompilationPolicy,
    DiffuseCompilationReceipt,
    DiffuseSourceCompilationReceipt,
    compile_diffuse_artifact,
)
from memory_condense.eval._recall_guarded_cumulative_contracts import (
    causal_graph_context_budget,
)
from memory_condense.eval.schemas import EvalConfig


COMBINED_CUMULATIVE_STORE_FORMAT = (
    "memory-condense-recall-guarded-combined-store-v1"
)
COMBINED_CUMULATIVE_STORE_MANIFEST_FORMAT = (
    "memory-condense-recall-guarded-combined-store-manifest-v1"
)
COMBINED_CUMULATIVE_STORE_MANIFEST = "combined-cumulative-store.json"


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _plain_payload(value: object) -> object:
    """Project frozen dataclasses/mapping proxies into strict JSON values."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _plain_payload(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _plain_payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_payload(item) for item in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError(f"combined manifest contains a non-JSON value: {type(value)!r}")


def _write_combined_manifest(
    store_dir: Path,
    *,
    receipt: "CombinedCumulativeStoreReceipt",
    compilation: DiffuseCompilationReceipt,
    staging_stats: Mapping[str, object],
    learning_stats: Mapping[str, object],
) -> None:
    """Persist everything required for a later verified read-only reopen."""

    path = store_dir / COMBINED_CUMULATIVE_STORE_MANIFEST
    payload = {
        "format": COMBINED_CUMULATIVE_STORE_MANIFEST_FORMAT,
        "combined_store_receipt": _plain_payload(receipt),
        "compilation_receipt": _plain_payload(compilation),
        "staging_stats": _plain_payload(staging_stats),
        "learning_stats": _plain_payload(learning_stats),
    }
    with path.open("xb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _read_combined_manifest(
    store_dir: Path,
) -> tuple[
    "CombinedCumulativeStoreReceipt",
    DiffuseCompilationReceipt,
    dict[str, object],
    dict[str, object],
]:
    path = store_dir / COMBINED_CUMULATIVE_STORE_MANIFEST
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("combined cumulative store manifest is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("combined cumulative store manifest must be an object")
    if payload.get("format") != COMBINED_CUMULATIVE_STORE_MANIFEST_FORMAT:
        raise ValueError("unsupported combined cumulative store manifest format")
    if raw != _canonical_json_bytes(payload):
        raise ValueError("combined cumulative store manifest is not canonical JSON")
    try:
        combined = CombinedCumulativeStoreReceipt(
            **payload["combined_store_receipt"]
        )
        compilation_body = dict(payload["compilation_receipt"])
        compilation = DiffuseCompilationReceipt(
            artifact=DiscourseArtifact(**compilation_body.pop("artifact")),
            source_receipts=tuple(
                DiffuseSourceCompilationReceipt(**item)
                for item in compilation_body.pop("source_receipts")
            ),
            final_snapshot=DiscourseSnapshot(
                **compilation_body.pop("final_snapshot")
            ),
            **compilation_body,
        )
        staging = dict(payload["staging_stats"])
        learning = dict(payload["learning_stats"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("combined cumulative store manifest is invalid") from exc
    if combined.compilation_receipt_sha256 != compilation.receipt_sha256:
        raise ValueError("combined manifest changed its compilation receipt")
    return combined, compilation, staging, learning


def _canonical_created_at(value: object) -> str | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(str(value))
    parsed = (
        parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None
        else parsed.astimezone(timezone.utc)
    )
    return parsed.isoformat()


def _lexical_weights_sha256(value: object) -> str | None:
    if value is None:
        return None
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("chunk lexical_weights must be a JSON object")
    return identity_sha256(
        {str(term): float(weight) for term, weight in parsed.items()}
    )


def _store_identity(database_path: Path) -> tuple[str, int, int]:
    connection = sqlite3.connect(database_path)
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
        turn_rows = connection.execute(
            f"SELECT turn_id, role, text, {source_expression}, "
            f"{created_expression} FROM turns ORDER BY {ordinal_expression}"
        ).fetchall()
        if any(row[4] is None for row in turn_rows):
            raise ValueError("combined store source contains an undated turn")
        turns = tuple(
            {
                # Replay preserves semantic order, but historical stores may
                # use rowid (or contain gaps) instead of a dense ordinal.
                "position": position,
                "turn_id": str(row[0]),
                "role": str(row[1]),
                "text_sha256": quote_sha256(str(row[2])),
                "source_id": None if row[3] is None else str(row[3]),
                "created_at": _canonical_created_at(row[4]),
            }
            for position, row in enumerate(turn_rows, 1)
        )
        turn_positions = {
            str(row[0]): position for position, row in enumerate(turn_rows, 1)
        }
        chunk_rows = connection.execute(
            "SELECT chunk_id, turn_id, text, start_char, end_char, token_count, "
            "embedding, lexical_weights FROM chunks"
        ).fetchall()
        if any(row[6] is None for row in chunk_rows):
            raise ValueError("combined store source contains an unembedded chunk")
        try:
            ordered_chunks = sorted(
                chunk_rows,
                key=lambda row: (
                    turn_positions[str(row[1])],
                    int(row[3]),
                    int(row[4]),
                    str(row[0]),
                ),
            )
        except KeyError as exc:
            raise ValueError("combined store chunk refers to an unknown turn") from exc
        chunks = tuple(
            {
                "chunk_id": str(row[0]),
                "turn_id": str(row[1]),
                "text_sha256": quote_sha256(str(row[2])),
                "start_char": int(row[3]),
                "end_char": int(row[4]),
                "token_count": int(row[5]),
                "embedding_sha256": hashlib.sha256(bytes(row[6])).hexdigest(),
                "lexical_weights_sha256": _lexical_weights_sha256(row[7]),
            }
            for row in ordered_chunks
        )
    finally:
        connection.close()
    return (
        identity_sha256({"turns": turns, "chunks": chunks}),
        len(turns),
        len(chunks),
    )


def _query_batch(queries: Sequence[str], config: EvalConfig) -> tuple[str, ...]:
    if isinstance(queries, (str, bytes)):
        raise TypeError("held_out_queries must be a sequence of exact strings")
    values: list[str] = []
    for raw in queries:
        if type(raw) is not str:
            raise TypeError("held-out queries must be exact strings")
        query = raw.strip()
        if not query:
            raise ValueError("held-out queries must be non-empty")
        values.append(query)
        if config.retrieval.query_facet_retrieval:
            values.extend(
                query_facets(
                    query,
                    max_facets=config.retrieval.query_facet_max,
                )
            )
    batch = tuple(dict.fromkeys(values))
    if not batch:
        raise ValueError("at least one held-out query is required")
    return batch


@dataclass(frozen=True, slots=True)
class CombinedCumulativeStoreReceipt(SealedIdentity):
    """Text-free proof that causal and discourse layers share one corpus."""

    _SEAL_MISMATCH = "combined cumulative store receipt does not match"

    source_store_identity_sha256: str
    target_store_identity_sha256: str
    source_database_sha256: str
    target_database_sha256: str
    target_index_sha256: str
    retrieval_policy_sha256: str
    context_budget_sha256: str
    training_query_batch_sha256: str
    held_out_query_batch_sha256: str
    compilation_receipt_sha256: str
    artifact_id: str
    snapshot_sha256: str
    turn_count: int
    chunk_count: int
    causal_events: int
    causal_graph_edges: int
    retained_request_token_state_bytes: int = 0
    format: str = COMBINED_CUMULATIVE_STORE_FORMAT
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.format != COMBINED_CUMULATIVE_STORE_FORMAT:
            raise ValueError("unsupported combined cumulative store format")
        for name in (
            "source_store_identity_sha256",
            "target_store_identity_sha256",
            "source_database_sha256",
            "target_database_sha256",
            "target_index_sha256",
            "retrieval_policy_sha256",
            "context_budget_sha256",
            "training_query_batch_sha256",
            "held_out_query_batch_sha256",
            "compilation_receipt_sha256",
            "snapshot_sha256",
        ):
            sha256_digest(getattr(self, name), name)
        if self.source_store_identity_sha256 != self.target_store_identity_sha256:
            raise ValueError("combined store changed source turn/chunk identities")
        artifact = str(self.artifact_id).strip()
        if not artifact:
            raise ValueError("artifact_id must be non-empty")
        object.__setattr__(self, "artifact_id", artifact)
        for name in (
            "turn_count",
            "chunk_count",
            "causal_events",
            "causal_graph_edges",
            "retained_request_token_state_bytes",
        ):
            object.__setattr__(
                self,
                name,
                exact_int(getattr(self, name), name, minimum=0),
            )
        if self.turn_count < 1 or self.chunk_count < 1:
            raise ValueError("combined store cannot be empty")
        if self.retained_request_token_state_bytes != 0:
            raise ValueError("combined store retained request-token state")
        self._seal()


@dataclass(frozen=True, slots=True)
class PreparedRecallGuardedCumulativeStore:
    """Owned read-only store plus the receipts needed by the query runner."""

    condenser: MemoryCondenser
    compilation: DiffuseCompilationReceipt
    receipt: CombinedCumulativeStoreReceipt
    staging_stats: Mapping[str, int]
    learning_stats: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.receipt.compilation_receipt_sha256 != (
            self.compilation.receipt_sha256
        ):
            raise ValueError("prepared store changed its compilation receipt")
        if self.receipt.artifact_id != self.compilation.artifact.artifact_id:
            raise ValueError("prepared store changed its discourse artifact")
        if self.receipt.snapshot_sha256 != (
            self.compilation.final_snapshot.snapshot_sha256
        ):
            raise ValueError("prepared store changed its discourse snapshot")
        if self.receipt.retained_request_token_state_bytes != (
            self.compilation.persisted_request_token_state_bytes
        ):
            raise ValueError("prepared store changed its retention attestation")
        object.__setattr__(
            self,
            "staging_stats",
            MappingProxyType(dict(self.staging_stats)),
        )
        object.__setattr__(
            self,
            "learning_stats",
            MappingProxyType(dict(self.learning_stats)),
        )

    def close(self) -> None:
        self.condenser.close()

    def __enter__(self) -> "PreparedRecallGuardedCumulativeStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def build_recall_guarded_cumulative_store(
    source_database: str | Path,
    target_dir: str | Path,
    *,
    config: EvalConfig,
    embedder: Any,
    held_out_queries: Sequence[str],
    compilation_policy: DiffuseCompilationPolicy,
    coverage_selector: Any,
    qwen_scorer: Any | None = None,
    embedding_identity: Mapping[str, object] | None = None,
) -> PreparedRecallGuardedCumulativeStore:
    """Build and reopen one exact combined store without accepting benchmark gold."""

    if not isinstance(config, EvalConfig):
        raise TypeError("config must be an EvalConfig")
    retrieval = config.retrieval
    if retrieval.mode != "causal_graph" or not retrieval.coverage_selection:
        raise ValueError(
            "combined cumulative build requires causal_graph coverage retrieval"
        )
    if coverage_selector is None:
        raise ValueError("combined cumulative build requires a coverage selector")
    source = Path(source_database)
    target = Path(target_dir)
    if not source.is_file():
        raise FileNotFoundError(f"source database does not exist: {source}")
    if target.exists():
        raise FileExistsError(f"refusing to replace combined store: {target}")
    source_identity, source_turns, source_chunks = _store_identity(source)
    source_database_sha256 = file_sha256(source)
    query_batch = _query_batch(held_out_queries, config)
    training_queries = tuple(
        dict.fromkeys(
            text
            for text in _source_user_queries(source)
            if len(text.strip()) > 0
        )
    )
    if not training_queries:
        raise ValueError("source corpus has no historical user queries")
    bounded_training = tuple(
        text
        for text in training_queries
        if count_tokens(text)
        <= retrieval.consolidation_max_training_prompt_tokens
    )
    if not bounded_training:
        raise ValueError("source corpus has no bounded historical user query")
    training_vectors = embedder.embed_queries(bounded_training)
    training_embedder = FrozenQueryEmbedder(
        dict(zip(bounded_training, training_vectors, strict=True))
    )
    query_vectors = embedder.embed_queries(query_batch)
    query_embedder = FrozenQueryEmbedder(
        dict(zip(query_batch, query_vectors, strict=True))
    )
    if (
        _store_identity(source),
        file_sha256(source),
    ) != (
        (source_identity, source_turns, source_chunks),
        source_database_sha256,
    ):
        raise RuntimeError("source database changed while batching queries")
    budget = causal_graph_context_budget(retrieval)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.building-", dir=target.parent)
    )
    build_target = temporary_root / "store"
    condenser: MemoryCondenser | None = None
    try:
        events, staging = stage_causal_store(
            source,
            build_target,
            training_embedder,
            expansion_tokens=retrieval.consolidation_training_expansion_tokens,
            retrieval_k=retrieval.consolidation_training_k,
            max_event_nodes=retrieval.consolidation_max_event_nodes,
            new_event_nodes=retrieval.consolidation_new_event_nodes,
            max_prompt_tokens=(
                retrieval.consolidation_max_training_prompt_tokens
            ),
        )
        learning = apply_rank_learning(build_target, training_embedder, events)
        with MemoryCondenser(
            data_dir=build_target,
            chunker_min_tokens=config.chunker.min_tokens,
            chunker_max_tokens=config.chunker.max_tokens,
            auto_extract=False,
            budget=budget,
            embedder=training_embedder,
            persist_index_on_close=False,
            retriever_max_elements=max(1, source_chunks),
        ) as compiler:
            compilation = compile_diffuse_artifact(
                compiler,
                policy=compilation_policy,
                qwen_scorer=qwen_scorer,
                embedding_identity=embedding_identity,
            )
            causal_graph_edges = int(
                compiler.consolidation.stats().get("edges", 0)
            )
            retained = int(
                compiler.discourse.stats().get(
                    "retained_request_token_state_bytes",
                    0,
                )
            )

        target_identity, target_turns, target_chunks = _store_identity(
            build_target / "memory.db"
        )
        if (target_turns, target_chunks) != (source_turns, source_chunks):
            raise RuntimeError("combined staging changed source turn/chunk counts")
        if (
            _store_identity(source),
            file_sha256(source),
        ) != (
            (source_identity, source_turns, source_chunks),
            source_database_sha256,
        ):
            raise RuntimeError("source database changed during combined-store build")
        receipt = CombinedCumulativeStoreReceipt(
            source_store_identity_sha256=source_identity,
            target_store_identity_sha256=target_identity,
            source_database_sha256=source_database_sha256,
            target_database_sha256=file_sha256(build_target / "memory.db"),
            target_index_sha256=file_sha256(build_target / "hnsw_index.bin"),
            retrieval_policy_sha256=identity_sha256(
                retrieval.model_dump(mode="json")
            ),
            context_budget_sha256=identity_sha256(
                {
                    name: getattr(budget, name)
                    for name in budget.__dataclass_fields__
                }
            ),
            training_query_batch_sha256=identity_sha256(
                [
                    {"query_sha256": quote_sha256(item)}
                    for item in bounded_training
                ]
            ),
            held_out_query_batch_sha256=identity_sha256(
                [{"query_sha256": quote_sha256(item)} for item in query_batch]
            ),
            compilation_receipt_sha256=compilation.receipt_sha256,
            artifact_id=compilation.artifact.artifact_id,
            snapshot_sha256=compilation.final_snapshot.snapshot_sha256,
            turn_count=target_turns,
            chunk_count=target_chunks,
            causal_events=len(events),
            causal_graph_edges=causal_graph_edges,
            retained_request_token_state_bytes=retained,
        )
        _write_combined_manifest(
            build_target,
            receipt=receipt,
            compilation=compilation,
            staging_stats=staging,
            learning_stats=learning,
        )
        build_target.rename(target)
        temporary_root.rmdir()
        condenser = MemoryCondenser(
            data_dir=target,
            chunker_min_tokens=config.chunker.min_tokens,
            chunker_max_tokens=config.chunker.max_tokens,
            auto_extract=False,
            budget=budget,
            embedder=query_embedder,
            persist_index_on_close=False,
            retriever_max_elements=max(1, source_chunks),
            read_only=True,
        )
        condenser.set_context_candidate_selector(coverage_selector)
        setattr(condenser, "combined_cumulative_store_receipt", receipt)
        setattr(condenser, "cumulative_compilation_receipt", compilation)
        prepared = PreparedRecallGuardedCumulativeStore(
            condenser=condenser,
            compilation=compilation,
            receipt=receipt,
            staging_stats=dict(staging),
            learning_stats=dict(learning),
        )
    except BaseException as original:
        if condenser is not None:
            try:
                condenser.close()
            except BaseException as cleanup_error:
                original.add_note(
                    f"combined-store condenser cleanup also failed: {cleanup_error!r}"
                )
        if temporary_root.exists():
            shutil.rmtree(temporary_root, ignore_errors=True)
        # Publication is the resumability boundary.  Once the complete store
        # and its manifest have been atomically renamed into place, a later
        # read-only-open failure must not discard the expensive durable build.
        raise
    return prepared


def open_recall_guarded_cumulative_store(
    target_dir: str | Path,
    *,
    config: EvalConfig,
    embedder: Any,
    held_out_queries: Sequence[str],
    coverage_selector: Any,
) -> PreparedRecallGuardedCumulativeStore:
    """Verify and reopen an already published combined store read-only.

    Only the declared held-out query strings are embedded.  Source chunks,
    causal edges, and discourse artifacts are never rebuilt or modified.
    """

    if not isinstance(config, EvalConfig):
        raise TypeError("config must be an EvalConfig")
    retrieval = config.retrieval
    if retrieval.mode != "causal_graph" or not retrieval.coverage_selection:
        raise ValueError(
            "combined cumulative open requires causal_graph coverage retrieval"
        )
    if coverage_selector is None:
        raise ValueError("combined cumulative open requires a coverage selector")
    target = Path(target_dir)
    if not target.is_dir():
        raise FileNotFoundError(f"combined store does not exist: {target}")
    receipt, compilation, staging, learning = _read_combined_manifest(target)
    database_path = target / "memory.db"
    index_path = target / "hnsw_index.bin"
    if not database_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("combined store is missing its database or index")
    if file_sha256(database_path) != receipt.target_database_sha256:
        raise RuntimeError("combined store database changed after publication")
    if file_sha256(index_path) != receipt.target_index_sha256:
        raise RuntimeError("combined store index changed after publication")
    target_identity, turn_count, chunk_count = _store_identity(database_path)
    if (
        target_identity != receipt.target_store_identity_sha256
        or turn_count != receipt.turn_count
        or chunk_count != receipt.chunk_count
    ):
        raise RuntimeError("combined store source identity changed after publication")
    if receipt.retrieval_policy_sha256 != identity_sha256(
        retrieval.model_dump(mode="json")
    ):
        raise ValueError("combined store was built for another retrieval policy")
    budget = causal_graph_context_budget(retrieval)
    if receipt.context_budget_sha256 != identity_sha256(
        {
            name: getattr(budget, name)
            for name in budget.__dataclass_fields__
        }
    ):
        raise ValueError("combined store was built for another context budget")
    query_batch = _query_batch(held_out_queries, config)
    if receipt.held_out_query_batch_sha256 != identity_sha256(
        [{"query_sha256": quote_sha256(item)} for item in query_batch]
    ):
        raise ValueError("combined store was built for another held-out query batch")
    query_vectors = embedder.embed_queries(query_batch)
    query_embedder = FrozenQueryEmbedder(
        dict(zip(query_batch, query_vectors, strict=True))
    )
    condenser: MemoryCondenser | None = None
    try:
        condenser = MemoryCondenser(
            data_dir=target,
            chunker_min_tokens=config.chunker.min_tokens,
            chunker_max_tokens=config.chunker.max_tokens,
            auto_extract=False,
            budget=budget,
            embedder=query_embedder,
            persist_index_on_close=False,
            retriever_max_elements=max(1, chunk_count),
            read_only=True,
        )
        snapshot = condenser.discourse.snapshot()
        if snapshot.snapshot_sha256 != receipt.snapshot_sha256:
            raise RuntimeError("combined store discourse snapshot changed")
        if compilation.artifact.artifact_id not in snapshot.artifact_ids:
            raise RuntimeError("combined store lost its compiled discourse artifact")
        condenser.set_context_candidate_selector(coverage_selector)
        setattr(condenser, "combined_cumulative_store_receipt", receipt)
        setattr(condenser, "cumulative_compilation_receipt", compilation)
        return PreparedRecallGuardedCumulativeStore(
            condenser=condenser,
            compilation=compilation,
            receipt=receipt,
            staging_stats=staging,
            learning_stats=learning,
        )
    except BaseException:
        if condenser is not None:
            condenser.close()
        raise


__all__ = [
    "COMBINED_CUMULATIVE_STORE_FORMAT",
    "COMBINED_CUMULATIVE_STORE_MANIFEST",
    "COMBINED_CUMULATIVE_STORE_MANIFEST_FORMAT",
    "CombinedCumulativeStoreReceipt",
    "PreparedRecallGuardedCumulativeStore",
    "build_recall_guarded_cumulative_store",
    "open_recall_guarded_cumulative_store",
]
