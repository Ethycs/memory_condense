"""Run the streamlined, matched S0/H1 experiment over the sealed 1M run.

History construction is the sole embedding/store-write phase.  Preflight is
provider-free, answer calls require an exact cardinality gate, replay consumes
only immutable journals, and gold data is imported only after both answer
artifacts and journals have been revalidated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import identity_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval import fast_1m_hebbian_answer_runtime as _answer_runtime
from memory_condense.eval._artifact_json import (
    canonical_json_bytes as _canonical_json_bytes,
)
from memory_condense.eval.consolidation_replay import (
    FrozenQueryEmbedder,
    RetrievalAccessCapture,
    RetrievalAccessCaptureSink,
    stage_causal_store,
)
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from memory_condense.eval.fast_hebbian_prompts import (
    ARM_IDS,
    S0_STAGE_ID,
    build_fast_hebbian_prompt_population,
)
from memory_condense.eval.hebbian_derived_store import (
    MANIFEST_NAME as DERIVED_MANIFEST_NAME,
    HebbianDerivedStoreReceipt,
    apply_hebbian_history_to_staged_store,
    load_hebbian_derived_store_receipt,
    verify_hebbian_derived_store,
)
from memory_condense.eval.hebbian_history import (
    HebbianHistoryArtifact,
    load_hebbian_history_artifact,
    seal_hebbian_history_artifact,
    verify_hebbian_history_artifact,
)
from memory_condense.eval.recall_guarded_cumulative_fast_artifact import (
    ORIGINAL_1M_RETRIEVAL_SHA256,
    FastRetrievalArtifact,
    load_fast_retrieval_artifact,
)
from memory_condense.eval.recall_guarded_cumulative_runtime import (
    COMBINED_CUMULATIVE_STORE_MANIFEST,
    CombinedCumulativeStoreReceipt,
    _read_combined_manifest,
)
from memory_condense.eval.reproducibility import (
    environment_lock_sha256,
    implementation_sha256,
)
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    EmbeddingService,
)


HISTORY_FILE_NAME = "history.json"
ANSWER_MANIFEST_FORMAT = _answer_runtime.ANSWER_MANIFEST_FORMAT
SCORE_MANIFEST_FORMAT = _answer_runtime.SCORE_MANIFEST_FORMAT
ZERO_STATE_CONTRACT = _answer_runtime.ZERO_STATE_CONTRACT

DEFAULT_RETRIEVAL = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-"
    "development-20260821/retrieval.json"
)
DEFAULT_SOURCE_STORE = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-"
    "development-20260821/combined-store"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-fast-hebbian-development-20260822"
)
DEFAULT_SPLIT = Path(
    "docs/10 - Research Log/data/longmemeval-95-target-split-v2.json"
)
DEFAULT_GATEWAY_URL = "https://central-dev.zt:4000/v1"
DEFAULT_GATEWAY_MODEL = "codex_sdk/gpt-5.6-terra"
DEFAULT_CALLER_MODEL = "openai/codex_sdk/gpt-5.6-terra"

_SOURCE_FILES = frozenset(
    {"memory.db", "hnsw_index.bin", COMBINED_CUMULATIVE_STORE_MANIFEST}
)


def _is_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _publish_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FileExistsError(f"refusing to replace a symbolic link: {path}")
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace another artifact: {path}")
        return
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


def _atomic_write_json(path: Path, value: object) -> str:
    """Publish immutable canonical JSON and its exact digest sidecar."""

    payload = _canonical_json_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    _publish_bytes(path, payload)
    _publish_bytes(
        path.with_name(path.name + ".sha256"),
        f"{digest}  {path.name}\n".encode("ascii"),
    )
    return digest


def _read_canonical_json(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular file: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"artifact is not valid JSON: {path}") from exc
    if type(payload) is not dict or raw != _canonical_json_bytes(payload):
        raise ValueError(f"artifact is not a canonical JSON object: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if (
        sidecar.is_symlink()
        or not sidecar.is_file()
        or sidecar.read_bytes() != expected
    ):
        raise ValueError(f"artifact digest sidecar is missing or invalid: {path}")
    return payload, digest


def _read_manifest(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"manifest must be a regular file: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifest is not valid JSON: {path}") from exc
    if type(payload) is not dict or raw != _canonical_json_bytes(payload):
        raise ValueError(f"manifest is not canonical JSON: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _resolved_regular_root(value: str | Path, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    try:
        root = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(f"{label} does not exist: {candidate}") from exc
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label} must be a regular directory")
    return root


@dataclass(frozen=True, slots=True)
class _SourceBinding:
    root: Path
    database_path: Path
    index_path: Path
    manifest_path: Path
    manifest_sha256: str
    receipt: CombinedCumulativeStoreReceipt


@dataclass(frozen=True, slots=True)
class _Experiment:
    artifact: FastRetrievalArtifact
    source: _SourceBinding
    history: HebbianHistoryArtifact
    history_file_sha256: str
    derived: HebbianDerivedStoreReceipt
    derived_manifest_sha256: str
    derived_store_path: Path


def _load_artifact(
    path: Path, expected_sha256: str
) -> FastRetrievalArtifact:
    return load_fast_retrieval_artifact(path, expected_sha256=expected_sha256)


def _validate_source_store(
    source_store: str | Path,
    artifact: FastRetrievalArtifact,
) -> _SourceBinding:
    root = _resolved_regular_root(source_store, "source combined store")
    children = {item.name for item in root.iterdir()}
    if children != _SOURCE_FILES:
        raise ValueError(
            "source combined store has unexpected or missing files: "
            f"{sorted(children)!r}"
        )
    database = root / "memory.db"
    index = root / "hnsw_index.bin"
    manifest = root / COMBINED_CUMULATIVE_STORE_MANIFEST
    for path, label in (
        (database, "source database"),
        (index, "source index"),
        (manifest, "source manifest"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} must be a regular file")
    if any(
        database.with_name(database.name + suffix).exists()
        for suffix in ("-wal", "-shm")
    ):
        raise ValueError("source database retained SQLite sidecars")

    combined, _compilation, _staging, _learning = _read_combined_manifest(root)
    if combined.receipt_sha256 != artifact.combined_store_receipt_sha256:
        raise ValueError("source manifest changed the retrieval combined-store receipt")
    if combined.target_database_sha256 != file_sha256(database):
        raise ValueError("source database does not match its combined-store receipt")
    if combined.target_index_sha256 != file_sha256(index):
        raise ValueError("source index does not match its combined-store receipt")
    if (
        combined.turn_count != artifact.turn_count
        or combined.retrieval_policy_sha256 != artifact.retrieval_policy_sha256
        or combined.retained_request_token_state_bytes != 0
    ):
        raise ValueError("source combined-store dimensions or policy changed")
    return _SourceBinding(
        root=root,
        database_path=database,
        index_path=index,
        manifest_path=manifest,
        manifest_sha256=file_sha256(manifest),
        receipt=combined,
    )


def _eligible_historical_queries(
    database_path: Path, *, max_prompt_tokens: int
) -> tuple[str, ...]:
    """Return only causally eligible user queries through an immutable handle."""

    target = f"{database_path.resolve(strict=True).as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(target, uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        turn_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(turns)")
        }
        chunk_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(chunks)")
        }
        if not {"ordinal", "role", "text"}.issubset(turn_columns) or not {
            "chunk_id",
            "turn_id",
        }.issubset(chunk_columns):
            raise ValueError("source database lacks causal turn/chunk coordinates")
        turns = connection.execute(
            "SELECT ordinal, role, text FROM turns ORDER BY ordinal"
        ).fetchall()
        if [row[0] for row in turns] != list(range(1, len(turns) + 1)):
            raise ValueError("source turn ordinals must be contiguous and one-based")
        embedded = (
            " AND chunks.embedding IS NOT NULL"
            if "embedding" in chunk_columns
            else ""
        )
        counts = {
            int(row[0]): int(row[1])
            for row in connection.execute(
                "SELECT turns.ordinal, COUNT(chunks.chunk_id) FROM turns "
                "LEFT JOIN chunks ON chunks.turn_id = turns.turn_id"
                + embedded
                + " GROUP BY turns.ordinal"
            )
        }
    finally:
        connection.close()

    chunks_seen = 0
    queries: list[str] = []
    for ordinal, role, text in turns:
        exact_text = str(text)
        if (
            str(role) == "user"
            and chunks_seen > 0
            and count_tokens(exact_text) <= max_prompt_tokens
        ):
            queries.append(exact_text)
        chunks_seen += counts.get(int(ordinal), 0)
    result = tuple(dict.fromkeys(queries))
    if not result:
        raise ValueError("source store has no eligible bounded historical queries")
    return result


def _capture_policy(
    args: argparse.Namespace, *, query_embedding_execution_sha256: str
) -> dict[str, object]:
    return {
        "format": "memory-condense.hebbian-capture-policy.v1",
        "retrieval_k": args.retrieval_k,
        "expansion_tokens": args.expansion_tokens,
        "max_prompt_tokens": args.history_max_prompt_tokens,
        "direct_expansion_only": True,
        "event_id_scheme": "causal-user:{ordinal}",
        "capture_point": "after_direct_context_pack_before_current_user_append",
        "exclude_current_and_future_turns": True,
        "query_embedding_model_id": DEFAULT_MODEL_NAME,
        "query_embedding_model_revision": DEFAULT_MODEL_REVISION,
        "query_embedding_checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
        "query_embedding_execution_sha256": query_embedding_execution_sha256,
    }


def _validate_reused_history_policy(
    args: argparse.Namespace, history: HebbianHistoryArtifact
) -> None:
    """Require reuse to mean reuse of the exact requested capture policy."""

    observed = dict(history.capture_policy_payload)
    execution_sha = observed.get("query_embedding_execution_sha256")
    if not _is_digest(execution_sha) or observed != _capture_policy(
        args, query_embedding_execution_sha256=execution_sha
    ):
        raise ValueError(
            "existing history capture policy does not match requested CLI policy"
        )


def _validate_history_args(args: argparse.Namespace) -> None:
    bounds = (
        ("retrieval_k", 1, 64),
        ("expansion_tokens", 1, 8_000),
        ("history_max_prompt_tokens", 1, 512),
        ("embedding_batch_size", 1, 1_000_000),
    )
    for name, minimum, maximum in bounds:
        value = getattr(args, name)
        if type(value) is not int or not minimum <= value <= maximum:
            flag = name.replace("_", "-")
            raise ValueError(f"--{flag} must lie in [{minimum}, {maximum}]")
    if type(args.max_event_nodes) is not int or args.max_event_nodes < 2:
        raise ValueError("--max-event-nodes must be at least two")
    if (
        type(args.new_event_nodes) is not int
        or not 1 <= args.new_event_nodes < args.max_event_nodes
    ):
        raise ValueError("--new-event-nodes must lie in [1, max-event-nodes)")


def _validate_completion_args(args: argparse.Namespace) -> None:
    if type(args.max_prompt_tokens) is not int or not (
        1 <= args.max_prompt_tokens <= 8_000
    ):
        raise ValueError("--max-prompt-tokens must lie in [1, 8000]")
    for name in ("max_new_tokens", "max_concurrency"):
        value = getattr(args, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def _remove_owned_temporary(root: Path, *, parent: Path, prefix: str) -> None:
    """Remove only the exact sibling temporary tree allocated by this runner."""

    try:
        resolved_parent = parent.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        return
    if resolved_root.parent != resolved_parent or not resolved_root.name.startswith(
        prefix
    ):
        raise RuntimeError("refusing to clean an unexpected temporary path")
    shutil.rmtree(resolved_root)


def _seal_staged_history(
    capture: RetrievalAccessCapture,
    *,
    source: _SourceBinding,
    implementation_digest: str,
    environment_digest: str,
    capture_policy: Mapping[str, object],
) -> HebbianHistoryArtifact:
    """Seal the stage capture; kept narrow for the capture-certificate boundary."""

    return seal_hebbian_history_artifact(
        capture,
        source_database_path=source.database_path,
        source_store_receipt_sha256=source.receipt.receipt_sha256,
        implementation_sha256=implementation_digest,
        environment_lock_sha256=environment_digest,
        capture_policy_payload=capture_policy,
    )


def _history_result(
    *,
    artifact: FastRetrievalArtifact,
    source: _SourceBinding,
    history: HebbianHistoryArtifact,
    history_file_sha256: str,
    derived: HebbianDerivedStoreReceipt,
    staging_stats: Mapping[str, int],
    embedding_query_count: int,
    embedding_batch_size: int,
    writes: int,
) -> dict[str, Any]:
    return {
        "phase": "history",
        "retrieval_sha256": artifact.raw_sha256,
        "combined_store_receipt_sha256": source.receipt.receipt_sha256,
        "source_manifest_sha256": source.manifest_sha256,
        "source_database_sha256": history.receipt.source_database_sha256,
        "source_index_sha256": derived.source_index_sha256,
        "history_file_sha256": history_file_sha256,
        "history_artifact_sha256": history.artifact_sha256,
        "history_receipt_sha256": history.receipt.receipt_sha256,
        "direct_capture_sha256": history.receipt.direct_capture_sha256,
        "derived_store_receipt_sha256": derived.receipt_sha256,
        "association_artifact_id": derived.association_artifact_id,
        "event_count": history.receipt.event_count,
        "empty_event_count": history.receipt.empty_event_count,
        "embedding_query_count": embedding_query_count,
        "embedding_api_calls": 1 if writes else 0,
        "embedding_forward_batches": (
            (embedding_query_count + embedding_batch_size - 1)
            // embedding_batch_size
            if writes
            else 0
        ),
        "staging_stats": dict(staging_stats),
        "writes": writes,
        "provider_calls": 0,
        "gold_loaded": False,
        "retained_request_token_state_bytes": 0,
    }


def run_history(args: argparse.Namespace) -> dict[str, Any]:
    """Build and atomically publish one source-bound Hebbian history/store."""

    _validate_history_args(args)
    if args.history_root is not None:
        raise ValueError(
            "--history-root is a consumer option; history publishes --output-root"
        )
    artifact = _load_artifact(
        Path(args.retrieval), str(args.expected_retrieval_sha256)
    )
    source = _validate_source_store(args.source_store, artifact)
    output_root = Path(args.output_root)
    if output_root.is_symlink():
        raise ValueError("--output-root must not be a symbolic link")
    if output_root.exists():
        experiment = _load_experiment(args, history_root=output_root)
        _validate_reused_history_policy(args, experiment.history)
        return _history_result(
            artifact=experiment.artifact,
            source=experiment.source,
            history=experiment.history,
            history_file_sha256=experiment.history_file_sha256,
            derived=experiment.derived,
            staging_stats={},
            embedding_query_count=0,
            embedding_batch_size=args.embedding_batch_size,
            writes=0,
        )

    implementation_digest = implementation_sha256()
    environment_digest = environment_lock_sha256()
    source_database_before = file_sha256(source.database_path)
    source_index_before = file_sha256(source.index_path)
    queries = _eligible_historical_queries(
        source.database_path,
        max_prompt_tokens=args.history_max_prompt_tokens,
    )

    embedder = EmbeddingService(
        model_name=DEFAULT_MODEL_NAME,
        model_revision=DEFAULT_MODEL_REVISION,
        device=str(args.embedding_device),
        batch_size=args.embedding_batch_size,
        verify_checkpoint=True,
    )
    expected_execution_identity = {
        "backend": "sentence-transformers.encode-v1",
        "device": str(args.embedding_device).casefold(),
        "batch_size": args.embedding_batch_size,
        "normalize_embeddings": False,
        "output_dtype": "float32",
    }
    if (
        embedder.model_name != DEFAULT_MODEL_NAME
        or embedder.model_revision != DEFAULT_MODEL_REVISION
        or embedder.checkpoint_sha256 != BGE_M3_CHECKPOINT_SHA256
        or embedder.execution_identity != expected_execution_identity
    ):
        embedder.close()
        raise RuntimeError("embedding service changed its pinned execution identity")
    execution_digest = identity_sha256(expected_execution_identity)
    try:
        raw_vectors = embedder.embed_queries(queries)
    finally:
        embedder.close()
    if (
        getattr(raw_vectors, "shape", None) != (len(queries), DEFAULT_MODEL_DIM)
        or str(getattr(raw_vectors, "dtype", "")) != "float32"
    ):
        raise RuntimeError("embedding batch changed cardinality, dimension, or dtype")
    frozen = FrozenQueryEmbedder(
        {query: vector for query, vector in zip(queries, raw_vectors, strict=True)}
    )

    parent = output_root.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{output_root.name}.history-"
    temporary_root = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    snapshot = temporary_root / "source-memory.db"
    derived_store = temporary_root / "derived-store"
    capture_policy = _capture_policy(
        args,
        query_embedding_execution_sha256=execution_digest,
    )
    capture_sink = RetrievalAccessCaptureSink()
    try:
        shutil.copy2(source.database_path, snapshot)
        if file_sha256(snapshot) != source_database_before:
            raise RuntimeError("private staging snapshot changed source bytes")
        _episodes, staging_stats = stage_causal_store(
            snapshot,
            derived_store,
            frozen,
            expansion_tokens=args.expansion_tokens,
            retrieval_k=args.retrieval_k,
            max_event_nodes=args.max_event_nodes,
            new_event_nodes=args.new_event_nodes,
            max_prompt_tokens=args.history_max_prompt_tokens,
            retrieval_access_capture_sink=capture_sink,
            retrieval_access_capture_policy_sha256=identity_sha256(capture_policy),
        )
        snapshot.unlink()
        history = _seal_staged_history(
            capture_sink.capture,
            source=source,
            implementation_digest=implementation_digest,
            environment_digest=environment_digest,
            capture_policy=capture_policy,
        )
        derived = apply_hebbian_history_to_staged_store(
            derived_store,
            source_database_path=source.database_path,
            source_index_path=source.index_path,
            history=history,
        )
        verify_hebbian_history_artifact(
            history, source_database_path=source.database_path
        )
        verify_hebbian_derived_store(derived_store, expected=derived)
        history_file_sha = _atomic_write_json(
            temporary_root / HISTORY_FILE_NAME, history.payload()
        )
        if (
            implementation_sha256() != implementation_digest
            or environment_lock_sha256() != environment_digest
        ):
            raise RuntimeError(
                "implementation or environment changed during history build"
            )
        source = _validate_source_store(source.root, artifact)
        if (
            file_sha256(source.database_path) != source_database_before
            or file_sha256(source.index_path) != source_index_before
        ):
            raise RuntimeError(
                "source store changed during isolated history construction"
            )
        if output_root.is_symlink() or output_root.exists():
            raise FileExistsError(f"refusing to replace output root: {output_root}")
        os.replace(temporary_root, output_root)
    except BaseException:
        if temporary_root.exists():
            _remove_owned_temporary(temporary_root, parent=parent, prefix=prefix)
        raise

    return _history_result(
        artifact=artifact,
        source=source,
        history=history,
        history_file_sha256=history_file_sha,
        derived=derived,
        staging_stats=staging_stats,
        embedding_query_count=len(queries),
        embedding_batch_size=args.embedding_batch_size,
        writes=1,
    )


def _load_experiment(
    args: argparse.Namespace, *, history_root: Path | None = None
) -> _Experiment:
    artifact = _load_artifact(
        Path(args.retrieval), str(args.expected_retrieval_sha256)
    )
    source = _validate_source_store(args.source_store, artifact)
    selected_root = history_root or args.history_root or args.output_root
    root = _resolved_regular_root(selected_root, "Hebbian history root")
    history_payload, history_file_sha = _read_canonical_json(root / HISTORY_FILE_NAME)
    history = load_hebbian_history_artifact(history_payload)
    verify_hebbian_history_artifact(
        history, source_database_path=source.database_path
    )
    current_implementation = implementation_sha256()
    current_environment = environment_lock_sha256()
    if (
        history.receipt.source_store_receipt_sha256
        != source.receipt.receipt_sha256
        or history.receipt.source_database_sha256
        != source.receipt.target_database_sha256
        or history.receipt.implementation_sha256 != current_implementation
        or history.receipt.environment_lock_sha256 != current_environment
    ):
        raise ValueError(
            "history artifact changed source, implementation, or environment"
        )

    derived_store = _resolved_regular_root(root / "derived-store", "derived store")
    derived_payload, derived_manifest_sha = _read_manifest(
        derived_store / DERIVED_MANIFEST_NAME
    )
    derived = load_hebbian_derived_store_receipt(derived_payload)
    verify_hebbian_derived_store(derived_store, expected=derived)
    if (
        derived.source_database_sha256 != source.receipt.target_database_sha256
        or derived.source_index_sha256 != source.receipt.target_index_sha256
        or derived.source_store_receipt_sha256 != source.receipt.receipt_sha256
        or derived.history_artifact_sha256 != history.artifact_sha256
        or derived.history_receipt_sha256 != history.receipt.receipt_sha256
        or derived.implementation_sha256 != current_implementation
        or derived.environment_lock_sha256 != current_environment
        or derived.retained_request_token_state_bytes != 0
    ):
        raise ValueError("derived store changed its sealed upstream experiment")
    return _Experiment(
        artifact=artifact,
        source=source,
        history=history,
        history_file_sha256=history_file_sha,
        derived=derived,
        derived_manifest_sha256=derived_manifest_sha,
        derived_store_path=derived_store,
    )


def _build_prompts(experiment: _Experiment) -> Any:
    return build_fast_hebbian_prompt_population(
        experiment.artifact,
        experiment.derived_store_path,
        association_artifact_id=experiment.derived.association_artifact_id,
        history_receipt_sha256=experiment.history.receipt.receipt_sha256,
        derived_store_receipt_sha256=experiment.derived.receipt_sha256,
    )


def _prompt_summary(prompts: Any, runtime_population: Any) -> dict[str, Any]:
    statuses = Counter(row.effective_status for row in prompts.question_receipts)
    membership_changes = sum(
        receipt.effective_h1_chunk_ids != receipt.protected_chunk_ids
        for receipt in prompts.question_receipts
    )
    return {
        "stage_id": prompts.stage_id,
        "question_count": len(prompts.question_receipts),
        "effective_statuses": dict(sorted(statuses.items())),
        "replacements": int(statuses.get("replaced", 0)),
        "membership_changes": membership_changes,
        "exact_prompt_budget_rollbacks": int(
            statuses.get("exact_prompt_budget_rollback", 0)
        ),
        "logical_prompt_count": prompts.logical_prompt_count,
        "unique_prompt_count": prompts.unique_prompt_count,
        "hard_prompt_token_cap": runtime_population.max_prompt_token_proxy,
        "max_observed_prompt_token_proxy": max(
            row.prompt_token_proxy for row in runtime_population.ordered_rows
        ),
        "prompt_population_sha256": prompts.prompt_population_sha256,
        "runtime_prompt_population_sha256": (
            runtime_population.prompt_population_sha256
        ),
        "retained_request_token_state_bytes": 0,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _validate_completion_args(args)
    experiment = _load_experiment(args)
    prompts = _build_prompts(experiment)
    runtime_population = preflight_fast_completion_prompts(
        prompts.logical_message_population,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    return {
        "phase": "preflight",
        "retrieval_sha256": experiment.artifact.raw_sha256,
        "history_receipt_sha256": experiment.history.receipt.receipt_sha256,
        "derived_store_receipt_sha256": experiment.derived.receipt_sha256,
        "association_artifact_id": experiment.derived.association_artifact_id,
        "prompt_preflight": _prompt_summary(prompts, runtime_population),
        "writes": 0,
        "provider_calls": 0,
        "gold_loaded": False,
        "retained_request_token_state_bytes": 0,
    }


def _make_provider_client(api_key: str, gateway_url: str) -> Any:
    return _answer_runtime.make_provider_client(api_key, gateway_url)


_experiment_binding = _answer_runtime.experiment_binding
_benchmark_provenance = _answer_runtime.benchmark_provenance
_answer_artifact = _answer_runtime.answer_artifact
_answers_path = _answer_runtime.answers_path
_replay_path = _answer_runtime.replay_path
_checkpoint_path = _answer_runtime.checkpoint_path
_stable_completion_batch_projection = (
    _answer_runtime.stable_completion_batch_projection
)
_replay_answer_journals = _answer_runtime.replay_answer_journals
_validate_answer_replay_pair = _answer_runtime.validate_answer_replay_pair
_score_artifact = _answer_runtime.score_artifact


def run_answer(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    _validate_completion_args(args)
    experiment = _load_experiment(args)
    prompts = _build_prompts(experiment)
    preflight_fast_completion_prompts(
        prompts.logical_message_population,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    unique_calls = prompts.unique_prompt_count
    if not args.enable_provider:
        raise ValueError("answer phase requires the explicit --enable-provider gate")
    if args.authorized_provider_calls != unique_calls:
        raise ValueError(
            "--authorized-provider-calls must exactly equal provider-free "
            f"unique prompt count ({args.authorized_provider_calls} != {unique_calls})"
        )
    api_key = os.environ.get(str(args.api_key_env), "").strip()
    if not api_key:
        raise RuntimeError(f"provider API key is empty: {args.api_key_env}")
    client = _make_provider_client(api_key, str(args.gateway_url))
    binding = _experiment_binding(experiment, prompts)
    provenance = _benchmark_provenance(
        binding,
        caller_model=str(args.caller_model),
        gateway_url=str(args.gateway_url),
    )
    runtime = FastCompletionRuntime(
        checkpoint_dir=_checkpoint_path(args),
        prompt_population=prompts.logical_message_population,
        model=str(args.gateway_model),
        client=client,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        max_concurrency=args.max_concurrency,
        retries=0,
        benchmark_provenance=provenance,
    )
    with runtime:
        batch = runtime.run()
    result = _answer_artifact(
        mode="answer",
        experiment=experiment,
        prompts=prompts,
        completion_batch=batch,
    )
    return result, _atomic_write_json(_answers_path(args), result)


def _read_and_validate_answers(
    experiment: _Experiment,
    prompts: Any,
    path: Path,
    *,
    expected_mode: str,
) -> tuple[dict[str, Any], str]:
    return _answer_runtime.read_and_validate_answers(
        experiment,
        prompts,
        path,
        expected_mode=expected_mode,
        read_canonical_json=_read_canonical_json,
        is_digest=_is_digest,
    )


def run_replay(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.enable_provider:
        raise ValueError("replay phase forbids --enable-provider")
    if args.authorized_provider_calls != 0:
        raise ValueError("replay phase requires --authorized-provider-calls 0")
    experiment = _load_experiment(args)
    prompts = _build_prompts(experiment)
    answers, _answer_sha = _read_and_validate_answers(
        experiment,
        prompts,
        _answers_path(args),
        expected_mode="answer",
    )
    replay_batch = _replay_answer_journals(
        answers=answers,
        prompts=prompts,
        checkpoint_dir=_checkpoint_path(args),
    )
    result = _answer_artifact(
        mode="replay",
        experiment=experiment,
        prompts=prompts,
        completion_batch=replay_batch,
    )
    return result, _atomic_write_json(_replay_path(args), result)


def _load_gold_population(dataset: Path, split: Path) -> Any:
    return _answer_runtime.load_gold_population(dataset, split)


def run_score(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.dataset is None:
        raise ValueError("score phase requires --dataset")
    experiment = _load_experiment(args)
    prompts = _build_prompts(experiment)
    answers, answer_sha = _read_and_validate_answers(
        experiment,
        prompts,
        _answers_path(args),
        expected_mode="answer",
    )
    replay, replay_sha = _read_and_validate_answers(
        experiment,
        prompts,
        _replay_path(args),
        expected_mode="replay",
    )
    _validate_answer_replay_pair(answers, replay)
    journal_replay = _replay_answer_journals(
        answers=answers,
        prompts=prompts,
        checkpoint_dir=_checkpoint_path(args),
    )
    if replay["completion_batch"] != journal_replay.model_dump():
        raise ValueError(
            "replay completion batch differs from immutable provider journals"
        )

    # Gold becomes reachable only after every answer, replay, and journal check.
    sample = _load_gold_population(Path(args.dataset), Path(args.split))
    result = _score_artifact(
        experiment=experiment,
        answers=answers,
        answer_sha256=answer_sha,
        replay_sha256=replay_sha,
        gold_population=sample,
    )
    digest = _atomic_write_json(Path(args.output_root) / "scores.json", result)
    return result, digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("history", "preflight", "answer", "replay", "score"),
        default="preflight",
    )
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--expected-retrieval-sha256", default=ORIGINAL_1M_RETRIEVAL_SHA256
    )
    parser.add_argument("--source-store", type=Path, default=DEFAULT_SOURCE_STORE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--history-root",
        type=Path,
        help="reuse a verified history/derived-store root while writing elsewhere",
    )
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--replay", type=Path)

    parser.add_argument("--retrieval-k", type=int, default=10)
    parser.add_argument("--expansion-tokens", type=int, default=1_600)
    parser.add_argument("--history-max-prompt-tokens", type=int, default=128)
    parser.add_argument("--max-event-nodes", type=int, default=9)
    parser.add_argument("--new-event-nodes", type=int, default=5)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=32)

    parser.add_argument("--max-prompt-tokens", type=int, default=8_000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--gateway-model", default=DEFAULT_GATEWAY_MODEL)
    parser.add_argument("--caller-model", default=DEFAULT_CALLER_MODEL)
    parser.add_argument("--api-key-env", default="LITELLM_KEY")
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--authorized-provider-calls", type=int, default=0)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.phase == "history":
        result = run_history(args)
        print(
            "Fast 1M Hebbian history ready: "
            f"events={result['event_count']}; empty={result['empty_event_count']}; "
            f"embedding_api_calls={result['embedding_api_calls']}; "
            f"embedding_forward_batches={result['embedding_forward_batches']}; "
            f"provider_calls=0; gold_loaded=false",
            flush=True,
        )
        return 0
    if args.phase == "preflight":
        result = run_preflight(args)
        prompt = result["prompt_preflight"]
        print(
            "Fast 1M Hebbian preflight passed: "
            f"questions={prompt['question_count']}; "
            f"logical={prompt['logical_prompt_count']}; "
            f"unique={prompt['unique_prompt_count']}; "
            f"replacements={prompt['replacements']}; provider_calls=0; writes=0",
            flush=True,
        )
        return 0
    if args.phase == "answer":
        result, digest = run_answer(args)
    elif args.phase == "replay":
        result, digest = run_replay(args)
    else:
        result, digest = run_score(args)
        print(
            f"Fast 1M Hebbian scores published ({digest}): "
            f"logical={result['logical_score_count']}",
            flush=True,
        )
        return 0
    usage = result["completion_batch"]["usage"]
    print(
        f"Fast 1M Hebbian {args.phase} published ({digest}): "
        f"logical={result['logical_answer_count']}; "
        f"unique={result['unique_completion_count']}; "
        f"physical={usage['physical_calls']}; "
        f"checkpoint_hits={usage['checkpoint_hits']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANSWER_MANIFEST_FORMAT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_RETRIEVAL",
    "DEFAULT_SOURCE_STORE",
    "HISTORY_FILE_NAME",
    "SCORE_MANIFEST_FORMAT",
    "ZERO_STATE_CONTRACT",
    "build_parser",
    "main",
    "run_answer",
    "run_history",
    "run_preflight",
    "run_replay",
    "run_score",
]
