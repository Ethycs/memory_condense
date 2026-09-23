"""Provider-free hot-retrieval assay for the original dev1M memory.

The five commands are deliberately process-separated:

``export-probes``
    Reads the pinned development dataset once and publishes only gold-free
    retrieval/prompt questions.
``compile-base``
    Reads no questions. It compiles the exact 7,895 x 1,024 normalized dense
    address matrix and a text-free chunk manifest from the sealed source.
``run``
    Reads only the compiled base, source store, and sealed probes. It compares
    BM25-8, exact-dense-8, separately budgeted source-neighborhood and
    temporal-event lanes, and their protected post-selection union, then emits
    raw provider-ready evidence with zero Qwen/provider calls.
``replay``
    Repeats the gold-blind selection and requires byte-identical semantics.
``score``
    Joins gold only after ``selection.json`` and its replay are sealed.

Timing is isolated in separate runtime artifacts, so clock noise never enters
the semantic selection digest.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
    tokenizer_proxy_identity,
)
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval._retrieval_qa_prompt import (
    RESPONDER_OUTPUT_TOKEN_RESERVE,
    build_qa_prompt,
)
from memory_condense.eval.answer_value_coverage import (
    answer_value_component_coverage,
    best_f1,
    contains_answer,
)
from memory_condense.eval.context_stress import transcript_tokens
from memory_condense.eval.recall_guarded_cumulative_1m import (
    ORIGINAL_ORDERED_QUESTION_IDS,
    ORIGINAL_QUESTIONS,
    ORIGINAL_TRANSCRIPT_TOKENS,
    ORIGINAL_TURNS,
    load_original_population,
    population_identity_payload,
    population_identity_sha256,
)
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    EmbeddingService,
)
from memory_condense.persistence.db import INDEXED_CHUNK_SQL, TURN_SOURCE_ID_SQL, Database
from memory_condense.search.hot_retrieval import (
    ExactDenseAddressIndex,
    RankedChunkAddress,
)
from memory_condense.search.hot_lexical import ResidentBM25Index
from memory_condense.search.indexes.lexical import LexicalIndex, tokenize
from memory_condense.search.indexes.retrieval_models import hydrate_chunk_result
from memory_condense.search.post_selection_lane_union import (
    RankedEvidenceLane,
    post_selection_lane_union,
)
from memory_condense.search.source_neighborhood import (
    SourceChunkMetadata,
    SourceNeighborhood,
    SourceNeighborhoodIndex,
)
from memory_condense.search.temporal_enumeration import (
    EventChunkFeatures,
    TEMPORAL_EVENT_LANE_BUDGET,
    TemporalEvidenceWindow,
    TemporalEnumerationPlan,
    compile_event_chunk_features,
    plan_temporal_enumeration,
    resolve_temporal_evidence_window,
)


PROBE_FORMAT = "memory-condense-hot-retrieval-1m-probes-v6"
COMPILED_CHUNKS_FORMAT = "memory-condense-hot-retrieval-1m-chunks-v6"
COMPILED_FORMAT = "memory-condense-hot-retrieval-1m-compiled-v6"
COMPILE_RUNTIME_FORMAT = "memory-condense-hot-retrieval-1m-compile-runtime-v6"
SELECTION_FORMAT = "memory-condense-hot-retrieval-1m-selection-v6"
RUNTIME_FORMAT = "memory-condense-hot-retrieval-1m-runtime-v6"
SCORE_FORMAT = "memory-condense-hot-retrieval-1m-score-v6"
REPLAY_FORMAT = "memory-condense-hot-retrieval-1m-replay-v6"

RETRIEVAL_QUERY_FORM = "plain_question_with_dated_responder_prompt"
MIN_LATENCY_REPEATS = 20

EXPECTED_POPULATION_SHA256 = (
    "fa9a06ebd103d87086943cfa94091bdf607fe07874bc871e465aad409b85ca18"
)
EXPECTED_SOURCE_SELECTION_SHA256 = (
    "16756d07d7ada13fec52387f9be585bcc24a5454499f33759b12d71a5d980f5b"
)
EXPECTED_SOURCE_RECEIPT_SHA256 = (
    "92c764d7fabfbeef9d068fc52210148eb44b4613530d987f2c5856baeda5bb45"
)
EXPECTED_SOURCE_DATABASE_SHA256 = (
    "222f2b3ff39d9e0b9ed4a04a75b60ef1edeb1ada3a7b67b74cf2b354fea88dbc"
)
EXPECTED_SOURCE_INDEX_SHA256 = (
    "5999fdc048ca02c1936957cff985f53808c9dc49b21b7101d2a2db51c78d561c"
)
EXPECTED_CHUNKS = 7_895

DEFAULT_SPLIT = Path(
    "docs/10 - Research Log/data/longmemeval-95-target-split-v2.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-retrieval-linked-windowed-development-20260905"
)
DEFAULT_COMPILED_NAME = "compiled.json"
DEFAULT_COMPILED_CHUNKS_NAME = "compiled-chunks.json"
DEFAULT_DENSE_NAME = "compiled-dense-f32.npy"
DEFAULT_PROBES_NAME = "probes.json"
DEFAULT_SELECTION_NAME = "selection.json"
DEFAULT_RUNTIME_NAME = "runtime.json"
DEFAULT_SCORE_NAME = "scores.json"
DEFAULT_REPLAY_NAME = "replay.json"
DEFAULT_COMPILE_RUNTIME_NAME = "compile-runtime.json"

ARM_IDS = (
    "a0_bm25",
    "a1_exact_dense",
    "a2_source_neighborhood",
    "a2_temporal_event",
    "a3_protected_union",
)
SOURCE_NEIGHBOR_LANE_BUDGET = 8
_FORBIDDEN_PROBE_KEYS = frozenset(
    {"answer", "answers", "gold", "reference", "evidence", "evidence_sources"}
)
_NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?!\w)")
_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|"
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> str:
    """Publish immutable bytes and a digest sidecar, accepting exact replay."""

    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if file_sha256(path) != digest:
            raise FileExistsError(f"refusing to replace another artifact: {path}")
    else:
        descriptor, raw_temp = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(raw_temp)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_payload = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar.exists():
        if sidecar.read_bytes() != sidecar_payload:
            raise FileExistsError(f"refusing to replace another digest: {sidecar}")
    else:
        sidecar.write_bytes(sidecar_payload)
    return digest


def _atomic_write_json(path: Path, value: object) -> str:
    return _atomic_write_bytes(path, _canonical_json_bytes(value))


def _read_json_artifact(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != _canonical_json_bytes(payload):
        raise ValueError(f"artifact is not a canonical JSON object: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if not sidecar.is_file() or sidecar.read_bytes() != expected:
        raise ValueError(f"artifact digest sidecar is missing or invalid: {path}")
    return payload, digest


def _safe_relative(root: Path, relative: str, *, label: str) -> Path:
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError(f"{label} must remain below its artifact root")
    target = (root / part).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes its artifact root")
    return target


def _source_paths(source_selection_path: Path) -> tuple[dict[str, Any], str, Path, Path]:
    selection, digest = _read_json_artifact(source_selection_path)
    if digest != EXPECTED_SOURCE_SELECTION_SHA256:
        raise ValueError("source selection is not the sealed dev1M selection")
    required = {
        "receipt_sha256": EXPECTED_SOURCE_RECEIPT_SHA256,
        "database_sha256": EXPECTED_SOURCE_DATABASE_SHA256,
        "index_sha256": EXPECTED_SOURCE_INDEX_SHA256,
        "chunk_count": EXPECTED_CHUNKS,
        "turn_count": ORIGINAL_TURNS,
    }
    if any(selection.get(key) != value for key, value in required.items()):
        raise ValueError("source selection changed its sealed dev1M coordinates")
    embedding = selection.get("embedding_identity")
    if not isinstance(embedding, Mapping) or (
        embedding.get("model_id") != DEFAULT_MODEL_NAME
        or embedding.get("model_revision") != DEFAULT_MODEL_REVISION
        or embedding.get("checkpoint_sha256") != BGE_M3_CHECKPOINT_SHA256
        or embedding.get("dimension") != DEFAULT_MODEL_DIM
    ):
        raise ValueError("source selection changed its pinned BGE-M3 identity")
    store_entry = selection.get("selected_store_entry")
    if not isinstance(store_entry, str) or not store_entry:
        raise ValueError("source selection omitted selected_store_entry")
    # The published selection sits at the campaign root while both selected
    # entries are explicitly relative to its ``source-current`` artifact
    # directory (the same convention used by the original source validator).
    source_root = source_selection_path.parent / "source-current"
    if not source_root.is_dir():
        raise FileNotFoundError("source-current artifact root is missing")
    store_root = _safe_relative(
        source_root,
        store_entry,
        label="selected_store_entry",
    )
    database_path = store_root / "store" / "memory.db"
    index_path = store_root / "store" / "hnsw_index.bin"
    if not database_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("sealed source store is missing its database or index")
    if file_sha256(database_path) != EXPECTED_SOURCE_DATABASE_SHA256:
        raise RuntimeError("sealed source database changed")
    if file_sha256(index_path) != EXPECTED_SOURCE_INDEX_SHA256:
        raise RuntimeError("sealed source HNSW image changed")
    return selection, digest, database_path, index_path


def _implementation_files() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "tools/assay_hot_retrieval_1m.py",
        "src/memory_condense/modeling/embedding.py",
        "src/memory_condense/persistence/db.py",
        "src/memory_condense/search/hot_retrieval.py",
        "src/memory_condense/search/hot_lexical.py",
        "src/memory_condense/search/post_selection_lane_union.py",
        "src/memory_condense/search/source_neighborhood.py",
        "src/memory_condense/search/temporal_enumeration.py",
        "src/memory_condense/search/indexes/lexical.py",
        "src/memory_condense/search/indexes/retrieval_models.py",
        "src/memory_condense/domain/_tokenizer.py",
        "src/memory_condense/domain/discourse.py",
        "src/memory_condense/eval/answer_value_coverage.py",
        "src/memory_condense/eval/_retrieval_qa_prompt.py",
    )
    return {relative: file_sha256(root / relative) for relative in relative_paths}


def _implementation_identity() -> dict[str, Any]:
    files = _implementation_files()
    return {
        "format": "memory-condense-hot-retrieval-implementation-v6",
        "files": files,
        "sha256": identity_sha256(
            [{"path": path, "sha256": digest} for path, digest in files.items()]
        ),
    }


def _assert_gold_free_probe_rows(rows: object) -> None:
    if not isinstance(rows, list):
        raise ValueError("probe questions must be a list")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("probe question must be an object")
        forbidden = _FORBIDDEN_PROBE_KEYS & {str(key).casefold() for key in row}
        if forbidden:
            raise ValueError(f"probe question contains forbidden fields: {sorted(forbidden)}")


def export_probes(*, dataset: Path, split_manifest: Path, output_root: Path) -> str:
    sample = load_original_population(dataset, split_manifest)
    population_sha = population_identity_sha256(sample)
    if population_sha != EXPECTED_POPULATION_SHA256:
        raise RuntimeError("dev1M population identity changed")
    rows = []
    for ordinal, question in enumerate(sample.questions):
        retrieval_query = str(question.question)
        prompt_question = str(question.dated_question)
        # Search receives only the benchmark question. The dated presentation
        # prompt is retained independently for the final responder.
        row = {
            "ordinal": ordinal,
            "question_id": question.question_id,
            "retrieval_query": retrieval_query,
            "prompt_question": prompt_question,
            "retrieval_query_sha256": quote_sha256(retrieval_query),
            "prompt_question_sha256": quote_sha256(prompt_question),
        }
        row["probe_sha256"] = identity_sha256(row)
        rows.append(row)
    _assert_gold_free_probe_rows(rows)
    artifact = {
        "format": PROBE_FORMAT,
        "status": "sealed_gold_free_development_probes",
        "population_identity": population_identity_payload(sample),
        "population_identity_sha256": population_sha,
        "transcript_tokens": transcript_tokens(sample),
        "turn_count": len(sample.turns),
        "question_count": len(sample.questions),
        "ordered_question_ids": [row["question_id"] for row in rows],
        "retrieval_query_form": RETRIEVAL_QUERY_FORM,
        "questions": rows,
        "gold_fields_present": False,
        "provider_calls": 0,
    }
    path = output_root / DEFAULT_PROBES_NAME
    digest = _atomic_write_json(path, artifact)
    print(f"Gold-free probes published: {path} ({digest})", flush=True)
    return digest


def _publish_numpy(path: Path, matrix: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, matrix, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        payload_digest = file_sha256(temporary)
        if path.exists():
            if file_sha256(path) != payload_digest:
                raise FileExistsError(f"refusing to replace another artifact: {path}")
            temporary.unlink()
        else:
            os.replace(temporary, path)
        sidecar = path.with_name(path.name + ".sha256")
        sidecar_payload = f"{payload_digest}  {path.name}\n".encode("ascii")
        if sidecar.exists():
            if sidecar.read_bytes() != sidecar_payload:
                raise FileExistsError(f"refusing to replace another digest: {sidecar}")
        else:
            sidecar.write_bytes(sidecar_payload)
        return payload_digest
    finally:
        if temporary.exists():
            temporary.unlink()


def _score_float(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("retrieval scores must be finite")
    if number == 0.0:
        number = 0.0
    return format(number, ".9g")


def compile_base(*, source_selection: Path, output_root: Path) -> str:
    compiled_path = output_root / DEFAULT_COMPILED_NAME
    source, selection_sha, database_path, index_path = _source_paths(source_selection)
    if compiled_path.exists():
        loaded = _load_compiled(output_root)
        if (
            loaded.artifact["source_binding"].get("source_selection_sha256")
            != selection_sha
            or loaded.artifact.get("implementation") != _implementation_identity()
        ):
            raise ValueError(
                "existing compiled base belongs to another source or implementation"
            )
        _, digest = _read_json_artifact(compiled_path)
        print(f"Compiled base verified: {compiled_path} ({digest})", flush=True)
        return digest

    started = time.perf_counter_ns()
    rows: list[dict[str, Any]] = []
    matrix = np.empty((EXPECTED_CHUNKS, DEFAULT_MODEL_DIM), dtype=np.float32)
    with Database(database_path, read_only=True) as database:
        lexical_stats = LexicalIndex(database).stats()
        cursor = database.execute(
            "SELECT c.chunk_id, c.embedding, c.text, c.token_count, c.turn_id, "
            + TURN_SOURCE_ID_SQL
            + ", t.role, t.created_at, t.ordinal, c.start_char FROM chunks AS c "
            "JOIN turns AS t ON t.turn_id = c.turn_id WHERE "
            + INDEXED_CHUNK_SQL
            + " ORDER BY c.chunk_id"
        )
        for row_index, raw in enumerate(cursor):
            if row_index >= EXPECTED_CHUNKS:
                raise RuntimeError("source contains more chunks than its receipt")
            chunk_id = str(raw[0])
            vector = np.frombuffer(raw[1], dtype=np.float32)
            if vector.shape != (DEFAULT_MODEL_DIM,) or not np.isfinite(vector).all():
                raise RuntimeError(f"chunk {chunk_id!r} has an invalid dense address")
            norm = float(np.linalg.norm(vector))
            if norm <= 0.0:
                raise RuntimeError(f"chunk {chunk_id!r} has a zero dense address")
            matrix[row_index] = vector / norm
            role = str(raw[6])
            event_features = compile_event_chunk_features(role=role, text=str(raw[2]))
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "token_count": int(raw[3]),
                    "turn_id": str(raw[4]),
                    "source_id": str(raw[5]),
                    "role": role,
                    "created_at": None if raw[7] is None else str(raw[7]),
                    "ordinal": int(raw[8]),
                    "start_char": int(raw[9]),
                    "event_first_person": event_features.first_person,
                    "event_fixed_completed": event_features.fixed_completed_event,
                    "event_ed_verbs": list(event_features.ed_verbs),
                }
            )
    if len(rows) != EXPECTED_CHUNKS:
        raise RuntimeError(f"source yielded {len(rows)} chunks, expected {EXPECTED_CHUNKS}")
    chunk_ids = [row["chunk_id"] for row in rows]
    if chunk_ids != sorted(set(chunk_ids)):
        raise RuntimeError("compiled chunk IDs are not unique and ordered")

    dense_path = output_root / DEFAULT_DENSE_NAME
    dense_sha = _publish_numpy(dense_path, matrix)
    chunks_artifact = {
        "format": COMPILED_CHUNKS_FORMAT,
        "source_receipt_sha256": EXPECTED_SOURCE_RECEIPT_SHA256,
        "chunk_count": len(rows),
        "chunk_sequence_sha256": identity_sha256(chunk_ids),
        "rows": rows,
        "gold_fields_present": False,
    }
    chunks_path = output_root / DEFAULT_COMPILED_CHUNKS_NAME
    chunks_sha = _atomic_write_json(chunks_path, chunks_artifact)
    implementation = _implementation_identity()
    compiled = {
        "format": COMPILED_FORMAT,
        "status": "query_independent_provider_free_addresses",
        "source_binding": {
            "source_selection_sha256": selection_sha,
            "source_receipt_sha256": EXPECTED_SOURCE_RECEIPT_SHA256,
            "database_sha256": EXPECTED_SOURCE_DATABASE_SHA256,
            "database_bytes": database_path.stat().st_size,
            "index_sha256": EXPECTED_SOURCE_INDEX_SHA256,
            "index_bytes": index_path.stat().st_size,
            "turn_count": ORIGINAL_TURNS,
            "chunk_count": EXPECTED_CHUNKS,
        },
        "embedding_identity": dict(source["embedding_identity"]),
        "dense_address": {
            "path": DEFAULT_DENSE_NAME,
            "sha256": dense_sha,
            "bytes": dense_path.stat().st_size,
            "dtype": "float32",
            "shape": [EXPECTED_CHUNKS, DEFAULT_MODEL_DIM],
            "row_l2_normalized": True,
            "chunk_sequence_sha256": chunks_artifact["chunk_sequence_sha256"],
        },
        "chunk_manifest": {
            "path": DEFAULT_COMPILED_CHUNKS_NAME,
            "sha256": chunks_sha,
            "bytes": chunks_path.stat().st_size,
            "contains_raw_text": False,
        },
        "lexical_address": {
            "durable_backend": "sqlite-okapi-bm25",
            "query_backend": "resident-exact-okapi-bm25",
            "source_database_sha256": EXPECTED_SOURCE_DATABASE_SHA256,
            "stats": {key: _score_float(value) for key, value in lexical_stats.items()},
        },
        "temporal_event_address": {
            "algorithm": "implicit-temporal-event-set-v1",
            "budget": TEMPORAL_EVENT_LANE_BUDGET,
            "query_independent_chunk_features": True,
            "raw_text_retained": False,
        },
        "source_neighborhood_address": {
            "algorithm": "same-source-adjacent-turn-round-robin-v1",
            "budget": SOURCE_NEIGHBOR_LANE_BUDGET,
            "turn_order": "ordinal_then_turn_id",
            "within_turn_order": "start_char_then_chunk_id",
            "query_independent_chunk_coordinates": True,
            "raw_text_retained": False,
        },
        "implementation": implementation,
        "question_count": 0,
        "gold_fields_present": False,
        "qwen_calls": 0,
        "provider_calls": 0,
    }
    compiled_sha = _atomic_write_json(compiled_path, compiled)
    elapsed = time.perf_counter_ns() - started
    runtime = {
        "format": COMPILE_RUNTIME_FORMAT,
        "compiled_sha256": compiled_sha,
        "elapsed_ns": elapsed,
        "chunks_per_second": EXPECTED_CHUNKS / (elapsed / 1_000_000_000),
        "derived_index_bytes": dense_path.stat().st_size + chunks_path.stat().st_size,
        "qwen_calls": 0,
        "provider_calls": 0,
    }
    _atomic_write_json(output_root / DEFAULT_COMPILE_RUNTIME_NAME, runtime)
    print(f"Compiled base published: {compiled_path} ({compiled_sha})", flush=True)
    return compiled_sha


@dataclass(slots=True)
class LoadedCompiledBase:
    artifact: dict[str, Any]
    artifact_sha256: str
    chunks_artifact: dict[str, Any]
    metadata_by_id: dict[str, dict[str, Any]]
    dense_index: ExactDenseAddressIndex
    source_neighborhood_index: SourceNeighborhoodIndex


def _load_compiled(output_root: Path) -> LoadedCompiledBase:
    path = output_root / DEFAULT_COMPILED_NAME
    artifact, digest = _read_json_artifact(path)
    if artifact.get("format") != COMPILED_FORMAT:
        raise ValueError("compiled artifact has another format")
    source = artifact.get("source_binding")
    dense = artifact.get("dense_address")
    chunks_ref = artifact.get("chunk_manifest")
    lexical = artifact.get("lexical_address")
    temporal_event = artifact.get("temporal_event_address")
    source_neighborhood = artifact.get("source_neighborhood_address")
    if not all(
        isinstance(value, Mapping)
        for value in (
            source,
            dense,
            chunks_ref,
            lexical,
            temporal_event,
            source_neighborhood,
        )
    ):
        raise ValueError("compiled artifact omitted a required binding")
    if (
        source.get("source_receipt_sha256") != EXPECTED_SOURCE_RECEIPT_SHA256
        or source.get("chunk_count") != EXPECTED_CHUNKS
        or dense.get("shape") != [EXPECTED_CHUNKS, DEFAULT_MODEL_DIM]
        or dense.get("dtype") != "float32"
        or dense.get("row_l2_normalized") is not True
        or lexical.get("durable_backend") != "sqlite-okapi-bm25"
        or lexical.get("query_backend") != "resident-exact-okapi-bm25"
        or lexical.get("source_database_sha256") != EXPECTED_SOURCE_DATABASE_SHA256
        or temporal_event.get("algorithm") != "implicit-temporal-event-set-v1"
        or temporal_event.get("budget") != TEMPORAL_EVENT_LANE_BUDGET
        or temporal_event.get("query_independent_chunk_features") is not True
        or temporal_event.get("raw_text_retained") is not False
        or source_neighborhood.get("algorithm")
        != "same-source-adjacent-turn-round-robin-v1"
        or source_neighborhood.get("budget") != SOURCE_NEIGHBOR_LANE_BUDGET
        or source_neighborhood.get("turn_order") != "ordinal_then_turn_id"
        or source_neighborhood.get("within_turn_order")
        != "start_char_then_chunk_id"
        or source_neighborhood.get("query_independent_chunk_coordinates")
        is not True
        or source_neighborhood.get("raw_text_retained") is not False
        or artifact.get("gold_fields_present") is not False
        or artifact.get("provider_calls") != 0
        or artifact.get("qwen_calls") != 0
    ):
        raise ValueError("compiled artifact changed its dev1M contract")
    dense_path = _safe_relative(output_root, str(dense.get("path", "")), label="dense path")
    chunks_path = _safe_relative(
        output_root,
        str(chunks_ref.get("path", "")),
        label="chunk manifest path",
    )
    if file_sha256(dense_path) != dense.get("sha256"):
        raise RuntimeError("compiled dense matrix changed")
    chunks_artifact, chunks_sha = _read_json_artifact(chunks_path)
    if chunks_sha != chunks_ref.get("sha256") or (
        chunks_artifact.get("format") != COMPILED_CHUNKS_FORMAT
        or chunks_artifact.get("source_receipt_sha256")
        != EXPECTED_SOURCE_RECEIPT_SHA256
        or chunks_artifact.get("chunk_count") != EXPECTED_CHUNKS
        or chunks_artifact.get("gold_fields_present") is not False
    ):
        raise ValueError("compiled chunk manifest changed")
    rows = chunks_artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CHUNKS:
        raise ValueError("compiled chunk manifest has the wrong population")
    if any(
        not isinstance(row, Mapping)
        or row.get("event_first_person") not in (True, False)
        or row.get("event_fixed_completed") not in (True, False)
        or not isinstance(row.get("event_ed_verbs"), list)
        or isinstance(row.get("ordinal"), bool)
        or not isinstance(row.get("ordinal"), int)
        or int(row["ordinal"]) < 0
        or isinstance(row.get("start_char"), bool)
        or not isinstance(row.get("start_char"), int)
        or int(row["start_char"]) < 0
        or not isinstance(row.get("source_id"), str)
        or not row.get("source_id")
        or not isinstance(row.get("turn_id"), str)
        or not row.get("turn_id")
        or any(not isinstance(verb, str) or not verb for verb in row["event_ed_verbs"])
        for row in rows
    ):
        raise ValueError("compiled temporal event features changed")
    chunk_ids = [str(row.get("chunk_id", "")) for row in rows if isinstance(row, Mapping)]
    if len(chunk_ids) != EXPECTED_CHUNKS or chunk_ids != sorted(set(chunk_ids)):
        raise ValueError("compiled chunk order changed")
    if identity_sha256(chunk_ids) != chunks_artifact.get("chunk_sequence_sha256") or (
        chunks_artifact.get("chunk_sequence_sha256")
        != dense.get("chunk_sequence_sha256")
    ):
        raise ValueError("compiled dense/chunk row binding changed")
    index = ExactDenseAddressIndex.open(chunk_ids, dense_path)
    metadata = {str(row["chunk_id"]): dict(row) for row in rows}
    neighborhood_index = SourceNeighborhoodIndex(
        [
            SourceChunkMetadata(
                chunk_id=str(row["chunk_id"]),
                source_id=str(row["source_id"]),
                turn_id=str(row["turn_id"]),
                ordinal=int(row["ordinal"]),
                start_char=int(row["start_char"]),
            )
            for row in rows
        ]
    )
    return LoadedCompiledBase(
        artifact,
        digest,
        chunks_artifact,
        metadata,
        index,
        neighborhood_index,
    )


def _load_probes(output_root: Path) -> tuple[dict[str, Any], str]:
    artifact, digest = _read_json_artifact(output_root / DEFAULT_PROBES_NAME)
    rows = artifact.get("questions")
    _assert_gold_free_probe_rows(rows)
    if (
        artifact.get("format") != PROBE_FORMAT
        or artifact.get("population_identity_sha256") != EXPECTED_POPULATION_SHA256
        or artifact.get("transcript_tokens") != ORIGINAL_TRANSCRIPT_TOKENS
        or artifact.get("turn_count") != ORIGINAL_TURNS
        or artifact.get("question_count") != ORIGINAL_QUESTIONS
        or artifact.get("ordered_question_ids") != list(ORIGINAL_ORDERED_QUESTION_IDS)
        or artifact.get("retrieval_query_form") != RETRIEVAL_QUERY_FORM
        or artifact.get("gold_fields_present") is not False
        or artifact.get("provider_calls") != 0
        or not isinstance(rows, list)
        or len(rows) != ORIGINAL_QUESTIONS
    ):
        raise ValueError("probe manifest changed its exact dev1M population")
    for ordinal, (row, question_id) in enumerate(
        zip(rows, ORIGINAL_ORDERED_QUESTION_IDS, strict=True)
    ):
        if (
            row.get("ordinal") != ordinal
            or row.get("question_id") != question_id
            or row.get("retrieval_query_sha256")
            != quote_sha256(str(row.get("retrieval_query", "")))
            or row.get("prompt_question_sha256")
            != quote_sha256(str(row.get("prompt_question", "")))
        ):
            raise ValueError(f"probe row {ordinal} changed its query binding")
        expected = dict(row)
        observed_probe_sha = expected.pop("probe_sha256", None)
        if observed_probe_sha != identity_sha256(expected):
            raise ValueError(f"probe row {ordinal} changed its receipt")
    return artifact, digest


def _cuda_synchronize(device: str) -> None:
    if not str(device).casefold().startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(device))
    except ImportError:
        return


def _reset_cuda_peak(device: str) -> None:
    if not str(device).casefold().startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.device(device))
    except ImportError:
        return


def _cuda_memory(device: str) -> dict[str, int | None]:
    if not str(device).casefold().startswith("cuda"):
        return {"allocated_bytes": None, "reserved_bytes": None, "peak_allocated_bytes": None}
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocated_bytes": None, "reserved_bytes": None, "peak_allocated_bytes": None}
        selected = torch.device(device)
        _cuda_synchronize(device)
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(selected)),
            "reserved_bytes": int(torch.cuda.memory_reserved(selected)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(selected)),
        }
    except ImportError:
        return {"allocated_bytes": None, "reserved_bytes": None, "peak_allocated_bytes": None}


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def _query_features(query: str) -> dict[str, int]:
    started = time.perf_counter_ns()
    terms = tokenize(query)
    values = {
        "bm25_term_count": len(terms),
        "bm25_unique_term_count": len(set(terms)),
        "number_count": len(_NUMBER_RE.findall(query)),
        "date_count": len(_DATE_RE.findall(query)),
    }
    values["elapsed_ns"] = time.perf_counter_ns() - started
    return values


def _timed_bm25(
    lexical: ResidentBM25Index,
    query: str,
    candidates: int,
) -> tuple[tuple[RankedChunkAddress, ...], int]:
    started = time.perf_counter_ns()
    hits = lexical.search(query, limit=candidates)
    elapsed = time.perf_counter_ns() - started
    return (
        tuple(
            RankedChunkAddress(chunk_id=chunk_id, score=score, route="bm25")
            for chunk_id, score in hits
        ),
        elapsed,
    )


def _compiled_event_features(metadata: Mapping[str, Any]) -> EventChunkFeatures:
    raw_verbs = metadata.get("event_ed_verbs")
    if not isinstance(raw_verbs, list) or any(
        not isinstance(value, str) or not value for value in raw_verbs
    ):
        if raw_verbs != []:
            raise ValueError("compiled event verbs changed")
    return EventChunkFeatures(
        first_person=metadata.get("event_first_person") is True,
        fixed_completed_event=metadata.get("event_fixed_completed") is True,
        ed_verbs=tuple(raw_verbs or ()),
    )


def _timed_temporal_events(
    lexical: ResidentBM25Index,
    plan: TemporalEnumerationPlan,
    metadata_by_id: Mapping[str, Mapping[str, Any]],
    *,
    candidate_limit: int | None = None,
    evidence_window: TemporalEvidenceWindow | None = None,
) -> tuple[tuple[RankedChunkAddress, ...], int]:
    started = time.perf_counter_ns()
    selected: list[RankedChunkAddress] = []
    limit = plan.budget if candidate_limit is None else candidate_limit
    if limit < plan.budget:
        raise ValueError("event candidate limit cannot be below the lane budget")
    if plan.active:
        # The event lane scans the resident corpus, not the development
        # fixture's globally pinned row count.  Using the compiled metadata
        # population keeps the exact v6 policy reusable across independently
        # sealed 1M-token namespaces.
        for chunk_id, score in lexical.search(
            plan.search_query,
            limit=len(metadata_by_id),
        ):
            metadata = metadata_by_id.get(chunk_id)
            if metadata is None:
                raise RuntimeError(f"event candidate is absent from metadata: {chunk_id}")
            if not _compiled_event_features(metadata).admits(plan):
                continue
            if evidence_window is not None and not evidence_window.admits(
                metadata.get("created_at")
            ):
                continue
            selected.append(
                RankedChunkAddress(
                    chunk_id=chunk_id,
                    score=score,
                    route="temporal_event",
                )
            )
            if len(selected) >= limit:
                break
    return tuple(selected), time.perf_counter_ns() - started


def _timed_lexical_routes(
    lexical: ResidentBM25Index,
    query: str,
    candidates: int,
    event_plan: TemporalEnumerationPlan,
    metadata_by_id: Mapping[str, Mapping[str, Any]],
    event_candidate_limit: int,
    evidence_window: TemporalEvidenceWindow | None,
) -> tuple[
    tuple[RankedChunkAddress, ...],
    int,
    tuple[RankedChunkAddress, ...],
    int,
]:
    lexical_hits, bm25_ns = _timed_bm25(lexical, query, candidates)
    event_hits, event_ns = _timed_temporal_events(
        lexical,
        event_plan,
        metadata_by_id,
        candidate_limit=event_candidate_limit,
        evidence_window=evidence_window,
    )
    return lexical_hits, bm25_ns, event_hits, event_ns


def _filter_temporal_window(
    addresses: Sequence[RankedChunkAddress],
    evidence_window: TemporalEvidenceWindow | None,
    metadata_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[RankedChunkAddress, ...], tuple[str, ...], int]:
    """Apply a question-derived time range before independent lane selection."""

    started = time.perf_counter_ns()
    if evidence_window is None:
        return tuple(addresses), (), time.perf_counter_ns() - started
    retained: list[RankedChunkAddress] = []
    excluded: list[str] = []
    for address in addresses:
        metadata = metadata_by_id.get(address.chunk_id)
        if metadata is None:
            raise RuntimeError(
                f"temporal candidate is absent from metadata: {address.chunk_id}"
            )
        if evidence_window.admits(metadata.get("created_at")):
            retained.append(address)
        else:
            excluded.append(address.chunk_id)
    return tuple(retained), tuple(excluded), time.perf_counter_ns() - started


def _timed_source_neighborhood(
    index: SourceNeighborhoodIndex,
    seed_addresses: Sequence[RankedChunkAddress],
) -> tuple[tuple[RankedChunkAddress, ...], SourceNeighborhood, int]:
    """Expand already-selected addresses through precompiled source topology."""

    started = time.perf_counter_ns()
    neighborhood = index.neighbors(
        [address.chunk_id for address in seed_addresses]
    )
    candidate_count = len(neighborhood.candidates)
    hits = tuple(
        RankedChunkAddress(
            chunk_id=candidate.chunk_id,
            score=float(candidate_count - rank),
            route="source_neighborhood",
        )
        for rank, candidate in enumerate(neighborhood.candidates)
    )
    return hits, neighborhood, time.perf_counter_ns() - started


def _source_neighborhood_audit(
    neighborhood: SourceNeighborhood,
) -> dict[str, Any]:
    links_by_candidate: dict[str, list[dict[str, str]]] = {}
    for link in neighborhood.links:
        links_by_candidate.setdefault(link.linked_chunk_id, []).append(
            {
                "seed_chunk_id": link.seed_chunk_id,
                "source_id": link.source_id,
                "direction": link.direction,
            }
        )
    return {
        "source_order": list(neighborhood.source_order),
        "seed_groups": [
            {
                "source_id": group.source_id,
                "seed_chunk_ids": list(group.seed_chunk_ids),
            }
            for group in neighborhood.seed_groups
        ],
        "links_by_candidate": links_by_candidate,
    }


def _candidate_row(
    address: RankedChunkAddress,
    metadata_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = metadata_by_id.get(address.chunk_id)
    if metadata is None:
        raise RuntimeError(f"candidate is absent from compiled metadata: {address.chunk_id}")
    return {
        "chunk_id": address.chunk_id,
        "source_id": str(metadata["source_id"]),
        "score": _score_float(address.score),
        "route": address.route,
    }


def _hydrate_addresses(
    database: Database,
    addresses: Sequence[RankedChunkAddress],
    metadata_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[Any], int]:
    started = time.perf_counter_ns()
    hydrated = []
    for address in addresses:
        result = hydrate_chunk_result(
            database,
            address.chunk_id,
            score=address.score,
            dense_score=(address.score if address.route == "exact_dense" else None),
            lexical_score=(address.score if address.route == "bm25" else None),
            route=address.route,
        )
        if result is None:
            raise RuntimeError(f"selected raw evidence disappeared: {address.chunk_id}")
        metadata = metadata_by_id.get(address.chunk_id)
        if metadata is None or result.durable_source_id != str(metadata["source_id"]):
            raise RuntimeError(f"selected evidence changed provenance: {address.chunk_id}")
        if result.chunk.token_count != int(metadata["token_count"]):
            raise RuntimeError(f"selected evidence changed token count: {address.chunk_id}")
        hydrated.append(result)
    return hydrated, time.perf_counter_ns() - started


@dataclass(frozen=True, slots=True)
class _PreparedRawEvidence:
    result: Any
    role: str
    created_at: str
    raw_text: str
    rendered_text: str


@dataclass(frozen=True, slots=True)
class _ProviderPromptEnvelope:
    packed_count: int
    context_token_proxy: int
    prompt_token_proxy: int
    provider_messages: list[dict[str, str]]
    serialized: bytes
    pack_and_prompt_render_count_ns: int
    serialize_ns: int
    provider_ready_at_ns: int


def _prepare_raw_evidence(result: Any) -> _PreparedRawEvidence:
    turn = result.turn
    timestamp = ""
    role = ""
    if turn is not None:
        role = str(turn.role).strip().casefold()
        created_at = turn.created_at
        timestamp = (
            created_at.isoformat()
            if hasattr(created_at, "isoformat")
            else str(created_at or "")
        )
    provenance = " | ".join(value for value in (timestamp, role) if value)
    prefix = f"[{provenance}] " if provenance else ""
    raw_text = str(result.chunk.text)
    rendered_text = prefix + raw_text
    return _PreparedRawEvidence(
        result=result,
        role=role,
        created_at=timestamp,
        raw_text=raw_text,
        rendered_text=rendered_text,
    )


def _raw_evidence_row(prepared: _PreparedRawEvidence) -> dict[str, Any]:
    result = prepared.result
    return {
        "evidence_id": result.chunk.chunk_id,
        "chunk_id": result.chunk.chunk_id,
        "turn_id": result.chunk.turn_id,
        "source_id": result.durable_source_id,
        "role": prepared.role,
        "created_at": prepared.created_at,
        "route": result.route or "",
        "score": _score_float(result.score),
        "raw_text": prepared.raw_text,
        "raw_text_sha256": quote_sha256(prepared.raw_text),
        "rendered_text": prepared.rendered_text,
        "rendered_text_sha256": quote_sha256(prepared.rendered_text),
    }


def _context_token_proxy(texts: Sequence[str]) -> int:
    if not texts:
        return 0
    return count_tokens(
        "\n".join(f"[{index}] {text}" for index, text in enumerate(texts, 1))
    )


def _render_and_count_prompt(
    prompt_question: str,
    texts: Sequence[str],
) -> tuple[int, list[dict[str, str]], int]:
    context_tokens = _context_token_proxy(texts)
    messages = build_qa_prompt(prompt_question, texts)
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    return context_tokens, messages, prompt_tokens


def _fits_prompt_workspace(
    *,
    context_tokens: int,
    prompt_tokens: int,
    max_context_tokens: int,
    max_prompt_tokens: int,
) -> bool:
    return context_tokens <= max_context_tokens and (
        prompt_tokens + RESPONDER_OUTPUT_TOKEN_RESERVE <= max_prompt_tokens
    )


def _pack_provider_prompt(
    rendered_texts: Sequence[str],
    *,
    prompt_question: str,
    max_context_tokens: int,
    max_prompt_tokens: int,
) -> _ProviderPromptEnvelope:
    """Build the provider envelope once whenever the complete packet fits."""

    started = time.perf_counter_ns()
    packed_count = 0
    state: tuple[int, list[dict[str, str]], int] | None = None

    # The common all-fit case needs one complete token-count pass, rather than
    # recounting prefixes 1..k and then recounting the accepted prompt again.
    complete = _render_and_count_prompt(prompt_question, rendered_texts)
    if _fits_prompt_workspace(
        context_tokens=complete[0],
        prompt_tokens=complete[2],
        max_context_tokens=max_context_tokens,
        max_prompt_tokens=max_prompt_tokens,
    ):
        packed_count = len(rendered_texts)
        state = complete

    # Preserve ranked-prefix behavior whenever the complete packet does not fit.
    if state is None:
        for end in range(1, len(rendered_texts) + 1):
            proposal = _render_and_count_prompt(
                prompt_question,
                rendered_texts[:end],
            )
            if not _fits_prompt_workspace(
                context_tokens=proposal[0],
                prompt_tokens=proposal[2],
                max_context_tokens=max_context_tokens,
                max_prompt_tokens=max_prompt_tokens,
            ):
                break
            packed_count = end
            state = proposal
        if state is None:
            state = _render_and_count_prompt(prompt_question, ())

    pack_elapsed = time.perf_counter_ns() - started
    context_tokens, messages, prompt_tokens = state

    serialize_started = time.perf_counter_ns()
    serialized = _canonical_json_bytes({"messages": messages})
    provider_ready_at = time.perf_counter_ns()
    serialize_elapsed = provider_ready_at - serialize_started
    return _ProviderPromptEnvelope(
        packed_count=packed_count,
        context_token_proxy=context_tokens,
        prompt_token_proxy=prompt_tokens,
        provider_messages=messages,
        serialized=serialized,
        pack_and_prompt_render_count_ns=pack_elapsed,
        serialize_ns=serialize_elapsed,
        provider_ready_at_ns=provider_ready_at,
    )


def _raw_packet_semantic(
    evidence: Sequence[dict[str, Any]],
    envelope: _ProviderPromptEnvelope,
) -> dict[str, Any]:
    packed = list(evidence[: envelope.packed_count])
    dropped = [str(item["chunk_id"]) for item in evidence[envelope.packed_count :]]
    semantic = {
        "selected_evidence": list(evidence),
        "packed_evidence": packed,
        "selected_chunk_ids": [str(item["chunk_id"]) for item in evidence],
        "packed_chunk_ids": [str(item["chunk_id"]) for item in packed],
        "dropped_chunk_ids": dropped,
        "context_token_proxy": envelope.context_token_proxy,
        "prompt_token_proxy": envelope.prompt_token_proxy,
        "prompt_workspace_token_proxy": (
            envelope.prompt_token_proxy + RESPONDER_OUTPUT_TOKEN_RESERVE
        ),
        "provider_messages": envelope.provider_messages,
        "provider_payload_sha256": hashlib.sha256(envelope.serialized).hexdigest(),
        "provider_payload_utf8_bytes": len(envelope.serialized),
        "raw_evidence_only": True,
    }
    return semantic


def _pack_raw_evidence(
    evidence: Sequence[dict[str, Any]],
    *,
    prompt_question: str,
    max_context_tokens: int,
    max_prompt_tokens: int,
) -> tuple[dict[str, Any], dict[str, int], int]:
    envelope = _pack_provider_prompt(
        [str(item["rendered_text"]) for item in evidence],
        prompt_question=prompt_question,
        max_context_tokens=max_context_tokens,
        max_prompt_tokens=max_prompt_tokens,
    )
    # Audit hashes and the semantic result dictionary are deliberately built
    # after provider_ready_at_ns, so they do not inflate the online boundary.
    semantic = _raw_packet_semantic(evidence, envelope)
    timings = {
        "pack_and_prompt_render_count_ns": envelope.pack_and_prompt_render_count_ns,
        "serialize_ns": envelope.serialize_ns,
    }
    return semantic, timings, envelope.provider_ready_at_ns


def _arm_payload(
    database: Database,
    addresses: Sequence[RankedChunkAddress],
    metadata_by_id: Mapping[str, Mapping[str, Any]],
    *,
    prompt_question: str,
    max_context_tokens: int,
    max_prompt_tokens: int,
) -> tuple[dict[str, Any], dict[str, int], int]:
    hydrated, hydration_ns = _hydrate_addresses(database, addresses, metadata_by_id)
    prepare_started = time.perf_counter_ns()
    prepared = [_prepare_raw_evidence(result) for result in hydrated]
    prepare_ns = time.perf_counter_ns() - prepare_started
    envelope = _pack_provider_prompt(
        [item.rendered_text for item in prepared],
        prompt_question=prompt_question,
        max_context_tokens=max_context_tokens,
        max_prompt_tokens=max_prompt_tokens,
    )
    # The provider can be called at this point. Everything below is assay
    # bookkeeping and therefore stays outside the A3 prompt-to-bytes clock.
    audit_started = time.perf_counter_ns()
    evidence = [_raw_evidence_row(item) for item in prepared]
    payload = _raw_packet_semantic(evidence, envelope)
    audit_ns = time.perf_counter_ns() - audit_started
    timings = {
        "hydrate_raw_ns": hydration_ns,
        "prepare_provider_text_ns": prepare_ns,
        "pack_and_prompt_render_count_ns": envelope.pack_and_prompt_render_count_ns,
        "serialize_ns": envelope.serialize_ns,
        "post_boundary_audit_materialization_ns": audit_ns,
    }
    return payload, timings, envelope.provider_ready_at_ns


def _lane_audit(union: Any) -> list[dict[str, Any]]:
    rows = []
    for lane in union.lane_selections:
        rows.append(
            {
                "lane_id": lane.lane_id,
                "budget": lane.budget,
                "selected_before_dedup": [item.chunk_id for item in lane.selected_before_dedup],
                "retained_after_dedup": [item.chunk_id for item in lane.retained_after_dedup],
                "dedup_excluded_evidence_ids": list(lane.dedup_excluded_evidence_ids),
                "refilled": [item.chunk_id for item in lane.refilled],
                "refill_skipped_evidence_ids": list(lane.refill_skipped_evidence_ids),
                "selected_after_refill": [item.chunk_id for item in lane.selected_after_refill],
                "unfilled_slots": lane.unfilled_slots,
            }
        )
    return rows


def _execute_question(
    *,
    row: Mapping[str, Any],
    embedder: EmbeddingService,
    dense_index: ExactDenseAddressIndex,
    source_neighborhood_index: SourceNeighborhoodIndex,
    lexical: ResidentBM25Index,
    database: Database,
    metadata_by_id: Mapping[str, Mapping[str, Any]],
    executor: ThreadPoolExecutor,
    device: str,
    lane_budget: int,
    candidates_per_lane: int,
    max_context_tokens: int,
    max_prompt_tokens: int,
    include_diagnostic_arms: bool = True,
) -> tuple[dict[str, Any], dict[str, int]]:
    sample_started = time.perf_counter_ns()
    prompt_question = str(row["prompt_question"])
    query = str(row["retrieval_query"])

    route_started = time.perf_counter_ns()
    event_plan = plan_temporal_enumeration(query)
    evidence_window = resolve_temporal_evidence_window(
        event_plan,
        prompt_question,
    )
    route_plan_ns = time.perf_counter_ns() - route_started

    parallel_started = time.perf_counter_ns()
    lexical_future = executor.submit(
        _timed_lexical_routes,
        lexical,
        query,
        candidates_per_lane,
        event_plan,
        metadata_by_id,
        event_plan.budget + (2 * lane_budget) + SOURCE_NEIGHBOR_LANE_BUDGET,
        evidence_window,
    )
    _cuda_synchronize(device)
    encode_started = time.perf_counter_ns()
    query_embedding = embedder.embed_query(query)
    _cuda_synchronize(device)
    encode_ns = time.perf_counter_ns() - encode_started

    dense_started = time.perf_counter_ns()
    dense = dense_index.search(query_embedding, limit=candidates_per_lane)
    dense_ns = time.perf_counter_ns() - dense_started
    lexical_hits, bm25_ns, event_hits, event_ns = lexical_future.result()
    parallel_ns = time.perf_counter_ns() - parallel_started
    lexical_hits, lexical_time_excluded, lexical_time_filter_ns = (
        _filter_temporal_window(
            lexical_hits,
            evidence_window,
            metadata_by_id,
        )
    )
    dense, dense_time_excluded, dense_time_filter_ns = _filter_temporal_window(
        dense,
        evidence_window,
        metadata_by_id,
    )

    anchor_union_started = time.perf_counter_ns()
    anchor_union = post_selection_lane_union(
        (
            RankedEvidenceLane("bm25", lane_budget, lexical_hits),
            RankedEvidenceLane("exact_dense", lane_budget, dense),
            RankedEvidenceLane(
                "temporal_event",
                TEMPORAL_EVENT_LANE_BUDGET,
                event_hits,
            ),
        ),
        evidence_id=lambda item: item.chunk_id,
    )
    anchor_union_ns = time.perf_counter_ns() - anchor_union_started
    source_neighbor_hits, source_neighborhood, source_neighborhood_ns = (
        _timed_source_neighborhood(
            source_neighborhood_index,
            anchor_union.items,
        )
    )
    (
        source_neighbor_hits,
        source_neighbor_time_excluded,
        source_neighbor_time_filter_ns,
    ) = _filter_temporal_window(
        source_neighbor_hits,
        evidence_window,
        metadata_by_id,
    )

    union_started = time.perf_counter_ns()
    union = post_selection_lane_union(
        (
            RankedEvidenceLane("bm25", lane_budget, lexical_hits),
            RankedEvidenceLane("exact_dense", lane_budget, dense),
            RankedEvidenceLane(
                "source_neighborhood",
                SOURCE_NEIGHBOR_LANE_BUDGET,
                source_neighbor_hits,
            ),
            RankedEvidenceLane(
                "temporal_event",
                TEMPORAL_EVENT_LANE_BUDGET,
                event_hits,
            ),
        ),
        evidence_id=lambda item: item.chunk_id,
    )
    union_ns = time.perf_counter_ns() - union_started

    # A3 is the actual proposed online boundary, so materialize it first and
    # stop its wall clock before the isolated-lane diagnostics do extra work.
    a3, a3_timings, a3_provider_ready_at_ns = _arm_payload(
        database,
        union.items,
        metadata_by_id,
        prompt_question=prompt_question,
        max_context_tokens=max_context_tokens,
        max_prompt_tokens=max_prompt_tokens,
    )
    a3_total_ns = a3_provider_ready_at_ns - sample_started
    diagnostic_arms: dict[str, dict[str, Any]] = {}
    diagnostic_timings: dict[str, int] = {}
    if include_diagnostic_arms:
        a0, a0_timings, _a0_provider_ready_at_ns = _arm_payload(
            database,
            lexical_hits[:lane_budget],
            metadata_by_id,
            prompt_question=prompt_question,
            max_context_tokens=max_context_tokens,
            max_prompt_tokens=max_prompt_tokens,
        )
        a1, a1_timings, _a1_provider_ready_at_ns = _arm_payload(
            database,
            dense[:lane_budget],
            metadata_by_id,
            prompt_question=prompt_question,
            max_context_tokens=max_context_tokens,
            max_prompt_tokens=max_prompt_tokens,
        )
        a2_source, a2_source_timings, _a2_source_provider_ready_at_ns = (
            _arm_payload(
                database,
                source_neighbor_hits[:SOURCE_NEIGHBOR_LANE_BUDGET],
                metadata_by_id,
                prompt_question=prompt_question,
                max_context_tokens=max_context_tokens,
                max_prompt_tokens=max_prompt_tokens,
            )
        )
        a2_event, a2_event_timings, _a2_event_provider_ready_at_ns = _arm_payload(
            database,
            event_hits[: event_plan.budget],
            metadata_by_id,
            prompt_question=prompt_question,
            max_context_tokens=max_context_tokens,
            max_prompt_tokens=max_prompt_tokens,
        )
        diagnostic_arms = {
            "a0_bm25": a0,
            "a1_exact_dense": a1,
            "a2_source_neighborhood": a2_source,
            "a2_temporal_event": a2_event,
        }
        diagnostic_timings = {
            **{f"a0_{key}": value for key, value in a0_timings.items()},
            **{f"a1_{key}": value for key, value in a1_timings.items()},
            **{
                f"a2_source_{key}": value
                for key, value in a2_source_timings.items()
            },
            **{
                f"a2_event_{key}": value
                for key, value in a2_event_timings.items()
            },
        }
    # These lexical diagnostics are retained for analysis but are neither
    # consumed by retrieval nor charged to the provider-ready A3 boundary.
    features = _query_features(query)
    full_assay_ns = time.perf_counter_ns() - sample_started

    semantic = {
        "ordinal": int(row["ordinal"]),
        "question_id": str(row["question_id"]),
        "probe_sha256": str(row["probe_sha256"]),
        "retrieval_query_form": RETRIEVAL_QUERY_FORM,
        "retrieval_query_sha256": str(row["retrieval_query_sha256"]),
        "prompt_question_sha256": str(row["prompt_question_sha256"]),
        "query_features": {key: value for key, value in features.items() if key != "elapsed_ns"},
        "temporal_event_plan": {
            "active": event_plan.active,
            "reason": event_plan.reason,
            "event_verb": event_plan.event_verb,
            "search_terms": list(event_plan.search_terms),
            "budget": event_plan.budget,
            "lookback_months": event_plan.lookback_months,
        },
        "temporal_evidence_window": {
            "active": evidence_window is not None,
            "bounds": (
                None if evidence_window is None else evidence_window.model_dump()
            ),
            "applied_before_independent_lane_selection": True,
            "excluded_chunk_ids": {
                "bm25": list(lexical_time_excluded),
                "exact_dense": list(dense_time_excluded),
                "source_neighborhood": list(source_neighbor_time_excluded),
            },
        },
        "source_neighborhood_plan": {
            "active": bool(anchor_union.items),
            "budget": SOURCE_NEIGHBOR_LANE_BUDGET,
            "policy": "same-source-adjacent-turn-round-robin-v1",
            "anchor_evidence_ids": list(anchor_union.evidence_ids),
            **_source_neighborhood_audit(source_neighborhood),
        },
        "wide_frontier": {
            "candidate_limit_per_lane": candidates_per_lane,
            "temporal_event_candidate_limit": (
                event_plan.budget
                + (2 * lane_budget)
                + SOURCE_NEIGHBOR_LANE_BUDGET
            ),
            "bm25": [_candidate_row(item, metadata_by_id) for item in lexical_hits],
            "exact_dense": [_candidate_row(item, metadata_by_id) for item in dense],
            "source_neighborhood": [
                _candidate_row(item, metadata_by_id)
                for item in source_neighbor_hits
            ],
            "temporal_event": [_candidate_row(item, metadata_by_id) for item in event_hits],
        },
        "post_selection_union": {
            "evidence_ids": list(union.evidence_ids),
            "lane_audit": _lane_audit(union),
        },
        "arms": {
            **diagnostic_arms,
            "a3_protected_union": a3,
        },
    }
    timings = {
        "post_boundary_query_diagnostics_ns": int(features["elapsed_ns"]),
        "query_route_plan_ns": route_plan_ns,
        "query_encode_ns": encode_ns,
        "bm25_ns": bm25_ns,
        "temporal_event_search_ns": event_ns,
        "exact_dense_scan_ns": dense_ns,
        "parallel_retrieval_wall_ns": parallel_ns,
        "bm25_temporal_admissibility_ns": lexical_time_filter_ns,
        "exact_dense_temporal_admissibility_ns": dense_time_filter_ns,
        "source_neighbor_anchor_union_ns": anchor_union_ns,
        "source_neighborhood_ns": source_neighborhood_ns,
        "source_neighbor_temporal_admissibility_ns": (
            source_neighbor_time_filter_ns
        ),
        "lane_union_ns": union_ns,
        **{f"a3_{key}": value for key, value in a3_timings.items()},
        "a3_prompt_to_serialized_bytes_ns": a3_total_ns,
        **diagnostic_timings,
        "post_boundary_semantic_and_diagnostics_ns": full_assay_ns - a3_total_ns,
    }
    return semantic, timings


def _online_semantic_projection(semantic: Mapping[str, Any]) -> dict[str, Any]:
    """Fields produced by the actual A3 route, excluding diagnostic arms."""

    arms = semantic.get("arms")
    if not isinstance(arms, Mapping) or "a3_protected_union" not in arms:
        raise ValueError("semantic result omitted A3")
    return {
        "ordinal": semantic.get("ordinal"),
        "question_id": semantic.get("question_id"),
        "probe_sha256": semantic.get("probe_sha256"),
        "retrieval_query_form": semantic.get("retrieval_query_form"),
        "retrieval_query_sha256": semantic.get("retrieval_query_sha256"),
        "prompt_question_sha256": semantic.get("prompt_question_sha256"),
        "temporal_event_plan": semantic.get("temporal_event_plan"),
        "temporal_evidence_window": semantic.get("temporal_evidence_window"),
        "source_neighborhood_plan": semantic.get("source_neighborhood_plan"),
        "wide_frontier": semantic.get("wide_frontier"),
        "post_selection_union": semantic.get("post_selection_union"),
        "a3_protected_union": arms["a3_protected_union"],
    }


def _numeric_summary(values: Sequence[int]) -> dict[str, int | float]:
    if not values or any(isinstance(value, bool) or int(value) < 0 for value in values):
        raise ValueError("timing populations must contain non-negative integers")
    ordered = sorted(int(value) for value in values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _timing_aggregates(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot summarize an empty runtime")
    keys = tuple(samples[0]["timings_ns"])
    if any(tuple(sample["timings_ns"]) != keys for sample in samples):
        raise ValueError("runtime samples changed their timing schema")
    return {
        key: _numeric_summary([int(sample["timings_ns"][key]) for sample in samples])
        for key in keys
    }


def _validated_runtime_aggregates(
    runtime: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every latency claim from the sealed per-sample clocks."""

    controls = runtime.get("controls")
    samples = runtime.get("samples")
    selected_questions = selection.get("questions")
    if (
        not isinstance(controls, Mapping)
        or not isinstance(samples, list)
        or not isinstance(selected_questions, list)
    ):
        raise ValueError("runtime omitted controls, samples, or selection questions")
    rounds = controls.get("measured_rounds")
    if (
        isinstance(rounds, bool)
        or not isinstance(rounds, int)
        or rounds < MIN_LATENCY_REPEATS
    ):
        raise ValueError("runtime measured rounds are invalid")
    if len(samples) != rounds * ORIGINAL_QUESTIONS or controls.get(
        "measured_samples"
    ) != len(samples):
        raise ValueError("runtime sample population is incomplete")

    identity_by_question = {
        str(row["question_id"]): (
            int(row["ordinal"]),
            hashlib.sha256(_canonical_json_bytes(row)).hexdigest(),
        )
        for row in selected_questions
    }
    expected_ids = set(ORIGINAL_ORDERED_QUESTION_IDS)
    for repeat in range(rounds):
        repeat_rows = [row for row in samples if row.get("repeat") == repeat]
        if len(repeat_rows) != ORIGINAL_QUESTIONS or {
            row.get("question_id") for row in repeat_rows
        } != expected_ids:
            raise ValueError(f"runtime repeat {repeat} changed question coverage")
        for row in repeat_rows:
            question_id = str(row.get("question_id", ""))
            expected_ordinal, expected_sha = identity_by_question[question_id]
            if (
                row.get("ordinal") != expected_ordinal
                or row.get("semantic_question_sha256") != expected_sha
                or not isinstance(row.get("timings_ns"), Mapping)
            ):
                raise ValueError("runtime sample changed its semantic binding")

    aggregates = _timing_aggregates(samples)
    if _canonical_json_bytes(aggregates) != _canonical_json_bytes(
        runtime.get("aggregates_ns")
    ):
        raise ValueError("runtime aggregates do not reproduce from samples")
    per_question = {
        question_id: _numeric_summary(
            [
                int(sample["timings_ns"]["a3_prompt_to_serialized_bytes_ns"])
                for sample in samples
                if sample["question_id"] == question_id
            ]
        )
        for question_id in ORIGINAL_ORDERED_QUESTION_IDS
    }
    if _canonical_json_bytes(per_question) != _canonical_json_bytes(
        runtime.get("per_question_a3_ns")
    ):
        raise ValueError("runtime per-question latency does not reproduce")
    p95_ns = int(aggregates["a3_prompt_to_serialized_bytes_ns"]["p95"])
    if (
        runtime.get("a3_warm_p95_milliseconds") != p95_ns / 1_000_000
        or runtime.get("initial_200ms_latency_gate") is not (p95_ns <= 200_000_000)
        or runtime.get("stretch_100ms_latency_gate") is not (p95_ns <= 100_000_000)
    ):
        raise ValueError("runtime latency gates do not reproduce from samples")
    return aggregates


def _collect_gold_blind_run(
    *,
    source_selection: Path,
    output_root: Path,
    device: str,
    lane_budget: int,
    candidates_per_lane: int,
    max_context_tokens: int,
    max_prompt_tokens: int,
    warmup_rounds: int,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for label, value in (
        ("lane_budget", lane_budget),
        ("candidates_per_lane", candidates_per_lane),
        ("max_context_tokens", max_context_tokens),
        ("max_prompt_tokens", max_prompt_tokens),
        ("warmup_rounds", warmup_rounds),
        ("repeats", repeats),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if candidates_per_lane < lane_budget:
        raise ValueError("candidates_per_lane must cover each lane budget")
    if max_prompt_tokens <= RESPONDER_OUTPUT_TOKEN_RESERVE:
        raise ValueError("max_prompt_tokens cannot fit the responder reserve")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    setup_started = time.perf_counter_ns()
    source_started = time.perf_counter_ns()
    _source, source_sha, database_path, _index_path = _source_paths(source_selection)
    source_verify_ns = time.perf_counter_ns() - source_started

    compiled_started = time.perf_counter_ns()
    compiled = _load_compiled(output_root)
    compiled_load_ns = time.perf_counter_ns() - compiled_started
    probes_started = time.perf_counter_ns()
    probes, probes_sha = _load_probes(output_root)
    probes_load_ns = time.perf_counter_ns() - probes_started
    if compiled.artifact["source_binding"]["source_selection_sha256"] != source_sha:
        raise ValueError("compiled addresses belong to another source selection")
    implementation = _implementation_identity()
    if compiled.artifact.get("implementation") != implementation:
        raise ValueError("compiled addresses belong to another implementation")

    database_started = time.perf_counter_ns()
    database = Database(database_path, read_only=True)
    durable_lexical = LexicalIndex(database)
    lexical_stats = durable_lexical.stats()
    database_open_ns = time.perf_counter_ns() - database_started
    resident_lexical_started = time.perf_counter_ns()
    lexical = ResidentBM25Index(durable_lexical)
    resident_lexical_compile_ns = time.perf_counter_ns() - resident_lexical_started
    if lexical.stats() != lexical_stats:
        raise RuntimeError("resident BM25 population differs from its durable source")
    embedder = EmbeddingService(device=device)
    semantic_by_question: dict[str, dict[str, Any]] = {}
    runtime_samples: list[dict[str, Any]] = []
    warmup_started = 0
    warmup_ns = 0
    diagnostic_arm_materialization_ns = 0
    first_touch_ns = 0
    measured_started = 0
    resident_rss_bytes: int | None = None
    resident_cuda_memory: dict[str, int | None] = {
        "allocated_bytes": None,
        "reserved_bytes": None,
        "peak_allocated_bytes": None,
    }
    try:
        first_touch_started = time.perf_counter_ns()
        _cuda_synchronize(device)
        embedder.embed_query("provider-free hot memory retrieval calibration")
        _cuda_synchronize(device)
        first_touch_ns = time.perf_counter_ns() - first_touch_started
        verified_checkpoint = getattr(embedder, "_verified_checkpoint_sha256", None)
        if verified_checkpoint != BGE_M3_CHECKPOINT_SHA256:
            raise RuntimeError("live query encoder did not verify the pinned checkpoint")

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="hot-bm25") as executor:
            warmup_started = time.perf_counter_ns()
            for _round in range(warmup_rounds):
                for probe in probes["questions"]:
                    _execute_question(
                        row=probe,
                        embedder=embedder,
                        dense_index=compiled.dense_index,
                        source_neighborhood_index=compiled.source_neighborhood_index,
                        lexical=lexical,
                        database=database,
                        metadata_by_id=compiled.metadata_by_id,
                        executor=executor,
                        device=device,
                        lane_budget=lane_budget,
                        candidates_per_lane=candidates_per_lane,
                        max_context_tokens=max_context_tokens,
                        max_prompt_tokens=max_prompt_tokens,
                        include_diagnostic_arms=False,
                    )
            warmup_ns = time.perf_counter_ns() - warmup_started

            diagnostic_started = time.perf_counter_ns()
            for probe in probes["questions"]:
                semantic, _timings = _execute_question(
                    row=probe,
                    embedder=embedder,
                    dense_index=compiled.dense_index,
                    source_neighborhood_index=compiled.source_neighborhood_index,
                    lexical=lexical,
                    database=database,
                    metadata_by_id=compiled.metadata_by_id,
                    executor=executor,
                    device=device,
                    lane_budget=lane_budget,
                    candidates_per_lane=candidates_per_lane,
                    max_context_tokens=max_context_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                    include_diagnostic_arms=True,
                )
                semantic_by_question[str(probe["question_id"])] = semantic
            diagnostic_arm_materialization_ns = (
                time.perf_counter_ns() - diagnostic_started
            )
            gc.collect()
            _reset_cuda_peak(device)
            measured_started = time.perf_counter_ns()

            probes_in_order = list(probes["questions"])
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                for repeat in range(repeats):
                    offset = repeat % len(probes_in_order)
                    rotated = probes_in_order[offset:] + probes_in_order[:offset]
                    for execution_position, probe in enumerate(rotated):
                        semantic, timings = _execute_question(
                            row=probe,
                            embedder=embedder,
                            dense_index=compiled.dense_index,
                            source_neighborhood_index=compiled.source_neighborhood_index,
                            lexical=lexical,
                            database=database,
                            metadata_by_id=compiled.metadata_by_id,
                            executor=executor,
                            device=device,
                            lane_budget=lane_budget,
                            candidates_per_lane=candidates_per_lane,
                            max_context_tokens=max_context_tokens,
                            max_prompt_tokens=max_prompt_tokens,
                            include_diagnostic_arms=False,
                        )
                        question_id = str(probe["question_id"])
                        previous = semantic_by_question[question_id]
                        if _canonical_json_bytes(
                            _online_semantic_projection(previous)
                        ) != _canonical_json_bytes(_online_semantic_projection(semantic)):
                            raise RuntimeError(
                                f"hot retrieval was not repeatable for {question_id}"
                            )
                        runtime_samples.append(
                            {
                                "repeat": repeat,
                                "execution_position": execution_position,
                                "ordinal": int(probe["ordinal"]),
                                "question_id": question_id,
                                "semantic_question_sha256": hashlib.sha256(
                                    _canonical_json_bytes(previous)
                                ).hexdigest(),
                                "timings_ns": timings,
                                "rss_bytes_after": _process_rss_bytes(),
                            }
                        )
            finally:
                if gc_was_enabled:
                    gc.enable()
        resident_rss_bytes = _process_rss_bytes()
        resident_cuda_memory = _cuda_memory(device)
    finally:
        embedder.close()
        database.close()

    ordered_questions = [
        semantic_by_question[question_id] for question_id in ORIGINAL_ORDERED_QUESTION_IDS
    ]
    model_binding = {
        "model_id": embedder.model_name,
        "model_revision": embedder.model_revision,
        "checkpoint_sha256": embedder.checkpoint_sha256,
        "execution_identity": embedder.execution_identity,
        "one_query_encoder_forward_per_question": True,
    }
    controls = {
        "candidate_limit_per_lane": candidates_per_lane,
        "lane_budgets": {
            "bm25": lane_budget,
            "exact_dense": lane_budget,
            "source_neighborhood": SOURCE_NEIGHBOR_LANE_BUDGET,
            "temporal_event": TEMPORAL_EVENT_LANE_BUDGET,
        },
        "lane_order": [
            "bm25",
            "exact_dense",
            "source_neighborhood",
            "temporal_event",
        ],
        "source_neighborhood_policy": (
            "same-source-adjacent-turn-round-robin-v1"
        ),
        "source_neighborhood_anchor_policy": (
            "protected-bm25-dense-event-union"
        ),
        "temporal_event_policy": "implicit-temporal-event-set-v1",
        "temporal_evidence_window_policy": (
            "dated-query-calendar-lookback-before-lane-selection-v1"
        ),
        "temporal_event_refill_reserve": (
            (2 * lane_budget) + SOURCE_NEIGHBOR_LANE_BUDGET
        ),
        "deduplication": "exact_chunk_id_after_independent_lane_selection_with_same_lane_refill",
        "max_context_token_proxy": max_context_tokens,
        "max_prompt_workspace_token_proxy": max_prompt_tokens,
        "responder_output_token_reserve": RESPONDER_OUTPUT_TOKEN_RESERVE,
        "raw_chunk_hydration": True,
        "retrieval_query_form": RETRIEVAL_QUERY_FORM,
        "query_encoder_device": str(device).casefold(),
    }
    selection = {
        "format": SELECTION_FORMAT,
        "status": "development_candidate_not_validated",
        "bindings": {
            "source_selection_sha256": source_sha,
            "source_receipt_sha256": EXPECTED_SOURCE_RECEIPT_SHA256,
            "compiled_sha256": compiled.artifact_sha256,
            "probes_sha256": probes_sha,
            "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        },
        "corpus": {
            "transcript_tokens": ORIGINAL_TRANSCRIPT_TOKENS,
            "turn_count": ORIGINAL_TURNS,
            "chunk_count": EXPECTED_CHUNKS,
            "question_count": ORIGINAL_QUESTIONS,
        },
        "arms": list(ARM_IDS),
        "controls": controls,
        "model_binding": model_binding,
        "tokenizer_proxy_identity": tokenizer_proxy_identity(),
        "implementation": implementation,
        "questions": ordered_questions,
        "gold_fields_present": False,
        "retained_request_token_state_bytes": 0,
        "qwen_calls": 0,
        "responder_calls": 0,
        "judge_calls": 0,
        "provider_calls": 0,
    }
    aggregates = _timing_aggregates(runtime_samples)
    a3_p95 = int(aggregates["a3_prompt_to_serialized_bytes_ns"]["p95"])
    runtime = {
        "format": RUNTIME_FORMAT,
        "status": "hot_resident_rotating_dev10_measurement",
        "timing_contract": {
            "clock": "time.perf_counter_ns",
            "cuda_synchronized_around_query_encode": True,
            "cold_setup_excluded_from_query_samples": True,
            "warmup_rotates_all_questions": True,
            "measured_start_question_rotates_cyclically": True,
            "cyclic_gc_disabled_during_measured_rotations": True,
            "artifact_publication_excluded": True,
            "provider_rtt_prefill_decode_excluded": True,
            "query_diagnostics_excluded_from_a3_boundary": True,
            "post_boundary_audit_materialization_excluded": True,
            "diagnostic_arms_run_outside_measured_rotations": True,
            "a3_boundary": (
                "dated_prompt_question_to_canonical_serialized_messages_envelope"
            ),
        },
        "setup_ns": {
            "source_integrity_verification": source_verify_ns,
            "compiled_integrity_and_mmap": compiled_load_ns,
            "probe_integrity": probes_load_ns,
            "database_open_and_lexical_stats": database_open_ns,
            "resident_bm25_compile": resident_lexical_compile_ns,
            "model_load_checkpoint_verify_and_first_forward": first_touch_ns,
            "warmup": warmup_ns,
            "diagnostic_arm_materialization": diagnostic_arm_materialization_ns,
            "total_before_measured_samples": (
                measured_started - setup_started if measured_started else 0
            ),
        },
        "controls": {
            "warmup_rounds": warmup_rounds,
            "measured_rounds": repeats,
            "measured_samples": len(runtime_samples),
            **controls,
        },
        "lexical_stats": {key: _score_float(value) for key, value in lexical_stats.items()},
        "resident_bm25_numeric_bytes": lexical.resident_bytes,
        "resident_source_neighborhood_approx_bytes": (
            compiled.source_neighborhood_index.approx_resident_bytes
        ),
        "resident_source_neighborhood_counts": {
            "chunks": compiled.source_neighborhood_index.chunk_count,
            "turns": compiled.source_neighborhood_index.turn_count,
            "sources": compiled.source_neighborhood_index.source_count,
        },
        "samples": runtime_samples,
        "aggregates_ns": aggregates,
        "per_question_a3_ns": {
            question_id: _numeric_summary(
                [
                    int(sample["timings_ns"]["a3_prompt_to_serialized_bytes_ns"])
                    for sample in runtime_samples
                    if sample["question_id"] == question_id
                ]
            )
            for question_id in ORIGINAL_ORDERED_QUESTION_IDS
        },
        "a3_warm_p95_milliseconds": a3_p95 / 1_000_000,
        "initial_200ms_latency_gate": a3_p95 <= 200_000_000,
        "stretch_100ms_latency_gate": a3_p95 <= 100_000_000,
        "process_rss_bytes_resident": resident_rss_bytes,
        "cuda_memory_resident": resident_cuda_memory,
        "additional_compiled_artifact_bytes": (
            int(compiled.artifact["dense_address"]["bytes"])
            + int(compiled.artifact["chunk_manifest"]["bytes"])
        ),
        "qwen_calls": 0,
        "provider_calls": 0,
    }
    return selection, runtime


def run_assay(
    *,
    source_selection: Path,
    output_root: Path,
    device: str,
    lane_budget: int,
    candidates_per_lane: int,
    max_context_tokens: int,
    max_prompt_tokens: int,
    warmup_rounds: int,
    repeats: int,
) -> str:
    selection, runtime = _collect_gold_blind_run(
        source_selection=source_selection,
        output_root=output_root,
        device=device,
        lane_budget=lane_budget,
        candidates_per_lane=candidates_per_lane,
        max_context_tokens=max_context_tokens,
        max_prompt_tokens=max_prompt_tokens,
        warmup_rounds=warmup_rounds,
        repeats=repeats,
    )
    selection_path = output_root / DEFAULT_SELECTION_NAME
    selection_sha = _atomic_write_json(selection_path, selection)
    runtime["selection_sha256"] = selection_sha
    runtime_path = output_root / DEFAULT_RUNTIME_NAME
    runtime_sha = _atomic_write_json(runtime_path, runtime)
    print(
        "Hot retrieval published: "
        f"selection={selection_sha}; runtime={runtime_sha}; "
        f"a3_p95_ms={runtime['a3_warm_p95_milliseconds']:.3f}",
        flush=True,
    )
    return selection_sha


def _load_selection(output_root: Path) -> tuple[dict[str, Any], str]:
    selection, digest = _read_json_artifact(output_root / DEFAULT_SELECTION_NAME)
    bindings = selection.get("bindings")
    controls = selection.get("controls")
    questions = selection.get("questions")
    if (
        selection.get("format") != SELECTION_FORMAT
        or selection.get("status") != "development_candidate_not_validated"
        or selection.get("arms") != list(ARM_IDS)
        or selection.get("gold_fields_present") is not False
        or selection.get("retained_request_token_state_bytes") != 0
        or any(selection.get(key) != 0 for key in ("qwen_calls", "responder_calls", "judge_calls", "provider_calls"))
        or not isinstance(bindings, Mapping)
        or bindings.get("population_identity_sha256") != EXPECTED_POPULATION_SHA256
        or bindings.get("source_receipt_sha256") != EXPECTED_SOURCE_RECEIPT_SHA256
        or not isinstance(controls, Mapping)
        or not isinstance(questions, list)
        or len(questions) != ORIGINAL_QUESTIONS
        or [row.get("question_id") for row in questions]
        != list(ORIGINAL_ORDERED_QUESTION_IDS)
    ):
        raise ValueError("selection artifact changed its dev1M contract")
    return selection, digest


def _validate_arm_payload(
    arm: Mapping[str, Any],
    *,
    prompt_question: str,
    max_context_tokens: int,
    max_prompt_tokens: int,
) -> None:
    selected = arm.get("selected_evidence")
    packed = arm.get("packed_evidence")
    if not isinstance(selected, list) or not isinstance(packed, list):
        raise ValueError("arm omitted raw evidence")
    for evidence in selected:
        if not isinstance(evidence, Mapping):
            raise ValueError("arm evidence must be an object")
        raw_text = str(evidence.get("raw_text", ""))
        rendered_text = str(evidence.get("rendered_text", ""))
        if (
            not raw_text
            or evidence.get("raw_text_sha256") != quote_sha256(raw_text)
            or evidence.get("rendered_text_sha256") != quote_sha256(rendered_text)
            or not rendered_text.endswith(raw_text)
            or evidence.get("evidence_id") != evidence.get("chunk_id")
        ):
            raise ValueError("arm raw evidence changed")
    selected_ids = [str(row["chunk_id"]) for row in selected]
    packed_ids = [str(row["chunk_id"]) for row in packed]
    if arm.get("selected_chunk_ids") != selected_ids or arm.get("packed_chunk_ids") != packed_ids:
        raise ValueError("arm evidence ID binding changed")
    if packed != selected[: len(packed)] or arm.get("dropped_chunk_ids") != selected_ids[len(packed) :]:
        raise ValueError("arm packing is not a ranked prefix")
    texts = [str(row["rendered_text"]) for row in packed]
    messages = build_qa_prompt(prompt_question, texts)
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    context_tokens = _context_token_proxy(texts)
    serialized = _canonical_json_bytes({"messages": messages})
    if (
        arm.get("provider_messages") != messages
        or arm.get("context_token_proxy") != context_tokens
        or arm.get("prompt_token_proxy") != prompt_tokens
        or arm.get("prompt_workspace_token_proxy")
        != prompt_tokens + RESPONDER_OUTPUT_TOKEN_RESERVE
        or arm.get("provider_payload_sha256") != hashlib.sha256(serialized).hexdigest()
        or arm.get("provider_payload_utf8_bytes") != len(serialized)
        or context_tokens > max_context_tokens
        or prompt_tokens + RESPONDER_OUTPUT_TOKEN_RESERVE > max_prompt_tokens
        or arm.get("raw_evidence_only") is not True
    ):
        raise ValueError("arm provider boundary changed")


def _evidence_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    question: Any,
) -> dict[str, Any]:
    # Gold metrics use only memory content.  Provider-only timestamp/role
    # headers must not manufacture an answer date or numeric component.
    texts = [str(row["raw_text"]) for row in rows]
    retrieved_sources = tuple(
        dict.fromkeys(str(row["source_id"]) for row in rows if row.get("source_id"))
    )
    expected_sources = tuple(dict.fromkeys(str(value) for value in question.evidence_sources))
    expected = set(expected_sources)
    retrieved = set(retrieved_sources)
    recall = None if not expected else len(expected & retrieved) / len(expected)
    components = answer_value_component_coverage(question.answer, len(expected_sources), texts)
    return {
        "answer_present": contains_answer(texts, question.answer),
        "best_evidence_f1": best_f1(texts, question.answer),
        "expected_source_ids": list(expected_sources),
        "retrieved_source_ids": list(retrieved_sources),
        "evidence_source_recall": recall,
        "any_evidence_source": None if recall is None else bool(expected & retrieved),
        "all_evidence_sources": None if recall is None else recall == 1.0,
        "answer_value_components_expected": None if components is None else components.expected,
        "answer_value_components_found": None if components is None else components.found,
        "answer_value_component_recall": None if components is None else components.recall,
        "all_answer_value_components": None if components is None else components.all_components,
        "answer_value_component_hit_mask": [] if components is None else list(components.hit_mask),
        "answer_value_metric_kind": "" if components is None else components.metric_kind,
    }


def _wide_source_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_sources: Sequence[str],
) -> dict[str, Any]:
    retrieved = tuple(dict.fromkeys(str(row["source_id"]) for row in rows))
    expected = set(str(value) for value in expected_sources)
    recall = None if not expected else len(expected & set(retrieved)) / len(expected)
    return {
        "retrieved_source_ids": list(retrieved),
        "evidence_source_recall": recall,
        "all_evidence_sources": None if recall is None else recall == 1.0,
    }


def _aggregate_arm(rows: Sequence[Mapping[str, Any]], arm_id: str) -> dict[str, Any]:
    selected = [row["arms"][arm_id]["selected"] for row in rows]
    packed = [row["arms"][arm_id]["packed"] for row in rows]
    recalls = [float(row["evidence_source_recall"]) for row in packed if row["evidence_source_recall"] is not None]
    component_recalls = [
        float(row["answer_value_component_recall"])
        for row in packed
        if row["answer_value_component_recall"] is not None
    ]
    return {
        "arm_id": arm_id,
        "questions": len(rows),
        "selected_all_evidence_source_hits": sum(row["all_evidence_sources"] is True for row in selected),
        "packed_all_evidence_source_hits": sum(row["all_evidence_sources"] is True for row in packed),
        "packed_literal_answer_hits": sum(bool(row["answer_present"]) for row in packed),
        "packed_mean_best_evidence_f1": statistics.fmean(float(row["best_evidence_f1"]) for row in packed),
        "packed_mean_evidence_source_recall": None if not recalls else statistics.fmean(recalls),
        "packed_mean_answer_value_component_recall": (
            None if not component_recalls else statistics.fmean(component_recalls)
        ),
        "mean_context_token_proxy": statistics.fmean(
            int(row["arms"][arm_id]["context_token_proxy"]) for row in rows
        ),
        "max_context_token_proxy": max(
            int(row["arms"][arm_id]["context_token_proxy"]) for row in rows
        ),
        "max_prompt_workspace_token_proxy": max(
            int(row["arms"][arm_id]["prompt_workspace_token_proxy"]) for row in rows
        ),
    }


def score_assay(*, dataset: Path, split_manifest: Path, output_root: Path) -> str:
    selection, selection_sha = _load_selection(output_root)
    probes, probes_sha = _load_probes(output_root)
    compiled = _load_compiled(output_root)
    runtime, runtime_sha = _read_json_artifact(output_root / DEFAULT_RUNTIME_NAME)
    if (
        selection["bindings"]["probes_sha256"] != probes_sha
        or selection["bindings"]["compiled_sha256"] != compiled.artifact_sha256
        or runtime.get("format") != RUNTIME_FORMAT
        or runtime.get("selection_sha256") != selection_sha
        or runtime.get("qwen_calls") != 0
        or runtime.get("provider_calls") != 0
    ):
        raise ValueError("selection/runtime inputs are not cross-bound")
    runtime_aggregates = _validated_runtime_aggregates(runtime, selection)
    replay, replay_sha = _read_json_artifact(output_root / DEFAULT_REPLAY_NAME)
    if (
        replay.get("format") != REPLAY_FORMAT
        or replay.get("status") != "byte_identical_gold_blind_replay"
        or replay.get("selection_sha256") != selection_sha
        or replay.get("replayed_selection_sha256") != selection_sha
        or replay.get("byte_identical") is not True
        or replay.get("gold_fields_present") is not False
        or replay.get("qwen_calls") != 0
        or replay.get("provider_calls") != 0
    ):
        raise ValueError("score requires a byte-identical gold-blind replay")

    sample = load_original_population(dataset, split_manifest)
    if population_identity_sha256(sample) != EXPECTED_POPULATION_SHA256:
        raise RuntimeError("score population changed")
    question_rows: list[dict[str, Any]] = []
    protected_bm25_retained = True
    max_context = int(selection["controls"]["max_context_token_proxy"])
    max_prompt = int(selection["controls"]["max_prompt_workspace_token_proxy"])
    for probe, selected_row, question in zip(
        probes["questions"], selection["questions"], sample.questions, strict=True
    ):
        if (
            selected_row.get("question_id") != question.question_id
            or selected_row.get("probe_sha256") != probe["probe_sha256"]
            or selected_row.get("retrieval_query_form") != RETRIEVAL_QUERY_FORM
            or selected_row.get("retrieval_query_sha256")
            != quote_sha256(question.question)
            or selected_row.get("prompt_question_sha256") != quote_sha256(question.dated_question)
        ):
            raise ValueError("selection changed ordered question membership")
        scored_arms: dict[str, Any] = {}
        for arm_id in ARM_IDS:
            arm = selected_row["arms"][arm_id]
            _validate_arm_payload(
                arm,
                prompt_question=question.dated_question,
                max_context_tokens=max_context,
                max_prompt_tokens=max_prompt,
            )
            scored_arms[arm_id] = {
                "selected": _evidence_metric(arm["selected_evidence"], question=question),
                "packed": _evidence_metric(arm["packed_evidence"], question=question),
                "selected_count": len(arm["selected_evidence"]),
                "packed_count": len(arm["packed_evidence"]),
                "context_token_proxy": arm["context_token_proxy"],
                "prompt_workspace_token_proxy": arm["prompt_workspace_token_proxy"],
            }
        bm25_ids = set(selected_row["arms"]["a0_bm25"]["selected_chunk_ids"])
        union_packed = set(selected_row["arms"]["a3_protected_union"]["packed_chunk_ids"])
        protected_bm25_retained = protected_bm25_retained and bm25_ids <= union_packed

        wide_bm25 = selected_row["wide_frontier"]["bm25"]
        wide_dense = selected_row["wide_frontier"]["exact_dense"]
        wide_source_neighborhood = selected_row["wide_frontier"][
            "source_neighborhood"
        ]
        wide_event = selected_row["wide_frontier"]["temporal_event"]
        wide_union = list(wide_bm25)
        seen_wide = {row["chunk_id"] for row in wide_union}
        wide_union.extend(row for row in wide_dense if row["chunk_id"] not in seen_wide)
        seen_wide.update(row["chunk_id"] for row in wide_dense)
        wide_union.extend(
            row
            for row in wide_source_neighborhood
            if row["chunk_id"] not in seen_wide
        )
        seen_wide.update(row["chunk_id"] for row in wide_source_neighborhood)
        wide_union.extend(row for row in wide_event if row["chunk_id"] not in seen_wide)
        wide = {
            "bm25": _wide_source_metric(wide_bm25, expected_sources=question.evidence_sources),
            "exact_dense": _wide_source_metric(wide_dense, expected_sources=question.evidence_sources),
            "source_neighborhood": _wide_source_metric(
                wide_source_neighborhood,
                expected_sources=question.evidence_sources,
            ),
            "temporal_event": _wide_source_metric(
                wide_event,
                expected_sources=question.evidence_sources,
            ),
            "union": _wide_source_metric(wide_union, expected_sources=question.evidence_sources),
        }
        question_rows.append(
            {
                "ordinal": selected_row["ordinal"],
                "question_id": question.question_id,
                "category": question.category,
                "arms": scored_arms,
                "wide_frontier": wide,
            }
        )

    aggregates = [_aggregate_arm(question_rows, arm_id) for arm_id in ARM_IDS]
    aggregate_by_arm = {row["arm_id"]: row for row in aggregates}
    a3 = aggregate_by_arm["a3_protected_union"]
    component_gate = a3["packed_mean_answer_value_component_recall"]
    evidence_gates = {
        "all_source_reach_10_of_10": a3["packed_all_evidence_source_hits"] == ORIGINAL_QUESTIONS,
        "literal_answer_reach_at_least_5_of_10": a3["packed_literal_answer_hits"] >= 5,
        "mean_answer_value_component_recall_1": component_gate == 1.0,
        "protected_bm25_survives_union_and_packing": protected_bm25_retained,
        "context_cap": a3["max_context_token_proxy"] <= max_context,
        "prompt_workspace_cap": a3["max_prompt_workspace_token_proxy"] <= max_prompt,
        "zero_qwen_and_provider_calls": all(selection[key] == 0 for key in ("qwen_calls", "provider_calls")),
        "resident_and_compiled_addresses_at_most_512_mib": (
            runtime["additional_compiled_artifact_bytes"]
            + runtime["resident_bm25_numeric_bytes"]
            + runtime["resident_source_neighborhood_approx_bytes"]
            <= 512 * 1024 * 1024
        ),
        "byte_identical_gold_blind_replay": True,
    }
    latency_gates = {
        "warm_a3_p95_at_most_200ms": bool(runtime["initial_200ms_latency_gate"]),
        "warm_a3_p95_at_most_100ms": bool(runtime["stretch_100ms_latency_gate"]),
        "warm_a3_p95_milliseconds": runtime["a3_warm_p95_milliseconds"],
    }
    score = {
        "format": SCORE_FORMAT,
        "status": "development_retrieval_metrics_not_answer_accuracy",
        "selection_sha256": selection_sha,
        "runtime_sha256": runtime_sha,
        "replay_sha256": replay_sha,
        "compiled_sha256": compiled.artifact_sha256,
        "probes_sha256": probes_sha,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "corpus": selection["corpus"],
        "historical_s0_control": {
            "all_source_hits": 10,
            "literal_answer_hits": 5,
            "mean_answer_value_component_recall": 1.0,
            "mean_context_token_proxy": 2127.4,
            "max_context_token_proxy": 2332,
        },
        "aggregates": aggregates,
        "questions": question_rows,
        "promotion_gates": {
            "evidence": evidence_gates,
            "latency": latency_gates,
            "all_available_development_gates_pass": all(evidence_gates.values())
            and latency_gates["warm_a3_p95_at_most_200ms"],
            "formal_promotion_eligible": False,
            "formal_promotion_blocker": (
                "exact annotated-turn non-regression is unavailable on this pinned dev1M projection"
            ),
        },
        "validated_runtime_aggregates_ns": runtime_aggregates,
        "exact_annotated_turn_metric_available": False,
        "exact_annotated_turn_metric_unavailable_reason": (
            "the pinned cleaned LongMemEval rows expose labeled source sessions, not exact turn IDs"
        ),
        "gold_fields_present": True,
        "responder_calls": 0,
        "judge_calls": 0,
        "provider_calls": 0,
    }
    path = output_root / DEFAULT_SCORE_NAME
    digest = _atomic_write_json(path, score)
    print(
        f"Hot retrieval scored: {path} ({digest}); "
        f"a3_sources={a3['packed_all_evidence_source_hits']}/10; "
        f"literal={a3['packed_literal_answer_hits']}/10; "
        "available_gates="
        f"{score['promotion_gates']['all_available_development_gates_pass']}",
        flush=True,
    )
    return digest


def replay_assay(*, source_selection: Path, output_root: Path, device: str) -> str:
    expected, expected_sha = _load_selection(output_root)
    controls = expected["controls"]
    observed, runtime = _collect_gold_blind_run(
        source_selection=source_selection,
        output_root=output_root,
        device=device,
        lane_budget=int(controls["lane_budgets"]["bm25"]),
        candidates_per_lane=int(controls["candidate_limit_per_lane"]),
        max_context_tokens=int(controls["max_context_token_proxy"]),
        max_prompt_tokens=int(controls["max_prompt_workspace_token_proxy"]),
        warmup_rounds=1,
        repeats=1,
    )
    observed_bytes = _canonical_json_bytes(observed)
    observed_sha = hashlib.sha256(observed_bytes).hexdigest()
    if observed_bytes != _canonical_json_bytes(expected) or observed_sha != expected_sha:
        raise RuntimeError(
            f"gold-blind replay changed selection bytes ({observed_sha} != {expected_sha})"
        )
    replay = {
        "format": REPLAY_FORMAT,
        "status": "byte_identical_gold_blind_replay",
        "selection_sha256": expected_sha,
        "replayed_selection_sha256": observed_sha,
        "byte_identical": True,
        "runtime_aggregates_ns": runtime["aggregates_ns"],
        "a3_warm_p95_milliseconds": runtime["a3_warm_p95_milliseconds"],
        "gold_fields_present": False,
        "qwen_calls": 0,
        "provider_calls": 0,
    }
    path = output_root / DEFAULT_REPLAY_NAME
    digest = _atomic_write_json(path, replay)
    print(f"Byte-identical replay published: {path} ({digest})", flush=True)
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export-probes", help="seal gold-free dev1M questions")
    export.add_argument("--dataset", type=Path, required=True)
    export.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)

    compile_parser = commands.add_parser("compile-base", help="compile query-independent addresses")
    compile_parser.add_argument("--source-selection", type=Path, required=True)

    run_parser = commands.add_parser(
        "run",
        help="run A0/A1/A2/A3 with no gold or provider",
    )
    run_parser.add_argument("--source-selection", type=Path, required=True)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--lane-budget", type=int, default=8)
    run_parser.add_argument("--candidates-per-lane", type=int, default=96)
    run_parser.add_argument("--max-context-tokens", type=int, default=7000)
    run_parser.add_argument("--max-prompt-tokens", type=int, default=8000)
    run_parser.add_argument("--warmup-rounds", type=int, default=1)
    run_parser.add_argument("--repeats", type=int, default=MIN_LATENCY_REPEATS)

    score_parser = commands.add_parser("score", help="join gold after selection is sealed")
    score_parser.add_argument("--dataset", type=Path, required=True)
    score_parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)

    replay_parser = commands.add_parser("replay", help="require byte-identical gold-blind selection")
    replay_parser.add_argument("--source-selection", type=Path, required=True)
    replay_parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    if args.command == "export-probes":
        export_probes(
            dataset=args.dataset.resolve(),
            split_manifest=args.split_manifest.resolve(),
            output_root=output_root,
        )
    elif args.command == "compile-base":
        compile_base(
            source_selection=args.source_selection.resolve(),
            output_root=output_root,
        )
    elif args.command == "run":
        run_assay(
            source_selection=args.source_selection.resolve(),
            output_root=output_root,
            device=args.device,
            lane_budget=args.lane_budget,
            candidates_per_lane=args.candidates_per_lane,
            max_context_tokens=args.max_context_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            warmup_rounds=args.warmup_rounds,
            repeats=args.repeats,
        )
    elif args.command == "score":
        score_assay(
            dataset=args.dataset.resolve(),
            split_manifest=args.split_manifest.resolve(),
            output_root=output_root,
        )
    elif args.command == "replay":
        replay_assay(
            source_selection=args.source_selection.resolve(),
            output_root=output_root,
            device=args.device,
        )
    else:  # pragma: no cover - argparse owns this invariant
        raise AssertionError(f"unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
