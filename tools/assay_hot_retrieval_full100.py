"""Scale the frozen hot raw-chunk policy to locked LongMemEval full100.

The lifecycle deliberately keeps gold outside retrieval:

``prepare``
    Reconstructs the authenticated ten-shard population, verifies the ten
    exact-span source stores, and publishes only question probes.
``compile``
    Converts each immutable source store into a normalized dense matrix plus
    text-free chunk coordinates.  It makes no query or provider calls.
``run``
    Keeps one BGE-M3 query encoder resident, streams the ten independent
    approximately-1M-token namespaces, and applies the unchanged v6 lane
    policy to all 100 questions.
``replay``
    Repeats the gold-blind retrieval and requires byte-identical question
    payloads.
``score``
    Opens references only after selection and replay are sealed, producing
    retrieval diagnostics rather than answer-accuracy claims.

Terra answer generation and Sol judging live in the separate
``evaluate_hot_retrieval_full100.py`` process.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval.answer_value_coverage import (
    answer_value_component_coverage,
    best_f1,
    contains_answer,
)
from memory_condense.eval.recall_guarded_cumulative_1m_source import (
    validate_current_source_receipt,
)
from memory_condense.eval.recall_guarded_cumulative_population import (
    LOCKED_100Q_OFFSETS,
    LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
    QUESTION_PROBE_FORMAT,
    build_locked_cumulative_population_identity,
    validate_locked_cumulative_population_identity,
    validate_locked_cumulative_shard_identity,
)
from memory_condense.modeling.embedding import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    EmbeddingService,
)
from memory_condense.persistence.db import INDEXED_CHUNK_SQL, TURN_SOURCE_ID_SQL, Database
from memory_condense.search.hot_lexical import ResidentBM25Index
from memory_condense.search.hot_retrieval import ExactDenseAddressIndex
from memory_condense.search.indexes.lexical import LexicalIndex
from memory_condense.search.source_neighborhood import (
    SourceChunkMetadata,
    SourceNeighborhoodIndex,
)
from memory_condense.search.temporal_enumeration import compile_event_chunk_features

try:
    from tools import assay_hot_retrieval_1m as hot
except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
    import assay_hot_retrieval_1m as hot


PROBE_FORMAT = "memory-condense-hot-retrieval-full100-probes-v1"
COMPILED_CHUNKS_FORMAT = "memory-condense-hot-retrieval-full100-chunks-v1"
COMPILED_SHARD_FORMAT = "memory-condense-hot-retrieval-full100-compiled-shard-v1"
COMPILED_CATALOG_FORMAT = "memory-condense-hot-retrieval-full100-compiled-catalog-v1"
COMPILE_RUNTIME_FORMAT = "memory-condense-hot-retrieval-full100-compile-runtime-v1"
SELECTION_FORMAT = "memory-condense-hot-retrieval-full100-selection-v1"
RUNTIME_FORMAT = "memory-condense-hot-retrieval-full100-runtime-v1"
REPLAY_FORMAT = "memory-condense-hot-retrieval-full100-replay-v1"
SCORE_FORMAT = "memory-condense-hot-retrieval-full100-score-v1"

EXPECTED_POPULATION_SHA256 = (
    "9b8ad9337cfece1306358d0e03682a977f1b289a14b6ff7bfe40c90e6e2cb246"
)
EXPECTED_QUESTION_COUNT = 100
DEFAULT_SPLIT = Path("docs/10 - Research Log/data/longmemeval-95-target-split-v2.json")
DEFAULT_SOURCE_ROOT = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"
)
DEFAULT_OUTPUT_ROOT = Path(
    "eval_results/longmemeval-1m-hot-retrieval-full100-validation-20260905"
)

PROBES_NAME = "probes.json"
CATALOG_NAME = "compiled-catalog.json"
COMPILE_RUNTIME_NAME = "compile-runtime.json"
SELECTION_NAME = "selection.json"
RUNTIME_NAME = "runtime.json"
REPLAY_NAME = "replay.json"
SCORE_NAME = "scores.json"

LANE_BUDGET = 8
CANDIDATES_PER_LANE = 96
MAX_CONTEXT_TOKENS = 7_000
MAX_PROMPT_TOKENS = 8_000
POLICY_ID = "hot-raw-chunk-v6-frozen-dev10"
_FORBIDDEN_PROBE_KEYS = frozenset(
    {
        "answer",
        "answers",
        "answer_session_ids",
        "category",
        "evidence",
        "evidence_sources",
        "gold",
        "gold_answer",
        "question_type",
        "reference",
    }
)
_PROBE_ROW_FIELDS = frozenset(
    {
        "ordinal",
        "shard_offset",
        "local_ordinal",
        "question_id",
        "retrieval_query",
        "prompt_question",
        "retrieval_query_sha256",
        "prompt_question_sha256",
        "locked_probe_identity_sha256",
        "probe_sha256",
    }
)


def _implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "tools/assay_hot_retrieval_full100.py",
        "tools/assay_hot_retrieval_1m.py",
        "src/memory_condense/search/hot_retrieval.py",
        "src/memory_condense/search/hot_lexical.py",
        "src/memory_condense/search/post_selection_lane_union.py",
        "src/memory_condense/search/source_neighborhood.py",
        "src/memory_condense/search/temporal_enumeration.py",
        "src/memory_condense/search/indexes/lexical.py",
        "src/memory_condense/search/indexes/retrieval_models.py",
        "src/memory_condense/modeling/embedding.py",
        "src/memory_condense/eval/_retrieval_qa_prompt.py",
        "src/memory_condense/eval/answer_value_coverage.py",
        "src/memory_condense/eval/recall_guarded_cumulative_1m_source.py",
        "src/memory_condense/eval/recall_guarded_cumulative_population.py",
        "src/memory_condense/domain/_tokenizer.py",
        "src/memory_condense/domain/discourse.py",
        "src/memory_condense/domain/integrity.py",
        "src/memory_condense/persistence/db.py",
    )
    files = {path: file_sha256(root / path) for path in paths}
    return {
        "format": "memory-condense-hot-retrieval-full100-implementation-v1",
        "files": files,
        "sha256": identity_sha256(
            [{"path": path, "sha256": digest} for path, digest in files.items()]
        ),
    }


def _assert_gold_free_rows(rows: object) -> None:
    if not isinstance(rows, list):
        raise ValueError("probe population must be a list")
    def forbidden_keys(value: object) -> set[str]:
        if isinstance(value, Mapping):
            keys = {str(key).casefold() for key in value}
            nested = set().union(*(forbidden_keys(child) for child in value.values()))
            return (_FORBIDDEN_PROBE_KEYS & keys) | nested
        if isinstance(value, (list, tuple)):
            return set().union(*(forbidden_keys(child) for child in value))
        return set()

    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("probe row must be an object")
        forbidden = forbidden_keys(row)
        if forbidden:
            raise ValueError(f"probe row contains gold fields: {sorted(forbidden)}")


def _shard_root(source_root: Path, offset: int) -> Path:
    return source_root / "shards" / f"offset-{offset:03d}"


def _require_embedding_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("source receipt omitted embedding identity")
    identity = dict(value)
    required = {
        "model_id": DEFAULT_MODEL_NAME,
        "model_revision": DEFAULT_MODEL_REVISION,
        "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
        "dimension": DEFAULT_MODEL_DIM,
    }
    if any(identity.get(key) != expected for key, expected in required.items()):
        raise ValueError("source store does not use the pinned BGE-M3 address space")
    return identity


@dataclass(frozen=True, slots=True)
class SourceBinding:
    offset: int
    selection_sha256: str
    receipt_sha256: str
    database_sha256: str
    index_sha256: str
    chunk_count: int
    turn_count: int
    selection_relative_path: str
    database_path: Path
    index_path: Path
    embedding_identity: dict[str, Any]

    def artifact_row(self) -> dict[str, Any]:
        return {
            "shard_offset": self.offset,
            "source_selection_sha256": self.selection_sha256,
            "source_receipt_sha256": self.receipt_sha256,
            "database_sha256": self.database_sha256,
            "index_sha256": self.index_sha256,
            "chunk_count": self.chunk_count,
            "turn_count": self.turn_count,
            "selection_relative_path": self.selection_relative_path,
            "embedding_identity": self.embedding_identity,
        }


def _load_source_binding(
    source_root: Path,
    offset: int,
    *,
    expected_shard_identity: Mapping[str, Any] | None,
    expected_shard_identity_sha256: str | None = None,
    verify_large_files: bool,
    sample: Any | None = None,
) -> SourceBinding:
    shard = _shard_root(source_root, offset)
    selection_path = shard / "source-current-selection.json"
    selection, selection_sha = hot._read_json_artifact(selection_path)  # noqa: SLF001
    if sample is not None:
        validated = validate_current_source_receipt(
            selection,
            sample=sample,
            expected_device="cuda",
        )
        if validated != selection:
            raise ValueError(f"offset {offset:03d} source receipt normalization changed")
    embedding = _require_embedding_identity(selection.get("embedding_identity"))
    for field in ("chunk_count", "turn_count"):
        value = selection.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"source receipt has invalid {field}")
    for field in ("receipt_sha256", "database_sha256", "index_sha256"):
        value = selection.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"source receipt has invalid {field}")

    retrieval, _retrieval_sha = hot._read_json_artifact(  # noqa: SLF001
        shard / "retrieval.json"
    )
    retrieval_shard_identity = validate_locked_cumulative_shard_identity(
        retrieval.get("shard_identity", {})
    )
    if (
        retrieval.get("gold_fields_present") is not False
        or retrieval.get("provider_calls") != 0
        or retrieval.get("shard_offset") != offset
        or retrieval.get("source_store_receipt") != selection
        or retrieval.get("source_store_receipt_sha256")
        != selection.get("receipt_sha256")
        or retrieval.get("shard_identity_sha256")
        != retrieval_shard_identity["shard_identity_sha256"]
    ):
        raise ValueError(f"offset {offset:03d} source is not its sealed retrieval store")
    if expected_shard_identity is not None and (
        retrieval_shard_identity != dict(expected_shard_identity)
        or retrieval.get("shard_identity_sha256")
        != expected_shard_identity.get("shard_identity_sha256")
    ):
        raise ValueError(f"offset {offset:03d} store belongs to another population")
    if (
        expected_shard_identity_sha256 is not None
        and retrieval_shard_identity["shard_identity_sha256"]
        != expected_shard_identity_sha256
    ):
        raise ValueError(f"offset {offset:03d} store belongs to another population")

    store_entry = selection.get("selected_store_entry")
    if not isinstance(store_entry, str) or not store_entry:
        raise ValueError("source receipt omitted selected_store_entry")
    store_root = hot._safe_relative(  # noqa: SLF001
        shard / "source-current",
        store_entry,
        label=f"offset {offset:03d} selected store",
    )
    database_path = store_root / "store" / "memory.db"
    index_path = store_root / "store" / "hnsw_index.bin"
    if not database_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(f"offset {offset:03d} source store is incomplete")
    if verify_large_files and (
        file_sha256(database_path) != selection["database_sha256"]
        or file_sha256(index_path) != selection["index_sha256"]
    ):
        raise RuntimeError(f"offset {offset:03d} source store changed")
    return SourceBinding(
        offset=offset,
        selection_sha256=selection_sha,
        receipt_sha256=str(selection["receipt_sha256"]),
        database_sha256=str(selection["database_sha256"]),
        index_sha256=str(selection["index_sha256"]),
        chunk_count=int(selection["chunk_count"]),
        turn_count=int(selection["turn_count"]),
        selection_relative_path=str(selection_path.relative_to(source_root)).replace("\\", "/"),
        database_path=database_path,
        index_path=index_path,
        embedding_identity=embedding,
    )


def _load_population(dataset: Path, split_manifest: Path) -> tuple[Any, Any, Any]:
    samples, identities, population = build_locked_cumulative_population_identity(
        dataset,
        split_manifest,
        plan=LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
    )
    if population.get("population_identity_sha256") != EXPECTED_POPULATION_SHA256:
        raise RuntimeError("locked full100 population identity changed")
    if len(samples) != 10 or len(identities) != 10:
        raise RuntimeError("locked full100 must contain ten namespaces")
    return samples, identities, population


def prepare(
    *,
    dataset: Path,
    split_manifest: Path,
    source_root: Path,
    output_root: Path,
) -> str:
    path = output_root / PROBES_NAME
    if path.exists():
        artifact, digest = _load_probes(output_root)
        print(f"Full100 probes verified: {path} ({digest})", flush=True)
        return digest
    samples, identities, population = _load_population(dataset, split_manifest)
    sources: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    for sample, shard_identity, offset in zip(
        samples, identities, LOCKED_100Q_OFFSETS, strict=True
    ):
        binding = _load_source_binding(
            source_root,
            offset,
            expected_shard_identity=shard_identity,
            verify_large_files=True,
            sample=sample,
        )
        sources.append(binding.artifact_row())
        probes = shard_identity["ordered_question_probes"]
        for local_ordinal, (question, probe) in enumerate(
            zip(sample.questions, probes, strict=True)
        ):
            ordinal = offset + local_ordinal
            if (
                probe.get("ordinal") != local_ordinal
                or probe.get("retrieval_query_sha256")
                != quote_sha256(question.question)
                or probe.get("prompt_question_sha256")
                != quote_sha256(question.dated_question)
            ):
                raise ValueError(f"question probe changed at ordinal {ordinal}")
            row = {
                "ordinal": ordinal,
                "shard_offset": offset,
                "local_ordinal": local_ordinal,
                "question_id": question.question_id,
                "retrieval_query": question.question,
                "prompt_question": question.dated_question,
                "retrieval_query_sha256": quote_sha256(question.question),
                "prompt_question_sha256": quote_sha256(question.dated_question),
                "locked_probe_identity_sha256": probe["probe_identity_sha256"],
            }
            row["probe_sha256"] = identity_sha256(row)
            questions.append(row)
    _assert_gold_free_rows(questions)
    if len(questions) != EXPECTED_QUESTION_COUNT or [
        row["ordinal"] for row in questions
    ] != list(range(EXPECTED_QUESTION_COUNT)):
        raise RuntimeError("full100 question ordering changed")
    body = {
        "format": PROBE_FORMAT,
        "status": "sealed_gold_free_locked_full100_probes",
        "population_identity": population,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "question_count": len(questions),
        "shard_count": len(sources),
        "source_bindings": sources,
        "questions": questions,
        "retrieval_query_form": hot.RETRIEVAL_QUERY_FORM,
        "gold_fields_present": False,
        "provider_calls": 0,
    }
    digest = hot._atomic_write_json(path, body)  # noqa: SLF001
    print(f"Full100 probes published: {path} ({digest})", flush=True)
    return digest


def _load_probes(output_root: Path) -> tuple[dict[str, Any], str]:
    body, digest = hot._read_json_artifact(output_root / PROBES_NAME)  # noqa: SLF001
    rows = body.get("questions")
    sources = body.get("source_bindings")
    _assert_gold_free_rows(rows)
    if (
        body.get("format") != PROBE_FORMAT
        or body.get("status") != "sealed_gold_free_locked_full100_probes"
        or body.get("population_identity_sha256") != EXPECTED_POPULATION_SHA256
        or body.get("question_count") != EXPECTED_QUESTION_COUNT
        or body.get("shard_count") != 10
        or body.get("retrieval_query_form") != hot.RETRIEVAL_QUERY_FORM
        or body.get("gold_fields_present") is not False
        or body.get("provider_calls") != 0
        or not isinstance(rows, list)
        or not isinstance(sources, list)
        or len(rows) != EXPECTED_QUESTION_COUNT
        or len(sources) != 10
    ):
        raise ValueError("full100 probe artifact changed")
    population = validate_locked_cumulative_population_identity(
        body.get("population_identity", {}),
        plan=LOCKED_LONGMEMEVAL_VALIDATION_PLAN,
    )
    if population.get("population_identity_sha256") != EXPECTED_POPULATION_SHA256:
        raise ValueError("full100 probe population changed")
    if [row.get("ordinal") for row in rows] != list(range(EXPECTED_QUESTION_COUNT)):
        raise ValueError("full100 probe order changed")
    if [row.get("shard_offset") for row in sources] != list(LOCKED_100Q_OFFSETS):
        raise ValueError("full100 source order changed")
    for ordinal, row in enumerate(rows):
        if set(row) != _PROBE_ROW_FIELDS:
            raise ValueError("full100 probe row schema changed")
        question_id_sha256 = identity_sha256(
            {"question_id": str(row.get("question_id", ""))}
        )
        locked_probe_body = {
            "format": QUESTION_PROBE_FORMAT,
            "ordinal": ordinal % 10,
            "question_id_sha256": question_id_sha256,
            "retrieval_query_sha256": row.get("retrieval_query_sha256"),
            "prompt_question_sha256": row.get("prompt_question_sha256"),
        }
        expected = dict(row)
        observed = expected.pop("probe_sha256", None)
        if (
            observed != identity_sha256(expected)
            or row.get("retrieval_query_sha256")
            != quote_sha256(str(row.get("retrieval_query", "")))
            or row.get("prompt_question_sha256")
            != quote_sha256(str(row.get("prompt_question", "")))
            or row.get("shard_offset") != (ordinal // 10) * 10
            or row.get("local_ordinal") != ordinal % 10
            or question_id_sha256 != population["ordered_question_id_sha256s"][ordinal]
            or row.get("locked_probe_identity_sha256")
            != population["ordered_question_probe_sha256s"][ordinal]
            or row.get("locked_probe_identity_sha256")
            != identity_sha256(locked_probe_body)
        ):
            raise ValueError("full100 probe row changed")
    return body, digest


def _compiled_root(output_root: Path, offset: int) -> Path:
    return output_root / "compiled" / f"offset-{offset:03d}"


def _compile_one(binding: SourceBinding, output_root: Path) -> tuple[dict[str, Any], int]:
    root = _compiled_root(output_root, binding.offset)
    manifest_path = root / "compiled.json"
    if manifest_path.exists():
        manifest, digest = hot._read_json_artifact(manifest_path)  # noqa: SLF001
        if (
            manifest.get("format") != COMPILED_SHARD_FORMAT
            or manifest.get("source_binding") != binding.artifact_row()
            or manifest.get("implementation") != _implementation_identity()
        ):
            raise ValueError(f"offset {binding.offset:03d} compiled shard is foreign")
        loaded = _load_compiled_shard(root.parents[1], binding.offset, digest)
        try:
            pass
        finally:
            loaded.dense_index.close()
            del loaded
            gc.collect()
        return {"shard_offset": binding.offset, "compiled_sha256": digest}, 0

    started = time.perf_counter_ns()
    rows: list[dict[str, Any]] = []
    matrix = np.empty((binding.chunk_count, DEFAULT_MODEL_DIM), dtype=np.float32)
    with Database(binding.database_path, read_only=True) as database:
        stats = LexicalIndex(database).stats()
        cursor = database.execute(
            "SELECT c.chunk_id, c.embedding, c.text, c.token_count, c.turn_id, "
            + TURN_SOURCE_ID_SQL
            + ", t.role, t.created_at, t.ordinal, c.start_char FROM chunks AS c "
            "JOIN turns AS t ON t.turn_id = c.turn_id WHERE "
            + INDEXED_CHUNK_SQL
            + " ORDER BY c.chunk_id"
        )
        for row_index, raw in enumerate(cursor):
            if row_index >= binding.chunk_count:
                raise RuntimeError("source contains more chunks than its receipt")
            vector = np.frombuffer(raw[1], dtype=np.float32)
            if vector.shape != (DEFAULT_MODEL_DIM,) or not np.isfinite(vector).all():
                raise RuntimeError(f"chunk {raw[0]!r} has an invalid dense address")
            norm = float(np.linalg.norm(vector))
            if norm <= 0.0:
                raise RuntimeError(f"chunk {raw[0]!r} has a zero dense address")
            matrix[row_index] = vector / norm
            role = str(raw[6])
            event = compile_event_chunk_features(role=role, text=str(raw[2]))
            rows.append(
                {
                    "chunk_id": str(raw[0]),
                    "token_count": int(raw[3]),
                    "turn_id": str(raw[4]),
                    "source_id": str(raw[5]),
                    "role": role,
                    "created_at": None if raw[7] is None else str(raw[7]),
                    "ordinal": int(raw[8]),
                    "start_char": int(raw[9]),
                    "event_first_person": event.first_person,
                    "event_fixed_completed": event.fixed_completed_event,
                    "event_ed_verbs": list(event.ed_verbs),
                }
            )
    if len(rows) != binding.chunk_count:
        raise RuntimeError(
            f"offset {binding.offset:03d} yielded {len(rows)} chunks, "
            f"expected {binding.chunk_count}"
        )
    chunk_ids = [row["chunk_id"] for row in rows]
    if chunk_ids != sorted(set(chunk_ids)):
        raise RuntimeError("compiled chunk IDs are not unique and ordered")

    dense_path = root / "dense-f32.npy"
    dense_sha = hot._publish_numpy(dense_path, matrix)  # noqa: SLF001
    chunks = {
        "format": COMPILED_CHUNKS_FORMAT,
        "shard_offset": binding.offset,
        "source_receipt_sha256": binding.receipt_sha256,
        "chunk_count": len(rows),
        "chunk_sequence_sha256": identity_sha256(chunk_ids),
        "rows": rows,
        "gold_fields_present": False,
    }
    chunks_path = root / "chunks.json"
    chunks_sha = hot._atomic_write_json(chunks_path, chunks)  # noqa: SLF001
    manifest = {
        "format": COMPILED_SHARD_FORMAT,
        "status": "query_independent_provider_free_addresses",
        "shard_offset": binding.offset,
        "source_binding": binding.artifact_row(),
        "embedding_identity": binding.embedding_identity,
        "dense_address": {
            "path": "dense-f32.npy",
            "sha256": dense_sha,
            "bytes": dense_path.stat().st_size,
            "dtype": "float32",
            "shape": [binding.chunk_count, DEFAULT_MODEL_DIM],
            "row_l2_normalized": True,
            "chunk_sequence_sha256": chunks["chunk_sequence_sha256"],
        },
        "chunk_manifest": {
            "path": "chunks.json",
            "sha256": chunks_sha,
            "bytes": chunks_path.stat().st_size,
            "contains_raw_text": False,
        },
        "lexical_stats": {
            key: hot._score_float(value) for key, value in stats.items()  # noqa: SLF001
        },
        "policy_id": POLICY_ID,
        "implementation": _implementation_identity(),
        "gold_fields_present": False,
        "qwen_calls": 0,
        "provider_calls": 0,
    }
    manifest_sha = hot._atomic_write_json(manifest_path, manifest)  # noqa: SLF001
    elapsed = time.perf_counter_ns() - started
    return {"shard_offset": binding.offset, "compiled_sha256": manifest_sha}, elapsed


def compile_catalog(*, source_root: Path, output_root: Path) -> str:
    probes, probes_sha = _load_probes(output_root)
    identities = probes["population_identity"]["ordered_shard_identity_sha256s"]
    catalog_path = output_root / CATALOG_NAME
    if catalog_path.exists():
        catalog, digest = _load_catalog(output_root)
        for source_row, row, offset in zip(
            probes["source_bindings"],
            catalog["shards"],
            LOCKED_100Q_OFFSETS,
            strict=True,
        ):
            binding = _load_source_binding(
                source_root,
                offset,
                expected_shard_identity=None,
                expected_shard_identity_sha256=str(identities[offset // 10]),
                verify_large_files=True,
            )
            if binding.artifact_row() != source_row:
                raise ValueError(f"offset {offset:03d} cached source binding changed")
            loaded = _load_compiled_shard(
                output_root,
                offset,
                str(row["compiled_sha256"]),
            )
            try:
                if loaded.manifest.get("source_binding") != binding.artifact_row():
                    raise ValueError(
                        f"offset {offset:03d} cached compiled source changed"
                    )
            finally:
                loaded.dense_index.close()
        runtime, _runtime_sha = hot._read_json_artifact(  # noqa: SLF001
            output_root / COMPILE_RUNTIME_NAME
        )
        if (
            runtime.get("format") != COMPILE_RUNTIME_FORMAT
            or runtime.get("compiled_catalog_sha256") != digest
            or runtime.get("qwen_calls") != 0
            or runtime.get("provider_calls") != 0
        ):
            raise ValueError("compiled catalog completion receipt changed")
        print(f"Full100 compiled catalog verified: {catalog_path} ({digest})", flush=True)
        return digest
    entries: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    total_started = time.perf_counter_ns()
    for source_row, offset in zip(
        probes["source_bindings"], LOCKED_100Q_OFFSETS, strict=True
    ):
        binding = _load_source_binding(
            source_root,
            offset,
            expected_shard_identity=None,
            expected_shard_identity_sha256=str(identities[offset // 10]),
            verify_large_files=True,
        )
        if binding.artifact_row() != source_row:
            raise ValueError(f"offset {offset:03d} source binding changed")
        entry, elapsed = _compile_one(binding, output_root)
        entry["shard_identity_sha256"] = identities[offset // 10]
        entries.append(entry)
        runtime_rows.append(
            {
                "shard_offset": offset,
                "elapsed_ns": elapsed,
                "chunk_count": binding.chunk_count,
                "cache_hit": elapsed == 0,
            }
        )
        print(
            f"Compiled offset-{offset:03d}: {binding.chunk_count} chunks "
            f"in {elapsed / 1e9:.3f}s",
            flush=True,
        )
    catalog = {
        "format": COMPILED_CATALOG_FORMAT,
        "status": "ten_query_independent_hot_namespaces",
        "probes_sha256": probes_sha,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "shards": entries,
        "implementation": _implementation_identity(),
        "gold_fields_present": False,
        "qwen_calls": 0,
        "provider_calls": 0,
    }
    digest = hot._atomic_write_json(catalog_path, catalog)  # noqa: SLF001
    runtime = {
        "format": COMPILE_RUNTIME_FORMAT,
        "compiled_catalog_sha256": digest,
        "elapsed_ns": time.perf_counter_ns() - total_started,
        "shards": runtime_rows,
        "qwen_calls": 0,
        "provider_calls": 0,
    }
    hot._atomic_write_json(output_root / COMPILE_RUNTIME_NAME, runtime)  # noqa: SLF001
    print(f"Full100 compiled catalog published: {catalog_path} ({digest})", flush=True)
    return digest


def _load_catalog(output_root: Path) -> tuple[dict[str, Any], str]:
    body, digest = hot._read_json_artifact(output_root / CATALOG_NAME)  # noqa: SLF001
    _probes, probes_sha = _load_probes(output_root)
    shards = body.get("shards")
    if (
        body.get("format") != COMPILED_CATALOG_FORMAT
        or body.get("status") != "ten_query_independent_hot_namespaces"
        or body.get("population_identity_sha256") != EXPECTED_POPULATION_SHA256
        or body.get("probes_sha256") != probes_sha
        or body.get("implementation") != _implementation_identity()
        or body.get("gold_fields_present") is not False
        or body.get("qwen_calls") != 0
        or body.get("provider_calls") != 0
        or not isinstance(shards, list)
        or [row.get("shard_offset") for row in shards] != list(LOCKED_100Q_OFFSETS)
    ):
        raise ValueError("compiled full100 catalog changed")
    return body, digest


@dataclass(slots=True)
class LoadedCompiledShard:
    manifest: dict[str, Any]
    manifest_sha256: str
    metadata_by_id: dict[str, dict[str, Any]]
    dense_index: ExactDenseAddressIndex
    neighborhood_index: SourceNeighborhoodIndex


def _load_compiled_shard(
    output_root: Path,
    offset: int,
    expected_sha256: str,
) -> LoadedCompiledShard:
    root = _compiled_root(output_root, offset)
    manifest, digest = hot._read_json_artifact(root / "compiled.json")  # noqa: SLF001
    if digest != expected_sha256 or manifest.get("format") != COMPILED_SHARD_FORMAT:
        raise ValueError(f"offset {offset:03d} compiled manifest changed")
    dense = manifest.get("dense_address")
    chunks_ref = manifest.get("chunk_manifest")
    binding = manifest.get("source_binding")
    if not all(isinstance(value, Mapping) for value in (dense, chunks_ref, binding)):
        raise ValueError("compiled shard omitted a binding")
    count = int(binding["chunk_count"])
    dense_path = hot._safe_relative(  # noqa: SLF001
        root, str(dense["path"]), label=f"offset {offset:03d} dense address"
    )
    chunks_path = hot._safe_relative(  # noqa: SLF001
        root, str(chunks_ref["path"]), label=f"offset {offset:03d} chunk manifest"
    )
    if (
        manifest.get("shard_offset") != offset
        or manifest.get("implementation") != _implementation_identity()
        or manifest.get("policy_id") != POLICY_ID
        or manifest.get("gold_fields_present") is not False
        or manifest.get("qwen_calls") != 0
        or manifest.get("provider_calls") != 0
    ):
        raise ValueError(f"offset {offset:03d} compiled contract changed")
    if (
        file_sha256(dense_path) != dense.get("sha256")
        or dense.get("shape") != [count, DEFAULT_MODEL_DIM]
        or dense.get("row_l2_normalized") is not True
    ):
        raise ValueError("compiled dense address changed")
    chunks, chunks_sha = hot._read_json_artifact(chunks_path)  # noqa: SLF001
    rows = chunks.get("rows")
    if (
        chunks_sha != chunks_ref.get("sha256")
        or chunks.get("format") != COMPILED_CHUNKS_FORMAT
        or chunks.get("shard_offset") != offset
        or chunks.get("source_receipt_sha256")
        != binding.get("source_receipt_sha256")
        or chunks.get("chunk_count") != count
        or chunks.get("gold_fields_present") is not False
        or not isinstance(rows, list)
        or len(rows) != count
    ):
        raise ValueError("compiled chunk coordinates changed")
    chunk_ids = [str(row["chunk_id"]) for row in rows]
    if (
        chunk_ids != sorted(set(chunk_ids))
        or identity_sha256(chunk_ids) != dense.get("chunk_sequence_sha256")
        or identity_sha256(chunk_ids) != chunks.get("chunk_sequence_sha256")
    ):
        raise ValueError("compiled dense/chunk row binding changed")
    metadata = {str(row["chunk_id"]): dict(row) for row in rows}
    dense_index = ExactDenseAddressIndex.open(chunk_ids, dense_path)
    neighborhood = SourceNeighborhoodIndex(
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
    return LoadedCompiledShard(manifest, digest, metadata, dense_index, neighborhood)


def _timing_summary(values: Sequence[int]) -> dict[str, int | float]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize empty timings")
    p95 = ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": p95,
        "max": ordered[-1],
    }


def _collect(
    *,
    source_root: Path,
    output_root: Path,
    device: str,
    warmup_rounds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    probes, _probes_sha = _load_probes(output_root)
    catalog, _catalog_sha = _load_catalog(output_root)
    identities = probes["population_identity"]["ordered_shard_identity_sha256s"]
    questions_by_offset = {
        offset: [
            row for row in probes["questions"] if row["shard_offset"] == offset
        ]
        for offset in LOCKED_100Q_OFFSETS
    }
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    encoder_started = time.perf_counter_ns()
    embedder: EmbeddingService | None = None
    semantic_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    setup_rows: list[dict[str, Any]] = []
    try:
        embedder = EmbeddingService(device=device)
        hot._cuda_synchronize(device)  # noqa: SLF001
        embedder.embed_query("provider-free hot full100 calibration")
        hot._cuda_synchronize(device)  # noqa: SLF001
        encoder_first_touch_ns = time.perf_counter_ns() - encoder_started
        if (
            getattr(embedder, "_verified_checkpoint_sha256", None)
            != BGE_M3_CHECKPOINT_SHA256
        ):
            raise RuntimeError(
                "query encoder did not verify the pinned BGE-M3 checkpoint"
            )
        for source_row, catalog_row, offset in zip(
            probes["source_bindings"], catalog["shards"], LOCKED_100Q_OFFSETS, strict=True
        ):
            setup_started = time.perf_counter_ns()
            binding = _load_source_binding(
                source_root,
                offset,
                expected_shard_identity=None,
                expected_shard_identity_sha256=str(identities[offset // 10]),
                verify_large_files=True,
            )
            if binding.artifact_row() != source_row:
                raise ValueError(f"offset {offset:03d} source changed after prepare")
            verified_at = time.perf_counter_ns()
            compiled = _load_compiled_shard(
                output_root, offset, str(catalog_row["compiled_sha256"])
            )
            compiled_at = time.perf_counter_ns()
            if compiled.manifest.get("source_binding") != binding.artifact_row():
                raise ValueError(
                    f"offset {offset:03d} compiled/raw source binding changed"
                )
            database: Database | None = None
            try:
                database = Database(binding.database_path, read_only=True)
                durable = LexicalIndex(database)
                lexical = ResidentBM25Index(durable)
                resident_at = time.perf_counter_ns()
                if {
                    key: hot._score_float(value)  # noqa: SLF001
                    for key, value in lexical.stats().items()
                } != compiled.manifest["lexical_stats"]:
                    raise RuntimeError(f"offset {offset:03d} resident BM25 changed")
                shard_questions = questions_by_offset[offset]
                with ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="hot-full100-bm25",
                ) as executor:
                    for _round in range(warmup_rounds):
                        for probe in shard_questions:
                            hot._execute_question(  # noqa: SLF001
                                row=probe,
                                embedder=embedder,
                                dense_index=compiled.dense_index,
                                source_neighborhood_index=compiled.neighborhood_index,
                                lexical=lexical,
                                database=database,
                                metadata_by_id=compiled.metadata_by_id,
                                executor=executor,
                                device=device,
                                lane_budget=LANE_BUDGET,
                                candidates_per_lane=CANDIDATES_PER_LANE,
                                max_context_tokens=MAX_CONTEXT_TOKENS,
                                max_prompt_tokens=MAX_PROMPT_TOKENS,
                                include_diagnostic_arms=False,
                            )
                    measured_started = time.perf_counter_ns()
                    for probe in shard_questions:
                        semantic, timings = hot._execute_question(  # noqa: SLF001
                            row=probe,
                            embedder=embedder,
                            dense_index=compiled.dense_index,
                            source_neighborhood_index=compiled.neighborhood_index,
                            lexical=lexical,
                            database=database,
                            metadata_by_id=compiled.metadata_by_id,
                            executor=executor,
                            device=device,
                            lane_budget=LANE_BUDGET,
                            candidates_per_lane=CANDIDATES_PER_LANE,
                            max_context_tokens=MAX_CONTEXT_TOKENS,
                            max_prompt_tokens=MAX_PROMPT_TOKENS,
                            include_diagnostic_arms=False,
                        )
                        semantic["shard_offset"] = offset
                        semantic["local_ordinal"] = int(probe["local_ordinal"])
                        semantic_rows.append(semantic)
                        timing_rows.append(
                            {
                                "ordinal": int(probe["ordinal"]),
                                "question_id": str(probe["question_id"]),
                                "shard_offset": offset,
                                "timings_ns": timings,
                            }
                        )
                    measured_ns = time.perf_counter_ns() - measured_started
                setup_rows.append(
                    {
                        "shard_offset": offset,
                        "source_integrity_ns": verified_at - setup_started,
                        "compiled_integrity_and_mmap_ns": compiled_at - verified_at,
                        "database_open_and_resident_bm25_ns": resident_at - compiled_at,
                        "warmup_rounds": warmup_rounds,
                        "measured_questions": len(shard_questions),
                        "measured_ns": measured_ns,
                        "resident_bm25_numeric_bytes": lexical.resident_bytes,
                        "resident_source_neighborhood_approx_bytes": (
                            compiled.neighborhood_index.approx_resident_bytes
                        ),
                        "dense_address_bytes": compiled.dense_index.nbytes,
                    }
                )
            finally:
                if database is not None:
                    database.close()
                compiled.dense_index.close()
                del compiled
                gc.collect()
            print(f"Retrieved offset-{offset:03d}: 10/10", flush=True)
    finally:
        if embedder is not None:
            embedder.close()
    semantic_rows.sort(key=lambda row: int(row["ordinal"]))
    timing_rows.sort(key=lambda row: int(row["ordinal"]))
    if [row["ordinal"] for row in semantic_rows] != list(range(EXPECTED_QUESTION_COUNT)):
        raise RuntimeError("retrieval did not produce the complete ordered full100")
    model = {
        "model_id": embedder.model_name,
        "model_revision": embedder.model_revision,
        "checkpoint_sha256": embedder.checkpoint_sha256,
        "execution_identity": embedder.execution_identity,
        "one_query_encoder_forward_per_question": True,
        "one_encoder_resident_across_all_namespaces": True,
        "first_touch_ns": encoder_first_touch_ns,
    }
    return semantic_rows, timing_rows, setup_rows, model


def run(
    *,
    source_root: Path,
    output_root: Path,
    device: str,
    warmup_rounds: int,
) -> str:
    selection_path = output_root / SELECTION_NAME
    if selection_path.exists():
        _selection, digest = _load_selection(output_root)
        runtime, _runtime_sha = hot._read_json_artifact(  # noqa: SLF001
            output_root / RUNTIME_NAME
        )
        if (
            runtime.get("format") != RUNTIME_FORMAT
            or runtime.get("selection_sha256") != digest
            or runtime.get("population_identity_sha256")
            != EXPECTED_POPULATION_SHA256
            or runtime.get("qwen_calls") != 0
            or runtime.get("provider_calls") != 0
            or len(runtime.get("samples", [])) != EXPECTED_QUESTION_COUNT
        ):
            raise ValueError("full100 selection completion receipt changed")
        print(f"Full100 selection verified: {selection_path} ({digest})", flush=True)
        return digest
    probes, probes_sha = _load_probes(output_root)
    catalog, catalog_sha = _load_catalog(output_root)
    started = time.perf_counter_ns()
    questions, timing_rows, setup_rows, model = _collect(
        source_root=source_root,
        output_root=output_root,
        device=device,
        warmup_rounds=warmup_rounds,
    )
    controls = {
        "policy_id": POLICY_ID,
        "candidate_limit_per_lane": CANDIDATES_PER_LANE,
        "lane_budgets": {
            "bm25": LANE_BUDGET,
            "exact_dense": LANE_BUDGET,
            "source_neighborhood": hot.SOURCE_NEIGHBOR_LANE_BUDGET,
            "temporal_event": hot.TEMPORAL_EVENT_LANE_BUDGET,
        },
        "lane_order": ["bm25", "exact_dense", "source_neighborhood", "temporal_event"],
        "temporal_evidence_window_policy": (
            "dated-query-calendar-lookback-before-lane-selection-v1"
        ),
        "deduplication": (
            "exact_chunk_id_after_independent_lane_selection_with_same_lane_refill"
        ),
        "max_context_token_proxy": MAX_CONTEXT_TOKENS,
        "max_prompt_workspace_token_proxy": MAX_PROMPT_TOKENS,
        "raw_chunk_hydration": True,
        "retrieval_query_form": hot.RETRIEVAL_QUERY_FORM,
        "query_encoder_device": str(device).casefold(),
    }
    selection = {
        "format": SELECTION_FORMAT,
        "status": "sealed_gold_blind_locked_full100_frozen_v6_candidate",
        "bindings": {
            "probes_sha256": probes_sha,
            "compiled_catalog_sha256": catalog_sha,
            "population_identity_sha256": EXPECTED_POPULATION_SHA256,
            "ordered_compiled_shard_sha256s": [
                row["compiled_sha256"] for row in catalog["shards"]
            ],
        },
        "corpus": {
            "shard_count": 10,
            "question_count": EXPECTED_QUESTION_COUNT,
            "total_transcript_tokens": probes["population_identity"][
                "total_transcript_tokens"
            ],
            "total_turn_count": probes["population_identity"]["total_turn_count"],
            "chunk_count_by_shard": [
                row["chunk_count"] for row in probes["source_bindings"]
            ],
        },
        "controls": controls,
        "model_binding": model,
        "implementation": _implementation_identity(),
        "questions": questions,
        "gold_fields_present": False,
        "retained_request_token_state_bytes": 0,
        "qwen_calls": 0,
        "responder_calls": 0,
        "judge_calls": 0,
        "provider_calls": 0,
    }
    selection_sha = hot._atomic_write_json(selection_path, selection)  # noqa: SLF001
    latency_values = [
        int(row["timings_ns"]["a3_prompt_to_serialized_bytes_ns"])
        for row in timing_rows
    ]
    summary = _timing_summary(latency_values)
    runtime = {
        "format": RUNTIME_FORMAT,
        "status": "one_warm_measurement_per_locked_question",
        "selection_sha256": selection_sha,
        "population_identity_sha256": EXPECTED_POPULATION_SHA256,
        "total_elapsed_ns": time.perf_counter_ns() - started,
        "warmup_rounds_per_namespace": warmup_rounds,
        "samples": timing_rows,
        "a3_prompt_to_serialized_bytes_ns": summary,
        "a3_p50_milliseconds": float(summary["p50"]) / 1e6,
        "a3_p95_milliseconds": int(summary["p95"]) / 1e6,
        "per_namespace_setup": setup_rows,
        "provider_rtt_prefill_decode_excluded": True,
        "artifact_publication_excluded_from_samples": True,
        "qwen_calls": 0,
        "provider_calls": 0,
    }
    runtime_sha = hot._atomic_write_json(output_root / RUNTIME_NAME, runtime)  # noqa: SLF001
    print(
        f"Full100 selection published: {selection_sha}; runtime={runtime_sha}; "
        f"p95={runtime['a3_p95_milliseconds']:.3f}ms",
        flush=True,
    )
    return selection_sha


def _load_selection(output_root: Path) -> tuple[dict[str, Any], str]:
    body, digest = hot._read_json_artifact(output_root / SELECTION_NAME)  # noqa: SLF001
    rows = body.get("questions")
    bindings = body.get("bindings")
    if (
        body.get("format") != SELECTION_FORMAT
        or body.get("status") != "sealed_gold_blind_locked_full100_frozen_v6_candidate"
        or body.get("implementation") != _implementation_identity()
        or body.get("gold_fields_present") is not False
        or body.get("retained_request_token_state_bytes") != 0
        or any(body.get(key) != 0 for key in ("qwen_calls", "responder_calls", "judge_calls", "provider_calls"))
        or not isinstance(bindings, Mapping)
        or bindings.get("population_identity_sha256") != EXPECTED_POPULATION_SHA256
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_QUESTION_COUNT
        or [row.get("ordinal") for row in rows] != list(range(EXPECTED_QUESTION_COUNT))
    ):
        raise ValueError("full100 selection changed")
    probes, probes_sha = _load_probes(output_root)
    catalog, catalog_sha = _load_catalog(output_root)
    if (
        bindings.get("probes_sha256") != probes_sha
        or bindings.get("compiled_catalog_sha256") != catalog_sha
        or bindings.get("ordered_compiled_shard_sha256s")
        != [row["compiled_sha256"] for row in catalog["shards"]]
    ):
        raise ValueError("full100 selection input binding changed")
    controls = body.get("controls")
    if not isinstance(controls, Mapping) or (
        controls.get("policy_id") != POLICY_ID
        or controls.get("max_context_token_proxy") != MAX_CONTEXT_TOKENS
        or controls.get("max_prompt_workspace_token_proxy") != MAX_PROMPT_TOKENS
    ):
        raise ValueError("full100 selection controls changed")
    for row, probe in zip(rows, probes["questions"], strict=True):
        if any(
            row.get(key) != probe.get(key)
            for key in (
                "ordinal",
                "shard_offset",
                "local_ordinal",
                "question_id",
                "probe_sha256",
                "retrieval_query_sha256",
                "prompt_question_sha256",
            )
        ):
            raise ValueError("full100 selection question binding changed")
        arm = row.get("arms", {}).get("a3_protected_union")
        if not isinstance(arm, Mapping) or arm.get("raw_evidence_only") is not True:
            raise ValueError("full100 selection omitted a raw A3 packet")
        payload = hot._canonical_json_bytes(  # noqa: SLF001
            {"messages": arm.get("provider_messages")}
        )
        if (
            arm.get("provider_payload_sha256") != hashlib.sha256(payload).hexdigest()
            or arm.get("provider_payload_utf8_bytes") != len(payload)
            or arm.get("prompt_workspace_token_proxy", 0) > MAX_PROMPT_TOKENS
        ):
            raise ValueError("full100 provider packet binding changed")
        hot._validate_arm_payload(  # noqa: SLF001
            arm,
            prompt_question=str(probe["prompt_question"]),
            max_context_tokens=MAX_CONTEXT_TOKENS,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
        )
    return body, digest


def replay(*, source_root: Path, output_root: Path, device: str) -> str:
    expected, selection_sha = _load_selection(output_root)
    questions, _timings, _setups, _model = _collect(
        source_root=source_root,
        output_root=output_root,
        device=device,
        warmup_rounds=0,
    )
    expected_bytes = hot._canonical_json_bytes(expected["questions"])  # noqa: SLF001
    replayed_bytes = hot._canonical_json_bytes(questions)  # noqa: SLF001
    if replayed_bytes != expected_bytes:
        raise RuntimeError("full100 gold-blind replay differs from sealed selection")
    body = {
        "format": REPLAY_FORMAT,
        "status": "byte_identical_gold_blind_full100_replay",
        "selection_sha256": selection_sha,
        "question_population_sha256": hashlib.sha256(expected_bytes).hexdigest(),
        "replayed_question_population_sha256": hashlib.sha256(replayed_bytes).hexdigest(),
        "byte_identical": True,
        "question_count": EXPECTED_QUESTION_COUNT,
        "gold_fields_present": False,
        "qwen_calls": 0,
        "provider_calls": 0,
    }
    digest = hot._atomic_write_json(output_root / REPLAY_NAME, body)  # noqa: SLF001
    print(f"Full100 replay published: {digest}; byte_identical=true", flush=True)
    return digest


def _flatten_questions(samples: Sequence[Any]) -> list[Any]:
    return [question for sample in samples for question in sample.questions]


def score(
    *,
    dataset: Path,
    split_manifest: Path,
    output_root: Path,
) -> str:
    selection, selection_sha = _load_selection(output_root)
    replay_body, replay_sha = hot._read_json_artifact(output_root / REPLAY_NAME)  # noqa: SLF001
    expected_question_population_sha = hashlib.sha256(
        hot._canonical_json_bytes(selection["questions"])  # noqa: SLF001
    ).hexdigest()
    if (
        replay_body.get("format") != REPLAY_FORMAT
        or replay_body.get("status")
        != "byte_identical_gold_blind_full100_replay"
        or replay_body.get("selection_sha256") != selection_sha
        or replay_body.get("question_population_sha256")
        != replay_body.get("replayed_question_population_sha256")
        or replay_body.get("question_population_sha256")
        != expected_question_population_sha
        or replay_body.get("byte_identical") is not True
        or replay_body.get("question_count") != EXPECTED_QUESTION_COUNT
        or replay_body.get("gold_fields_present") is not False
        or replay_body.get("qwen_calls") != 0
        or replay_body.get("provider_calls") != 0
    ):
        raise ValueError("score requires the sealed byte-identical replay")
    samples, _identities, population = _load_population(dataset, split_manifest)
    questions = _flatten_questions(samples)
    rows: list[dict[str, Any]] = []
    for selected, question in zip(selection["questions"], questions, strict=True):
        if (
            selected.get("question_id") != question.question_id
            or selected.get("retrieval_query_sha256") != quote_sha256(question.question)
            or selected.get("prompt_question_sha256") != quote_sha256(question.dated_question)
        ):
            raise ValueError("selection differs from locked gold population")
        arm = selected["arms"]["a3_protected_union"]
        evidence = arm["packed_evidence"]
        texts = [str(item["raw_text"]) for item in evidence]
        retrieved_sources = tuple(
            dict.fromkeys(str(item["source_id"]) for item in evidence if item.get("source_id"))
        )
        expected_sources = tuple(dict.fromkeys(str(value) for value in question.evidence_sources))
        expected_set = set(expected_sources)
        source_recall = (
            None
            if not expected_set
            else len(expected_set & set(retrieved_sources)) / len(expected_set)
        )
        components = answer_value_component_coverage(
            question.answer, len(expected_sources), texts
        )
        rows.append(
            {
                "ordinal": selected["ordinal"],
                "question_id": question.question_id,
                "category": question.category,
                "packed_count": len(evidence),
                "context_token_proxy": arm["context_token_proxy"],
                "prompt_workspace_token_proxy": arm["prompt_workspace_token_proxy"],
                "answer_present": contains_answer(texts, question.answer),
                "best_evidence_f1": best_f1(texts, question.answer),
                "evidence_source_recall": source_recall,
                "all_evidence_sources": None if source_recall is None else source_recall == 1.0,
                "answer_value_component_recall": (
                    None if components is None else components.recall
                ),
                "all_answer_value_components": (
                    None if components is None else components.all_components
                ),
            }
        )
    recalls = [float(row["evidence_source_recall"]) for row in rows if row["evidence_source_recall"] is not None]
    component_recalls = [
        float(row["answer_value_component_recall"])
        for row in rows
        if row["answer_value_component_recall"] is not None
    ]
    body = {
        "format": SCORE_FORMAT,
        "status": "locked_full100_retrieval_diagnostics_not_answer_accuracy",
        "selection_sha256": selection_sha,
        "replay_sha256": replay_sha,
        "population_identity_sha256": population["population_identity_sha256"],
        "questions": rows,
        "aggregate": {
            "question_count": len(rows),
            "packed_all_evidence_source_hits": sum(row["all_evidence_sources"] is True for row in rows),
            "packed_literal_answer_hits": sum(bool(row["answer_present"]) for row in rows),
            "mean_best_evidence_f1": statistics.fmean(float(row["best_evidence_f1"]) for row in rows),
            "mean_evidence_source_recall": None if not recalls else statistics.fmean(recalls),
            "mean_answer_value_component_recall": (
                None if not component_recalls else statistics.fmean(component_recalls)
            ),
            "max_context_token_proxy": max(int(row["context_token_proxy"]) for row in rows),
            "max_prompt_workspace_token_proxy": max(int(row["prompt_workspace_token_proxy"]) for row in rows),
        },
        "gold_fields_present": True,
        "responder_calls": 0,
        "judge_calls": 0,
        "provider_calls": 0,
    }
    digest = hot._atomic_write_json(output_root / SCORE_NAME, body)  # noqa: SLF001
    print(
        f"Full100 retrieval score published: {digest}; "
        f"sources={body['aggregate']['packed_all_evidence_source_hits']}/100; "
        f"literal={body['aggregate']['packed_literal_answer_hits']}/100",
        flush=True,
    )
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--dataset", type=Path, required=True)
    prepare_parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    commands.add_parser("compile")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--warmup-rounds", type=int, default=1)
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("--device", default="cuda")
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--dataset", type=Path, required=True)
    score_parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.resolve()
    source_root = args.source_root.resolve()
    if args.command == "prepare":
        prepare(
            dataset=args.dataset.resolve(),
            split_manifest=args.split_manifest.resolve(),
            source_root=source_root,
            output_root=output_root,
        )
    elif args.command == "compile":
        compile_catalog(source_root=source_root, output_root=output_root)
    elif args.command == "run":
        if args.warmup_rounds < 0:
            raise ValueError("warmup rounds cannot be negative")
        run(
            source_root=source_root,
            output_root=output_root,
            device=args.device,
            warmup_rounds=args.warmup_rounds,
        )
    elif args.command == "replay":
        replay(source_root=source_root, output_root=output_root, device=args.device)
    elif args.command == "score":
        score(
            dataset=args.dataset.resolve(),
            split_manifest=args.split_manifest.resolve(),
            output_root=output_root,
        )
    else:  # pragma: no cover
        raise AssertionError(f"unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
