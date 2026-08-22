"""Deterministic, resumable exact-span source preparation for the 1M route."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from memory_condense.domain.discourse import identity_sha256
from memory_condense.domain.integrity import file_sha256
from memory_condense.eval.diffuse_longmemeval_base import (
    DATABASE_NAME,
    INDEX_NAME,
    STORE_DIRECTORY_NAME,
    DiffuseBaseTreatmentIdentity,
    owned_build_runtime_identity,
    publish_diffuse_longmemeval_base,
    verify_diffuse_longmemeval_base,
)
from memory_condense.eval.diffuse_longmemeval_runtime import (
    DiffuseLongMemEvalExecutionBinding,
    DiffuseLongMemEvalRuntimeConfig,
    DiffuseLongMemEvalRuntimeFactories,
    gold_blind_from_treatment_sample,
)
from memory_condense.eval.schemas import EvalConfig, RetrievalConfig
from memory_condense.ingest.loader import BenchmarkSample


CURRENT_SOURCE_TIMESTAMP_SEMANTICS = (
    "exact_longmemeval_dataset_session_timestamps_v1"
)
CURRENT_SOURCE_SCOPE = (
    "gold_blind_haystack_store_with_separately_addressed_question_probes"
)
CURRENT_SOURCE_FORMAT = (
    "memory-condense-recall-guarded-current-exact-span-source-v1"
)
CURRENT_SOURCE_SELECTION_NAME = "source-current-selection.json"


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


def _write_selection(path: Path, value: object) -> str:
    payload = _canonical_json_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
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
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar.exists():
        if sidecar.read_bytes() != expected:
            raise FileExistsError(f"refusing to replace another digest: {sidecar}")
    else:
        sidecar.write_bytes(expected)
    return digest


def _read_selection(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != _canonical_json_bytes(payload):
        raise ValueError("current-source selection is not canonical JSON")
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if not sidecar.is_file() or sidecar.read_bytes() != expected:
        raise ValueError("current-source selection digest is missing or invalid")
    return payload


def source_acquisition_config(config: EvalConfig) -> EvalConfig:
    """Select a direct mode only for the separately addressed query bundle."""

    return config.model_copy(
        update={"retrieval": RetrievalConfig(mode="dense", k=10)}
    )


def source_treatment_identity(
    sample: BenchmarkSample,
    *,
    dataset_sha256: str,
    split_manifest_sha256: str,
    sanitized_projection_sha256: str,
) -> DiffuseBaseTreatmentIdentity:
    return DiffuseBaseTreatmentIdentity(
        treatment_file_sha256=dataset_sha256,
        sanitized_projection_sha256=sanitized_projection_sha256,
        dataset_sha256=dataset_sha256,
        split_manifest_sha256=split_manifest_sha256,
        ordered_question_ids_sha256=identity_sha256(
            [
                {
                    "question_id_sha256": identity_sha256(
                        {"question_id": question.question_id}
                    )
                }
                for question in sample.questions
            ]
        ),
        sample_count=1,
        sample_ordinal=0,
    )


def current_source_binding(
    config: EvalConfig,
    *,
    qwen_model_dir: Path,
) -> tuple[EvalConfig, DiffuseLongMemEvalExecutionBinding]:
    source_config = source_acquisition_config(config)
    binding = DiffuseLongMemEvalExecutionBinding(
        config=source_config,
        runtime=DiffuseLongMemEvalRuntimeConfig(
            qwen_model_dir=qwen_model_dir,
            residency_mode="staged_bge_then_qwen",
            embedding_batch_size=32,
        ),
        factories=DiffuseLongMemEvalRuntimeFactories(),
    )
    return source_config, binding


def _source_receipt(base: Any, *, source_root: Path) -> dict[str, object]:
    root = source_root.resolve()
    store = base.store_manifest
    query = base.query_manifest
    if Path(base.store_path).resolve() != root / "stores" / store.base_store_key:
        raise RuntimeError("verified current source escaped its store address")
    if Path(base.query_inputs_path).resolve() != (
        root / "query-inputs" / query.query_input_key
    ):
        raise RuntimeError("verified current source escaped its query address")
    body: dict[str, object] = {
        "format": CURRENT_SOURCE_FORMAT,
        "source_scope": CURRENT_SOURCE_SCOPE,
        "timestamp_semantics": CURRENT_SOURCE_TIMESTAMP_SEMANTICS,
        "base_store_key": store.base_store_key,
        "selected_store_entry": f"stores/{store.base_store_key}",
        "store_manifest_sha256": base.store_manifest_sha256,
        "store_artifact_sha256": store.artifact_sha256,
        "database_sha256": store.database_sha256,
        "index_sha256": store.index_sha256,
        "corpus_sha256": store.corpus_sha256,
        "turn_count": store.turn_count,
        "chunk_count": store.chunk_count,
        "deterministic_turn_ids_sha256": store.deterministic_turn_ids_sha256,
        "turn_sequence_sha256": store.turn_sequence_sha256,
        "chunk_sequence_sha256": store.chunk_sequence_sha256,
        "source_streams_sha256": store.source_streams_sha256,
        "embedding_identity": store.embedding_identity.model_dump(mode="json"),
        "embedding_identity_sha256": store.embedding_identity_sha256,
        "build_runtime_identity_sha256": store.build_runtime_identity_sha256,
        "implementation_sha256": store.implementation_sha256,
        "environment_lock_sha256": store.environment_lock_sha256,
        "query_input_key": query.query_input_key,
        "selected_query_entry": f"query-inputs/{query.query_input_key}",
        "query_manifest_sha256": base.query_manifest_sha256,
        "query_artifact_sha256": query.artifact_sha256,
    }
    body["receipt_sha256"] = identity_sha256(body)
    return body


def validate_current_source_receipt(
    receipt: object,
    *,
    sample: BenchmarkSample,
    expected_device: str = "cuda",
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise ValueError("retrieval artifact omitted its current-source receipt")
    value = dict(receipt)
    receipt_sha = value.pop("receipt_sha256", None)
    blind = gold_blind_from_treatment_sample(sample)
    embedding = value.get("embedding_identity")
    if not isinstance(embedding, Mapping):
        raise ValueError("current-source receipt omitted its BGE identity")
    from memory_condense.modeling.embedding import (
        BGE_M3_CHECKPOINT_SHA256,
        DEFAULT_MODEL_DIM,
        DEFAULT_MODEL_NAME,
        DEFAULT_MODEL_REVISION,
    )

    expected_embedding = {
        "backend": "sentence-transformers.encode-v1",
        "model_id": DEFAULT_MODEL_NAME,
        "model_revision": DEFAULT_MODEL_REVISION,
        "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
        "dimension": DEFAULT_MODEL_DIM,
        "device": str(expected_device).casefold(),
        "batch_size": 32,
        "normalize_embeddings": False,
        "output_dtype": "float32",
    }
    if (
        value.get("format") != CURRENT_SOURCE_FORMAT
        or value.get("source_scope") != CURRENT_SOURCE_SCOPE
        or value.get("timestamp_semantics") != CURRENT_SOURCE_TIMESTAMP_SEMANTICS
        or value.get("corpus_sha256") != blind.corpus_sha256
        or value.get("turn_count") != len(blind.turns)
        or value.get("chunk_count", 0) < 1
        or dict(embedding) != expected_embedding
        or receipt_sha != identity_sha256(value)
    ):
        raise ValueError("current-source receipt belongs to another corpus/runtime")
    digest_names = (
        "base_store_key", "store_manifest_sha256", "store_artifact_sha256",
        "database_sha256", "index_sha256", "deterministic_turn_ids_sha256",
        "turn_sequence_sha256", "chunk_sequence_sha256", "source_streams_sha256",
        "embedding_identity_sha256", "build_runtime_identity_sha256",
        "implementation_sha256", "environment_lock_sha256", "query_input_key",
        "query_manifest_sha256", "query_artifact_sha256",
    )
    for name in digest_names:
        raw = value.get(name)
        if not isinstance(raw, str) or len(raw) != 64 or any(
            character not in "0123456789abcdef" for character in raw
        ):
            raise ValueError(f"current-source receipt has invalid {name}")
    if value.get("selected_store_entry") != f"stores/{value['base_store_key']}" or (
        value.get("selected_query_entry")
        != f"query-inputs/{value['query_input_key']}"
    ):
        raise ValueError("current-source receipt changed its selected entry")
    return {**value, "receipt_sha256": receipt_sha}


def prepare_current_source_store(
    *,
    sample: BenchmarkSample,
    config: EvalConfig,
    treatment_identity: DiffuseBaseTreatmentIdentity,
    binding: DiffuseLongMemEvalExecutionBinding,
    source_root: Path,
    selection_path: Path,
) -> tuple[Path, dict[str, Any], str]:
    """Publish once; a declared selection makes every later run verify-only."""

    blind = gold_blind_from_treatment_sample(sample)
    build_runtime = owned_build_runtime_identity(binding.new_condenser)
    existing = _read_selection(selection_path) if selection_path.exists() else None
    if existing is not None:
        base = verify_diffuse_longmemeval_base(
            source_root,
            treatment_identity=treatment_identity,
            sample=blind,
            config=config,
            embedding_identity=binding.embedding_identity,
            build_runtime_identity=build_runtime,
            # The declared selection is an immutable address for an already
            # published source.  Verify it against the implementation and
            # environment that created it; retrieval-only source edits must
            # not silently retarget this expensive, content-addressed ingest.
            implementation_digest=str(existing["implementation_sha256"]),
            environment_digest=str(existing["environment_lock_sha256"]),
        )
        mode = "verified_cache_hit"
    else:
        base = publish_diffuse_longmemeval_base(
            source_root,
            treatment_identity=treatment_identity,
            sample=blind,
            config=config,
            embedding_identity=binding.embedding_identity,
            build_runtime_identity=build_runtime,
            embedder=binding.embedder,
            condenser_factory=binding.new_condenser,
        )
        mode = "fresh_or_recovered_atomic_publication"
    receipt = _source_receipt(base, source_root=source_root)
    validate_current_source_receipt(
        receipt,
        sample=sample,
        expected_device=str(config.embedding_device),
    )
    if existing is not None and existing != receipt:
        raise ValueError("current-source selection changed after verification")
    _write_selection(selection_path, receipt)
    store_dir = Path(base.store_path) / STORE_DIRECTORY_NAME
    database_path, index_path = store_dir / DATABASE_NAME, store_dir / INDEX_NAME
    if file_sha256(database_path) != receipt["database_sha256"] or (
        file_sha256(index_path) != receipt["index_sha256"]
    ):
        raise RuntimeError("current source changed after exhaustive verification")
    return database_path, receipt, mode


__all__ = [
    "CURRENT_SOURCE_FORMAT",
    "CURRENT_SOURCE_SCOPE",
    "CURRENT_SOURCE_SELECTION_NAME",
    "CURRENT_SOURCE_TIMESTAMP_SEMANTICS",
    "current_source_binding",
    "prepare_current_source_store",
    "source_acquisition_config",
    "source_treatment_identity",
    "validate_current_source_receipt",
]
