"""Isolated causal-consolidation stores for LongMemEval/LoCoMo.

The authoritative compiled store remains immutable.  For each benchmark
sample we replay its chunks chronologically into scratch storage, learn only
bounded scalar associations from completed prompt/response episodes, and then
answer the held-out benchmark question without teaching on it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from memory_condense._tokenizer import count_tokens
from memory_condense.condenser import MemoryCondenser
from memory_condense.context_packer import ContextBudget
from memory_condense.embedding import DEFAULT_MODEL_NAME, EmbeddingService
from memory_condense.db import CURRENT_SCHEMA_VERSION
from memory_condense.eval.benchmark import IngestFn, ingest_sample
from memory_condense.eval.compiled_cache import (
    compiled_store_ingest_fn,
    sample_sha256,
)
from memory_condense.eval.consolidation_replay import (
    FrozenQueryEmbedder,
    _source_user_queries,
    apply_rank_learning,
    stage_causal_store,
)
from memory_condense.eval.schemas import EvalConfig
from memory_condense.loader import BenchmarkSample


CAUSAL_CACHE_FORMAT = "memory-condense-causal-benchmark-store-v1"
CAUSAL_CACHE_REVISION = 1
CAUSAL_MANIFEST_NAME = "causal-store.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _causal_cache_key(
    sample: BenchmarkSample,
    config: EvalConfig,
    *,
    embedding_model: str,
    embedding_dim: int,
) -> str:
    retrieval = config.retrieval
    payload = {
        "format": CAUSAL_CACHE_FORMAT,
        "revision": CAUSAL_CACHE_REVISION,
        "sample_sha256": sample_sha256(sample),
        "chunker": config.chunker.model_dump(mode="json"),
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "write_policy": {
            "expansion_tokens": (
                retrieval.consolidation_training_expansion_tokens
            ),
            "training_k": retrieval.consolidation_training_k,
            "max_event_nodes": retrieval.consolidation_max_event_nodes,
            "new_event_nodes": retrieval.consolidation_new_event_nodes,
            "max_training_prompt_tokens": (
                retrieval.consolidation_max_training_prompt_tokens
            ),
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _causal_store_dir(root: Path, sample: BenchmarkSample, key: str) -> Path:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", sample.sample_id).strip("-._")
    return root / f"{label or 'sample'}-{key[:16]}"


def _verified_causal_manifest(
    store_dir: Path,
    *,
    expected_key: str,
) -> dict[str, Any]:
    manifest_path = store_dir / CAUSAL_MANIFEST_NAME
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid causal-store manifest: {manifest_path}") from exc
    if payload.get("format") != CAUSAL_CACHE_FORMAT:
        raise RuntimeError(f"causal-store format mismatch: {manifest_path}")
    if payload.get("cache_key") != expected_key:
        raise RuntimeError(f"causal-store key mismatch: {manifest_path}")
    database_path = store_dir / "memory.db"
    index_path = store_dir / "hnsw_index.bin"
    if _file_sha256(database_path) != payload.get("database_sha256"):
        raise RuntimeError(f"causal-store SQLite hash mismatch: {store_dir}")
    if _file_sha256(index_path) != payload.get("index_sha256"):
        raise RuntimeError(f"causal-store ANN hash mismatch: {store_dir}")
    return payload


def causal_consolidation_ingest_fn(
    cache_root: str | Path | None = None,
    *,
    causal_cache_root: str | Path | None = None,
    device: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    embedder: Any | None = None,
) -> IngestFn:
    """Create sample-local learned stores while reusing compiled embeddings.

    User prompts and held-out questions are embedded once per sample as a
    batch.  The returned store uses that frozen query lookup, so causal replay
    cannot accidentally embed unseen text or retain a transformer workspace.
    """

    shared_embedder = embedder or EmbeddingService(
        model_name=model_name,
        device=device,
    )
    compiled_ingest = (
        compiled_store_ingest_fn(cache_root, embedder=shared_embedder)
        if cache_root is not None
        else None
    )
    learned_root = Path(causal_cache_root) if causal_cache_root is not None else None
    if learned_root is not None:
        learned_root.mkdir(parents=True, exist_ok=True)
    embedding_model = str(
        getattr(shared_embedder, "model_name", type(shared_embedder).__qualname__)
    )
    embedding_dim = int(shared_embedder.dim)

    def ingest(
        sample: BenchmarkSample,
        config: EvalConfig,
        data_dir: Path,
    ) -> MemoryCondenser:
        if config.retrieval.mode not in {"causal_consolidation", "causal_graph"}:
            raise ValueError(
                "causal consolidation ingest requires a causal retrieval mode"
            )
        cache_key = _causal_cache_key(
            sample,
            config,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )
        cached_target = (
            _causal_store_dir(learned_root, sample, cache_key)
            if learned_root is not None
            else None
        )
        cached_manifest = (
            _verified_causal_manifest(cached_target, expected_key=cache_key)
            if cached_target is not None and cached_target.exists()
            else None
        )
        final_queries = [question.dated_question for question in sample.questions]
        if cached_manifest is not None:
            vectors = shared_embedder.embed_queries(final_queries)
            frozen_embedder = FrozenQueryEmbedder(
                dict(zip(final_queries, vectors, strict=True))
            )
            store = MemoryCondenser(
                data_dir=cached_target,
                chunker_min_tokens=config.chunker.min_tokens,
                chunker_max_tokens=config.chunker.max_tokens,
                auto_extract=False,
                budget=_read_budget(config),
                embedder=frozen_embedder,
                persist_index_on_close=False,
            )
            store.causal_consolidation_stats = cached_manifest[  # type: ignore[attr-defined]
                "stats"
            ]
            return store

        source_dir = data_dir.with_name(f"{data_dir.name}.compiled-source")
        source_store = (
            compiled_ingest(sample, config, source_dir)
            if compiled_ingest is not None
            else ingest_sample(
                sample,
                config,
                source_dir,
                embedder=shared_embedder,
            )
        )
        try:
            source_db = source_store.database_path
        finally:
            source_store.close()

        retrieval = config.retrieval
        prompt_limit = retrieval.consolidation_max_training_prompt_tokens
        training_queries = [
            text
            for text in _source_user_queries(source_db)
            if count_tokens(text) <= prompt_limit
        ]
        query_texts = list(dict.fromkeys([*training_queries, *final_queries]))
        if not query_texts:
            raise ValueError("causal benchmark sample has no retrievable query text")
        vectors = shared_embedder.embed_queries(query_texts)
        frozen_embedder = FrozenQueryEmbedder(
            dict(zip(query_texts, vectors, strict=True))
        )

        temporary_root = (
            Path(tempfile.mkdtemp(prefix=".building-", dir=learned_root))
            if learned_root is not None
            else None
        )
        build_target = temporary_root / "store" if temporary_root else data_dir
        try:
            staging_started = time.perf_counter()
            events, staging = stage_causal_store(
                source_db,
                build_target,
                frozen_embedder,
                expansion_tokens=(
                    retrieval.consolidation_training_expansion_tokens
                ),
                retrieval_k=retrieval.consolidation_training_k,
                max_event_nodes=retrieval.consolidation_max_event_nodes,
                new_event_nodes=retrieval.consolidation_new_event_nodes,
                max_prompt_tokens=(
                    retrieval.consolidation_max_training_prompt_tokens
                ),
            )
            staging["elapsed_s"] = time.perf_counter() - staging_started
            learning = apply_rank_learning(build_target, frozen_embedder, events)
            stats = {"staging": staging, "learning": learning}
            if cached_target is not None:
                manifest = {
                    "format": CAUSAL_CACHE_FORMAT,
                    "cache_revision": CAUSAL_CACHE_REVISION,
                    "cache_key": cache_key,
                    "sample_id": sample.sample_id,
                    "sample_sha256": sample_sha256(sample),
                    "database_sha256": _file_sha256(build_target / "memory.db"),
                    "index_sha256": _file_sha256(build_target / "hnsw_index.bin"),
                    "stats": stats,
                }
                (build_target / CAUSAL_MANIFEST_NAME).write_text(
                    json.dumps(manifest, indent=2),
                    encoding="utf-8",
                )
                try:
                    build_target.rename(cached_target)
                except FileExistsError:
                    _verified_causal_manifest(
                        cached_target,
                        expected_key=cache_key,
                    )
                    shutil.rmtree(build_target)
                build_target = cached_target
                if temporary_root is not None:
                    temporary_root.rmdir()
        except BaseException:
            if temporary_root is not None and temporary_root.exists():
                shutil.rmtree(temporary_root, ignore_errors=True)
            raise
        store = MemoryCondenser(
            data_dir=build_target,
            chunker_min_tokens=config.chunker.min_tokens,
            chunker_max_tokens=config.chunker.max_tokens,
            auto_extract=False,
            budget=_read_budget(config),
            embedder=frozen_embedder,
            persist_index_on_close=False,
            retriever_max_elements=max(1, int(staging["source_turns"])),
        )
        # Kept on the ephemeral store for diagnostic runners; only scalar
        # counts/timings are exposed, never prompts or activation tensors.
        store.causal_consolidation_stats = {  # type: ignore[attr-defined]
            "staging": staging,
            "learning": learning,
        }
        return store

    return ingest


def _read_budget(config: EvalConfig) -> ContextBudget:
    retrieval = config.retrieval
    direct_slots = retrieval.k
    if retrieval.mode == "causal_graph":
        direct_slots += retrieval.neighbor_slots + retrieval.source_slots
    return ContextBudget(
        recent_window_tokens=0,
        memory_header_tokens=0,
        expansion_tokens=retrieval.consolidation_expansion_tokens,
        max_expansions=direct_slots,
        max_consolidation_expansions=retrieval.consolidation_chunk_slots,
        budget_aware_expansions=retrieval.consolidation_budget_aware_packing,
        source_diverse_expansions=(
            retrieval.consolidation_source_diverse_packing
        ),
        query_aware_sentence_expansions=(
            retrieval.consolidation_query_aware_sentence_packing
        ),
        max_sentences_per_expansion=(
            retrieval.consolidation_max_sentences_per_expansion
        ),
        information_gain_expansions=(
            retrieval.consolidation_information_gain_packing
        ),
        min_information_gain_per_token=(
            retrieval.consolidation_min_information_gain_per_token
        ),
    )
