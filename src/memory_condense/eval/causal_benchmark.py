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

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.integrity import file_sha256 as _file_sha256
from memory_condense.application.condenser import MemoryCondenser, query_facets
from memory_condense.search.packing.context_packer import ContextBudget
from memory_condense.modeling.embedding import DEFAULT_MODEL_NAME, EmbeddingService
from memory_condense.persistence.db import CURRENT_SCHEMA_VERSION
from memory_condense.eval.benchmark import IngestFn, ingest_sample
from memory_condense.eval.compiled_cache import (
    _embedding_execution_identity,
    _embedding_identity,
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
from memory_condense.eval.cache_receipts import canonical_sha256
from memory_condense.eval.reproducibility import (
    environment_lock_sha256,
    implementation_sha256,
)
from memory_condense.ingest.loader import BenchmarkSample


CAUSAL_CACHE_FORMAT = "memory-condense-causal-benchmark-store-v1"
CAUSAL_CACHE_REVISION = 3
CAUSAL_MANIFEST_NAME = "causal-store.json"
CAUSAL_BUILD_PROTOCOL = "causal-training-query-only-v1"


def _held_out_query_batch(
    sample: BenchmarkSample,
    config: EvalConfig,
) -> list[str]:
    """Held-out questions plus any deterministic bounded retrieval facets."""

    queries: list[str] = []
    for question in sample.questions:
        queries.append(question.dated_question)
        if config.retrieval.query_facet_retrieval:
            queries.extend(
                query_facets(
                    question.dated_question,
                    max_facets=config.retrieval.query_facet_max,
                )
            )
    return list(dict.fromkeys(queries))


def _causal_cache_key(
    sample: BenchmarkSample,
    config: EvalConfig,
    *,
    embedding_model: str,
    embedding_dim: int,
    embedding_execution: dict[str, str | int | bool],
    implementation_digest: str,
    environment_digest: str,
) -> str:
    retrieval = config.retrieval
    payload = {
        "format": CAUSAL_CACHE_FORMAT,
        "revision": CAUSAL_CACHE_REVISION,
        "build_protocol": CAUSAL_BUILD_PROTOCOL,
        "sample_sha256": sample_sha256(sample),
        "chunker": config.chunker.model_dump(mode="json"),
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "embedding_execution": embedding_execution,
        "implementation_sha256": implementation_digest,
        "environment_lock_sha256": environment_digest,
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
    expected_sample_sha256: str,
    expected_embedding_execution: dict[str, str | int | bool],
    expected_implementation_sha256: str,
    expected_environment_lock_sha256: str,
) -> dict[str, Any]:
    manifest_path = store_dir / CAUSAL_MANIFEST_NAME
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid causal-store manifest: {manifest_path}") from exc
    if payload.get("format") != CAUSAL_CACHE_FORMAT:
        raise RuntimeError(f"causal-store format mismatch: {manifest_path}")
    if payload.get("cache_revision") != CAUSAL_CACHE_REVISION:
        raise RuntimeError(f"causal-store revision mismatch: {manifest_path}")
    if payload.get("build_protocol") != CAUSAL_BUILD_PROTOCOL:
        raise RuntimeError(f"causal-store build protocol mismatch: {manifest_path}")
    if payload.get("cache_key") != expected_key:
        raise RuntimeError(f"causal-store key mismatch: {manifest_path}")
    if payload.get("sample_sha256") != expected_sample_sha256:
        raise RuntimeError(f"causal-store sample mismatch: {manifest_path}")
    if payload.get("embedding_execution") != expected_embedding_execution:
        raise RuntimeError(
            f"causal-store embedding execution mismatch: {manifest_path}"
        )
    if payload.get("implementation_sha256") != expected_implementation_sha256:
        raise RuntimeError(f"causal-store implementation mismatch: {manifest_path}")
    if payload.get("environment_lock_sha256") != expected_environment_lock_sha256:
        raise RuntimeError(f"causal-store environment mismatch: {manifest_path}")
    database_path = store_dir / "memory.db"
    index_path = store_dir / "hnsw_index.bin"
    if _file_sha256(database_path) != payload.get("database_sha256"):
        raise RuntimeError(f"causal-store SQLite hash mismatch: {store_dir}")
    if _file_sha256(index_path) != payload.get("index_sha256"):
        raise RuntimeError(f"causal-store ANN hash mismatch: {store_dir}")
    return payload


def _causal_manifest_receipt(
    store_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, str]:
    """Return only hashes from one verified active causal artifact."""

    return {
        "manifest_sha256": _file_sha256(store_dir / CAUSAL_MANIFEST_NAME),
        "cache_key": str(manifest["cache_key"]),
        "sample_sha256": str(manifest["sample_sha256"]),
        "compiled_cache_key": str(manifest["compiled_cache_key"]),
        "database_sha256": str(manifest["database_sha256"]),
        "index_sha256": str(manifest["index_sha256"]),
        "build_protocol_sha256": hashlib.sha256(
            str(manifest["build_protocol"]).encode("utf-8")
        ).hexdigest(),
        "embedding_execution_sha256": canonical_sha256(
            manifest["embedding_execution"]
        ),
        "implementation_sha256": str(manifest["implementation_sha256"]),
        "environment_lock_sha256": str(manifest["environment_lock_sha256"]),
    }


def _set_cache_receipts(
    store: MemoryCondenser,
    *,
    compiled: dict[str, str | int] | None,
    causal_dir: Path | None,
    causal_manifest: dict[str, Any] | None,
) -> None:
    """Attach exact text-free artifact receipts for the CLI reporter."""

    causal = (
        _causal_manifest_receipt(causal_dir, causal_manifest)
        if causal_dir is not None and causal_manifest is not None
        else None
    )
    store.blind_cache_receipts = (  # type: ignore[attr-defined]
        {
            "compiled": [dict(compiled)],
            "causal": [causal],
        }
        if compiled is not None and causal is not None
        else {}
    )


def causal_consolidation_ingest_fn(
    cache_root: str | Path | None = None,
    *,
    causal_cache_root: str | Path | None = None,
    device: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    embedder: Any | None = None,
    prepare_only: bool = False,
    require_cache_hit: bool = False,
    implementation_digest: str | None = None,
    environment_digest: str | None = None,
) -> IngestFn:
    """Create sample-local learned stores while reusing compiled embeddings.

    Historical user prompts are embedded in one training-only batch before
    causal replay. Held-out questions are embedded separately, only after the
    durable artifact is closed, and only for an actual evaluation reader. The
    returned store uses a frozen lookup, so replay cannot accidentally embed
    unseen text or retain a transformer workspace.
    ``prepare_only`` is the blind cache-construction seam: it runs the same
    compiled ingest, causal staging, learning, publication, and verification
    path but never reads or embeds held-out questions.  The returned store is
    only suitable for immediate close; normal evaluation reopens the cache
    with its own frozen held-out query batch.
    """

    if prepare_only and require_cache_hit:
        raise ValueError("prepare_only and require_cache_hit are mutually exclusive")
    active_implementation_digest = (
        implementation_digest or implementation_sha256()
    ).casefold()
    active_environment_digest = (
        environment_digest or environment_lock_sha256()
    ).casefold()
    owns_shared_embedder = embedder is None
    shared_embedder = embedder or EmbeddingService(
        model_name=model_name,
        device=device,
    )
    try:
        compiled_ingest = (
            compiled_store_ingest_fn(
                cache_root,
                embedder=shared_embedder,
                require_cache_hit=require_cache_hit,
                implementation_digest=active_implementation_digest,
                environment_digest=active_environment_digest,
            )
            if cache_root is not None
            else None
        )
        learned_root = (
            Path(causal_cache_root) if causal_cache_root is not None else None
        )
        if require_cache_hit and learned_root is None:
            raise RuntimeError("validation requires a causal-store cache root")
        if require_cache_hit and compiled_ingest is None:
            raise RuntimeError("validation requires a compiled-store cache root")
        if require_cache_hit and learned_root is not None and not learned_root.is_dir():
            raise RuntimeError(
                f"required causal-store cache root does not exist: {learned_root}"
            )
        if learned_root is not None and not require_cache_hit:
            learned_root.mkdir(parents=True, exist_ok=True)
        embedding_model, embedding_dim = _embedding_identity(shared_embedder)
        embedding_execution = _embedding_execution_identity(shared_embedder)
    except BaseException:
        # A cache-root or embedding-identity failure happens before ``ingest``
        # exists, so its ordinary ``release_embedder`` seam cannot run. Close
        # only an embedder allocated by this factory; injected embedders remain
        # owned by their caller.
        if owns_shared_embedder:
            close = getattr(shared_embedder, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        raise

    def ingest(
        sample: BenchmarkSample,
        config: EvalConfig,
        data_dir: Path,
    ) -> MemoryCondenser:
        if config.retrieval.mode not in {"causal_consolidation", "causal_graph"}:
            raise ValueError(
                "causal consolidation ingest requires a causal retrieval mode"
            )
        sample_digest = sample_sha256(sample)
        cache_key = _causal_cache_key(
            sample,
            config,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            embedding_execution=embedding_execution,
            implementation_digest=active_implementation_digest,
            environment_digest=active_environment_digest,
        )
        cached_target = (
            _causal_store_dir(learned_root, sample, cache_key)
            if learned_root is not None
            else None
        )
        cached_manifest = (
            _verified_causal_manifest(
                cached_target,
                expected_key=cache_key,
                expected_sample_sha256=sample_digest,
                expected_embedding_execution=embedding_execution,
                expected_implementation_sha256=active_implementation_digest,
                expected_environment_lock_sha256=active_environment_digest,
            )
            if cached_target is not None and cached_target.exists()
            else None
        )
        if require_cache_hit and cached_target is not None and not cached_target.exists():
            raise RuntimeError(
                f"required causal-store cache entry is missing: {cached_target}"
            )
        final_queries = [] if prepare_only else _held_out_query_batch(sample, config)

        def open_source_store() -> tuple[Path, dict[str, str | int] | None]:
            """Ensure the compiled source exists and close its read handle."""

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
                raw_receipt = getattr(
                    source_store,
                    "compiled_cache_receipt",
                    None,
                )
                compiled_receipt = (
                    dict(raw_receipt) if isinstance(raw_receipt, dict) else None
                )
            finally:
                source_store.close()
            return source_db, compiled_receipt

        def runtime_embedder():
            """Embed held-out probes only after the durable build is closed."""

            if prepare_only or not final_queries:
                return shared_embedder
            vectors = shared_embedder.embed_queries(final_queries)
            return FrozenQueryEmbedder(
                dict(zip(final_queries, vectors, strict=True))
            )

        def verify_compiled_link(
            manifest: dict[str, Any],
            receipt: dict[str, str | int] | None,
        ) -> None:
            if receipt is None:
                return
            if manifest.get("compiled_cache_key") != receipt.get("cache_key"):
                raise RuntimeError(
                    "causal-store compiled cache identity mismatch"
                )

        if cached_manifest is not None:
            # A blind preparation receipt promises both cache layers. A causal
            # hit must not short-circuit past a missing compiled artifact.
            compiled_receipt = None
            if (prepare_only or require_cache_hit) and compiled_ingest is not None:
                _source_db, compiled_receipt = open_source_store()
                verify_compiled_link(cached_manifest, compiled_receipt)
            if require_cache_hit and compiled_receipt is None:
                raise RuntimeError(
                    "required causal-store hit has no compiled-cache receipt"
                )
            store = MemoryCondenser(
                data_dir=cached_target,
                chunker_min_tokens=config.chunker.min_tokens,
                chunker_max_tokens=config.chunker.max_tokens,
                auto_extract=False,
                budget=_read_budget(config),
                embedder=runtime_embedder(),
                persist_index_on_close=False,
                read_only=True,
            )
            store.causal_consolidation_stats = cached_manifest[  # type: ignore[attr-defined]
                "stats"
            ]
            _set_cache_receipts(
                store,
                compiled=compiled_receipt,
                causal_dir=cached_target,
                causal_manifest=cached_manifest,
            )
            return store

        source_db, compiled_receipt = open_source_store()

        retrieval = config.retrieval
        prompt_limit = retrieval.consolidation_max_training_prompt_tokens
        training_queries = [
            text
            for text in _source_user_queries(source_db)
            if count_tokens(text) <= prompt_limit
        ]
        training_queries = list(dict.fromkeys(training_queries))
        if not training_queries:
            raise ValueError(
                "causal benchmark sample has no bounded training query text"
            )
        training_vectors = shared_embedder.embed_queries(training_queries)
        training_embedder = FrozenQueryEmbedder(
            dict(zip(training_queries, training_vectors, strict=True))
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
                training_embedder,
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
            learning = apply_rank_learning(
                build_target,
                training_embedder,
                events,
            )
            stats = {"staging": staging, "learning": learning}
            active_manifest = None
            if cached_target is not None:
                manifest = {
                    "format": CAUSAL_CACHE_FORMAT,
                    "cache_revision": CAUSAL_CACHE_REVISION,
                    "build_protocol": CAUSAL_BUILD_PROTOCOL,
                    "cache_key": cache_key,
                    "sample_id": sample.sample_id,
                    "sample_sha256": sample_digest,
                    "embedding_execution": embedding_execution,
                    "implementation_sha256": active_implementation_digest,
                    "environment_lock_sha256": active_environment_digest,
                    "compiled_cache_key": (
                        compiled_receipt.get("cache_key")
                        if compiled_receipt is not None
                        else None
                    ),
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
                        expected_sample_sha256=sample_digest,
                        expected_embedding_execution=embedding_execution,
                        expected_implementation_sha256=active_implementation_digest,
                        expected_environment_lock_sha256=active_environment_digest,
                    )
                    shutil.rmtree(build_target)
                build_target = cached_target
                if temporary_root is not None:
                    temporary_root.rmdir()
                # Bind the returned store and receipt to the actual winner of
                # a concurrent publication race.
                active_manifest = _verified_causal_manifest(
                    cached_target,
                    expected_key=cache_key,
                    expected_sample_sha256=sample_digest,
                    expected_embedding_execution=embedding_execution,
                    expected_implementation_sha256=active_implementation_digest,
                    expected_environment_lock_sha256=active_environment_digest,
                )
                verify_compiled_link(active_manifest, compiled_receipt)
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
            embedder=runtime_embedder(),
            persist_index_on_close=False,
            retriever_max_elements=max(1, int(staging["source_turns"])),
            read_only=cached_target is not None,
        )
        # Kept on the ephemeral store for diagnostic runners; only scalar
        # counts/timings are exposed, never prompts or activation tensors.
        effective_stats = (
            active_manifest["stats"]
            if active_manifest is not None
            else {"staging": staging, "learning": learning}
        )
        store.causal_consolidation_stats = effective_stats  # type: ignore[attr-defined]
        _set_cache_receipts(
            store,
            compiled=compiled_receipt,
            causal_dir=cached_target,
            causal_manifest=active_manifest,
        )
        return store

    # Evaluation controls that need the same small GPU can stage after the
    # held-out queries have been embedded into ``FrozenQueryEmbedder``. Function
    # attributes keep this optional seam out of the production condenser API.
    release_embedder = getattr(shared_embedder, "close", None)
    if callable(release_embedder):
        ingest.release_embedder = release_embedder  # type: ignore[attr-defined]
    ingest.prepare_only = prepare_only  # type: ignore[attr-defined]
    ingest.require_cache_hit = require_cache_hit  # type: ignore[attr-defined]
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
        source_metadata_expansions=(
            retrieval.consolidation_source_metadata_packing
        ),
    )
