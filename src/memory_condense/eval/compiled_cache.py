"""Content-addressed compiled benchmark stores for repeatable read sweeps."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from memory_condense.condenser import MemoryCondenser
from memory_condense.db import CURRENT_SCHEMA_VERSION, Database
from memory_condense.embedding import DEFAULT_MODEL_NAME, EmbeddingService
from memory_condense.eval.benchmark import IngestFn, ingest_sample
from memory_condense.eval.cache_receipts import canonical_sha256
from memory_condense.eval.reproducibility import (
    environment_lock_sha256,
    implementation_sha256,
)
from memory_condense.eval.sample_identity import sample_sha256
from memory_condense.eval.schemas import EvalConfig
from memory_condense.loader import BenchmarkSample


CACHE_FORMAT = "memory-condense-compiled-benchmark-store-v1"
CACHE_REVISION = 3
MANIFEST_NAME = "compiled-store.json"


class CompiledStoreCacheError(RuntimeError):
    """A content-addressed store exists but fails its integrity contract."""


class CompiledStoreManifest(BaseModel):
    format: str = CACHE_FORMAT
    cache_revision: int = CACHE_REVISION
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_id: str
    sample_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunker_min_tokens: int
    chunker_max_tokens: int
    embedding_model: str
    embedding_dim: int
    embedding_execution: dict[str, str | int | bool]
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: int
    turn_count: int
    chunk_count: int
    database_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    model_config = {"frozen": True}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _embedding_identity(embedder: Any) -> tuple[str, int]:
    model_name = str(getattr(embedder, "model_name", type(embedder).__qualname__))
    raw_revision = getattr(embedder, "model_revision", "")
    revision = "" if raw_revision is None else str(raw_revision).strip()
    raw_checkpoint = getattr(embedder, "checkpoint_sha256", "")
    checkpoint = (
        "" if raw_checkpoint is None else str(raw_checkpoint).strip().casefold()
    )
    if revision:
        model_name = f"{model_name}@{revision}"
    if checkpoint:
        model_name = f"{model_name}#{checkpoint}"
    return model_name, int(embedder.dim)


def _embedding_execution_identity(
    embedder: Any,
) -> dict[str, str | int | bool]:
    """Return only vector-affecting runtime controls suitable for a cache key."""

    raw = getattr(embedder, "execution_identity", None)
    if callable(raw):
        raw = raw()
    if isinstance(raw, Mapping):
        identity = dict(raw)
    else:
        configured_device = getattr(embedder, "device", None)
        if configured_device is None:
            configured_device = getattr(embedder, "_device", None)
        batch_size = getattr(embedder, "batch_size", None)
        if batch_size is None:
            batch_size = getattr(embedder, "_batch_size", 0)
        identity = {
            "backend": (
                f"{type(embedder).__module__}.{type(embedder).__qualname__}"
            ),
            "device": str(configured_device or "auto").casefold(),
            "batch_size": int(batch_size or 0),
            "normalize_embeddings": False,
            "output_dtype": "float32",
        }
    allowed = (str, int, bool)
    if not identity or any(
        not isinstance(key, str)
        or not key
        or isinstance(value, float)
        or not isinstance(value, allowed)
        for key, value in identity.items()
    ):
        raise ValueError("embedding execution identity must contain JSON scalars")
    return {
        str(key): value
        for key, value in sorted(identity.items(), key=lambda item: item[0])
    }


def cache_key(
    sample: BenchmarkSample,
    config: EvalConfig,
    *,
    embedding_model: str,
    embedding_dim: int,
    embedding_execution: Mapping[str, str | int | bool] | None = None,
    implementation_digest: str | None = None,
    environment_digest: str | None = None,
) -> str:
    """Address only write-time inputs; retrieval policy does not affect bytes."""
    return _canonical_sha256(
        {
            "format": CACHE_FORMAT,
            "revision": CACHE_REVISION,
            "sample_sha256": sample_sha256(sample),
            "chunker": config.chunker.model_dump(mode="json"),
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim,
            "embedding_execution": dict(embedding_execution or {}),
            "implementation_sha256": (
                implementation_digest or implementation_sha256()
            ),
            "environment_lock_sha256": (
                environment_digest or environment_lock_sha256()
            ),
            "schema_version": CURRENT_SCHEMA_VERSION,
        }
    )


def _store_dir(root: Path, sample: BenchmarkSample, key: str) -> Path:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", sample.sample_id).strip("-._")
    return root / f"{label or 'sample'}-{key[:16]}"


def _manifest_for(
    store_dir: Path,
    sample: BenchmarkSample,
    config: EvalConfig,
    *,
    key: str,
    embedding_model: str,
    embedding_dim: int,
    embedding_execution: dict[str, str | int | bool],
    implementation_digest: str,
    environment_digest: str,
) -> CompiledStoreManifest:
    database_path = store_dir / "memory.db"
    index_path = store_dir / "hnsw_index.bin"
    # The ingest writer has already closed and produced a complete current
    # schema.  Manifest construction is an audit read, not a second writable
    # open that may create WAL/SHM sidecars or run migrations after the bytes
    # we intend to attest have supposedly become immutable.
    with Database(database_path, read_only=True) as db:
        turn_count = int(db.execute("SELECT COUNT(*) FROM turns").fetchone()[0])
        chunk_count = int(db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        schema_version = db.schema_version
    return CompiledStoreManifest(
        cache_key=key,
        sample_id=sample.sample_id,
        sample_sha256=sample_sha256(sample),
        chunker_min_tokens=config.chunker.min_tokens,
        chunker_max_tokens=config.chunker.max_tokens,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        embedding_execution=embedding_execution,
        implementation_sha256=implementation_digest,
        environment_lock_sha256=environment_digest,
        schema_version=schema_version,
        turn_count=turn_count,
        chunk_count=chunk_count,
        database_sha256=_file_sha256(database_path),
        index_sha256=_file_sha256(index_path),
    )


def compiled_manifest_receipt(
    store_dir: Path,
    manifest: CompiledStoreManifest,
) -> dict[str, str | int]:
    """Return the text-free identity of one *verified active* cache entry.

    The caller supplies the manifest returned by ``_load_verified_manifest``;
    unlike a root-directory scan this cannot accidentally attest a stale or
    different-config sibling that merely shares the benchmark sample hash.
    """

    return {
        "manifest_sha256": _file_sha256(store_dir / MANIFEST_NAME),
        "cache_key": manifest.cache_key,
        "sample_sha256": manifest.sample_sha256,
        "database_sha256": manifest.database_sha256,
        "index_sha256": manifest.index_sha256,
        "embedding_execution_sha256": canonical_sha256(
            manifest.embedding_execution
        ),
        "implementation_sha256": manifest.implementation_sha256,
        "environment_lock_sha256": manifest.environment_lock_sha256,
        "turn_count": manifest.turn_count,
        "chunk_count": manifest.chunk_count,
    }


def _load_verified_manifest(
    store_dir: Path,
    *,
    expected_key: str,
    expected_embedding_execution: Mapping[str, str | int | bool] | None = None,
    expected_implementation_sha256: str | None = None,
    expected_environment_lock_sha256: str | None = None,
) -> CompiledStoreManifest:
    manifest_path = store_dir / MANIFEST_NAME
    try:
        manifest = CompiledStoreManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise CompiledStoreCacheError(
            f"invalid compiled-store manifest: {manifest_path}"
        ) from exc
    if manifest.format != CACHE_FORMAT:
        raise CompiledStoreCacheError(
            f"compiled-store format mismatch in {manifest_path}"
        )
    if manifest.cache_revision != CACHE_REVISION:
        raise CompiledStoreCacheError(
            f"compiled-store revision mismatch in {manifest_path}"
        )
    if manifest.cache_key != expected_key:
        raise CompiledStoreCacheError(
            f"compiled-store key mismatch in {manifest_path}"
        )
    if (
        expected_embedding_execution is not None
        and manifest.embedding_execution != dict(expected_embedding_execution)
    ):
        raise CompiledStoreCacheError(
            f"compiled-store embedding execution mismatch in {manifest_path}"
        )
    if (
        expected_implementation_sha256 is not None
        and manifest.implementation_sha256 != expected_implementation_sha256
    ):
        raise CompiledStoreCacheError(
            f"compiled-store implementation mismatch in {manifest_path}"
        )
    if (
        expected_environment_lock_sha256 is not None
        and manifest.environment_lock_sha256 != expected_environment_lock_sha256
    ):
        raise CompiledStoreCacheError(
            f"compiled-store environment mismatch in {manifest_path}"
        )
    database_path = store_dir / "memory.db"
    index_path = store_dir / "hnsw_index.bin"
    if not database_path.is_file() or not index_path.is_file():
        raise CompiledStoreCacheError(f"compiled store is incomplete: {store_dir}")
    if _file_sha256(database_path) != manifest.database_sha256:
        raise CompiledStoreCacheError(
            f"compiled SQLite hash mismatch: {database_path}"
        )
    if _file_sha256(index_path) != manifest.index_sha256:
        raise CompiledStoreCacheError(f"compiled ANN hash mismatch: {index_path}")
    with Database(database_path, read_only=True) as db:
        actual_schema = db.schema_version
        actual_turns = int(
            db.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        )
        actual_chunks = int(
            db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        )
    if (
        actual_schema != CURRENT_SCHEMA_VERSION
        or manifest.schema_version != CURRENT_SCHEMA_VERSION
    ):
        raise CompiledStoreCacheError(
            f"compiled-store schema metadata mismatch: {manifest_path}"
        )
    if actual_turns != manifest.turn_count or actual_chunks != manifest.chunk_count:
        raise CompiledStoreCacheError(
            f"compiled-store count metadata mismatch: {manifest_path}"
        )
    return manifest


def _publish_store(temporary: Path, target: Path, *, expected_key: str) -> None:
    """Publish a complete store, with a Windows-safe manifest-last fallback."""
    try:
        temporary.rename(target)
        return
    except FileExistsError:
        _load_verified_manifest(target, expected_key=expected_key)
        shutil.rmtree(temporary)
        return
    except PermissionError:
        # Windows can reject a directory rename briefly after SQLite/HNSW
        # handles close. Publish files instead, but move the manifest last so
        # readers still cannot mistake a partial target for a valid cache hit.
        try:
            target.mkdir()
        except FileExistsError:
            _load_verified_manifest(target, expected_key=expected_key)
            shutil.rmtree(temporary)
            return
        try:
            for child in temporary.iterdir():
                if child.name != MANIFEST_NAME:
                    child.replace(target / child.name)
            (temporary / MANIFEST_NAME).replace(target / MANIFEST_NAME)
            temporary.rmdir()
        except Exception:
            # The target has no valid manifest unless publication completed.
            # Remove only the exact content-addressed directory created above.
            if target.parent.resolve() == temporary.parent.resolve():
                shutil.rmtree(target, ignore_errors=True)
            raise


def compiled_store_ingest_fn(
    cache_root: str | Path,
    *,
    device: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    embedder: Any | None = None,
    require_cache_hit: bool = False,
    implementation_digest: str | None = None,
    environment_digest: str | None = None,
) -> IngestFn:
    """Return an ingest seam that builds once and reopens verified stores.

    Store directories are content-addressed by the complete sample payload,
    chunker, embedding identity, and database schema. A manifest is written
    only after SQLite and HNSW have closed successfully, then the temporary
    directory is atomically renamed into the cache. Cache hits verify both
    file hashes and open with index persistence disabled, so evaluation reads
    cannot mutate the locked artifact.
    """
    root = Path(cache_root)
    if require_cache_hit:
        if not root.is_dir():
            raise CompiledStoreCacheError(
                f"required compiled-store cache root does not exist: {root}"
            )
    else:
        root.mkdir(parents=True, exist_ok=True)
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
        embedding_model, embedding_dim = _embedding_identity(shared_embedder)
        embedding_execution = _embedding_execution_identity(shared_embedder)
    except BaseException:
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
        _scratch_dir: Path,
    ) -> MemoryCondenser:
        if config.retrieval.mode == "memory":
            raise ValueError(
                "compiled benchmark stores support non-extracting retrieval modes only"
            )
        key = cache_key(
            sample,
            config,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            embedding_execution=embedding_execution,
            implementation_digest=active_implementation_digest,
            environment_digest=active_environment_digest,
        )
        target = _store_dir(root, sample, key)
        if target.exists():
            active_manifest = _load_verified_manifest(
                target,
                expected_key=key,
                expected_embedding_execution=embedding_execution,
                expected_implementation_sha256=active_implementation_digest,
                expected_environment_lock_sha256=active_environment_digest,
            )
        elif require_cache_hit:
            raise CompiledStoreCacheError(
                f"required compiled-store cache entry is missing: {target}"
            )
        else:
            temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=root))
            built: MemoryCondenser | None = None
            try:
                built = ingest_sample(
                    sample,
                    config,
                    temporary,
                    embedder=shared_embedder,
                )
                built.close()
                built = None
                manifest = _manifest_for(
                    temporary,
                    sample,
                    config,
                    key=key,
                    embedding_model=embedding_model,
                    embedding_dim=embedding_dim,
                    embedding_execution=embedding_execution,
                    implementation_digest=active_implementation_digest,
                    environment_digest=active_environment_digest,
                )
                (temporary / MANIFEST_NAME).write_text(
                    manifest.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                _publish_store(temporary, target, expected_key=key)
            except BaseException:
                if built is not None:
                    try:
                        built.close()
                    except Exception:
                        pass
                if temporary.exists():
                    shutil.rmtree(temporary, ignore_errors=True)
                raise
            # Publication can lose a benign race to another complete writer.
            # Always re-read the target so the receipt below identifies the
            # artifact this process actually opens, not its discarded build.
            active_manifest = _load_verified_manifest(
                target,
                expected_key=key,
                expected_embedding_execution=embedding_execution,
                expected_implementation_sha256=active_implementation_digest,
                expected_environment_lock_sha256=active_environment_digest,
            )
        store = MemoryCondenser(
            data_dir=target,
            chunker_min_tokens=config.chunker.min_tokens,
            chunker_max_tokens=config.chunker.max_tokens,
            auto_extract=False,
            embedder=shared_embedder,
            persist_index_on_close=False,
            read_only=True,
        )
        store.compiled_cache_receipt = compiled_manifest_receipt(  # type: ignore[attr-defined]
            target,
            active_manifest,
        )
        return store

    ingest.require_cache_hit = require_cache_hit  # type: ignore[attr-defined]
    return ingest
