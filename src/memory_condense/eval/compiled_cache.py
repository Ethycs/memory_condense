"""Content-addressed compiled benchmark stores for repeatable read sweeps."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from memory_condense.condenser import MemoryCondenser
from memory_condense.db import CURRENT_SCHEMA_VERSION, Database
from memory_condense.embedding import DEFAULT_MODEL_NAME, EmbeddingService
from memory_condense.eval.benchmark import IngestFn, ingest_sample
from memory_condense.eval.schemas import EvalConfig
from memory_condense.loader import BenchmarkSample


CACHE_FORMAT = "memory-condense-compiled-benchmark-store-v1"
CACHE_REVISION = 2
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


def sample_sha256(sample: BenchmarkSample) -> str:
    """Stable identity over all haystack, source, question, and date fields."""
    return _canonical_sha256(sample.model_dump(mode="json"))


def _embedding_identity(embedder: Any) -> tuple[str, int]:
    model_name = str(getattr(embedder, "model_name", type(embedder).__qualname__))
    return model_name, int(embedder.dim)


def cache_key(
    sample: BenchmarkSample,
    config: EvalConfig,
    *,
    embedding_model: str,
    embedding_dim: int,
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
) -> CompiledStoreManifest:
    database_path = store_dir / "memory.db"
    index_path = store_dir / "hnsw_index.bin"
    with Database(database_path) as db:
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
        schema_version=schema_version,
        turn_count=turn_count,
        chunk_count=chunk_count,
        database_sha256=_file_sha256(database_path),
        index_sha256=_file_sha256(index_path),
    )


def _load_verified_manifest(
    store_dir: Path,
    *,
    expected_key: str,
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
    if manifest.cache_key != expected_key:
        raise CompiledStoreCacheError(
            f"compiled-store key mismatch in {manifest_path}"
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
    root.mkdir(parents=True, exist_ok=True)
    shared_embedder = embedder or EmbeddingService(model_name=model_name, device=device)
    embedding_model, embedding_dim = _embedding_identity(shared_embedder)

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
        )
        target = _store_dir(root, sample, key)
        if target.exists():
            _load_verified_manifest(target, expected_key=key)
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
        return MemoryCondenser(
            data_dir=target,
            chunker_min_tokens=config.chunker.min_tokens,
            chunker_max_tokens=config.chunker.max_tokens,
            auto_extract=False,
            embedder=shared_embedder,
            persist_index_on_close=False,
        )

    return ingest
