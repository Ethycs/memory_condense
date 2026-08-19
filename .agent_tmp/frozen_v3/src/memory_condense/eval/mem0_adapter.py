"""Auditable Mem0 OSS 2.0.18 boundary for LongMemEval comparisons.

The adapter deliberately uses only the public ``Memory.add``, ``search``, and
``delete_all`` methods.  ``mem0ai`` remains an optional benchmark dependency:
importing this module neither imports nor installs it.

Mem0 2.0.18 does not expose which source text grounded an inferred memory.
Consequently this module records only *request-window attribution*: the current
one-or-two-turn request and the prior ten messages that the pinned V3 pipeline
can place in its extraction request.  This is useful for auditing reachability,
but it is explicitly not evidence provenance.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import shutil
import sys
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias

from memory_condense._tokenizer import count_tokens, tokenizer_proxy_identity
from memory_condense.loader import BenchmarkSample


MEM0AI_PIN = "2.0.18"
MEM0_API_VERSION = "v1.1"
MEM0_CONTEXT_SEPARATOR = "\n\n"
MEM0_REQUEST_WINDOW_MESSAGES = 10
MEM0_OFFICIAL_TOP_K = 200
MEM0_OFFICIAL_THRESHOLD = 0.1
MEM0_ATTRIBUTION_KIND = "request_window_non_evidence"
MEM0_DATE_EXPOSURE_KIND = "diagnostics_only_not_model_input"
MEM0_PROVIDER_USAGE_STATUS = "unavailable_from_mem0_oss_public_api"
MEM0_CERTIFIED_RENDERING = "official-memory-text-created-at"
MEM0_ENRICHED_RENDERING = "enriched-attribution-noncertifying"
MEM0_BM25_MODEL = "Qdrant/bm25"
MEM0_SPACY_MODEL = "en_core_web_sm"

_OWNERSHIP_MARKER = ".memory-condense-owned-state"
_SESSION_DATE_RE = re.compile(
    r"^\[(?P<session>.+?) took place at (?P<date>.+?)\]$",
    re.IGNORECASE,
)
_WEEKDAY_RE = re.compile(r"\s+\([^)]*\)\s+")

TokenCounter: TypeAlias = Callable[[str], int]
Clock: TypeAlias = Callable[[], float]
BackendFactory: TypeAlias = Callable[[], Any]
PromptRenderer: TypeAlias = Callable[[str, str], str]
ScopedMemoryKey: TypeAlias = tuple[str, str]
StackPreflight: TypeAlias = Callable[[], "Mem0StackIdentity"]


class Mem0AdapterError(RuntimeError):
    """Base error raised by the optional Mem0 benchmark boundary."""


class Mem0DependencyError(Mem0AdapterError):
    """The exact optional Mem0 distribution is unavailable."""


class Mem0ConfigurationError(Mem0AdapterError, ValueError):
    """A real Mem0 factory was not given a fully isolated frozen config."""


class Mem0ProtocolError(Mem0AdapterError):
    """Mem0 or the benchmark input violated the frozen comparison protocol."""


class Mem0AttributionError(Mem0ProtocolError):
    """A search row cannot be tied to an audited scoped add request window."""


class Mem0PromptBudgetError(Mem0AdapterError, ValueError):
    """The declared final responder prompt cannot fit its token cap."""


class Mem0PoisonedError(Mem0ProtocolError):
    """A possibly mutating operation failed and only cleanup remains safe."""


class _Closable(Protocol):
    def close(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Text-free identity of one official consecutive 1--2 turn add request."""

    sample_id: str
    source: str
    session: str
    session_index: int
    original_session_index: int
    batch_index: int
    date: str
    turn_start: int
    turn_count: int
    roles: tuple[str, ...]

    @property
    def pair(self) -> int:
        """Compatibility spelling for the historical two-turn batch index."""

        return self.batch_index

    @property
    def metadata(self) -> dict[str, str | int]:
        """Text-free audit metadata that is never supplied to Mem0."""

        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "session": self.session,
            "session_index": self.session_index,
            "original_session_index": self.original_session_index,
            "pair": self.batch_index,
            "date": self.date,
            "turn_start": self.turn_start,
            "turn_count": self.turn_count,
            "roles": ",".join(self.roles),
            "date_exposure": MEM0_DATE_EXPOSURE_KIND,
        }


# Kept as an import-compatible name while removing the false implication that
# every official slice contains a user/assistant pair.
SourcePair = SourceRef


@dataclass(frozen=True, slots=True)
class _PreparedBatch:
    ref: SourceRef
    messages: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _PreparedCorpus:
    sample_id: str
    batches: tuple[_PreparedBatch, ...]
    raw_pair_count: int
    skipped_empty_pair_count: int
    official_longmemeval_protocol: bool


@dataclass(frozen=True, slots=True)
class Mem0StackIdentity:
    """Stable, secret-free identity of the certified OSS runtime stack."""

    dependency_versions: Mapping[str, str]
    bm25_model: str
    spacy_model: str
    bm25_operational: bool
    entity_extraction_operational: bool

    @property
    def certified(self) -> bool:
        return self.bm25_operational and self.entity_extraction_operational

    def as_dict(self) -> dict[str, Any]:
        return {
            "dependency_versions": dict(self.dependency_versions),
            "bm25_model": self.bm25_model,
            "spacy_model": self.spacy_model,
            "bm25_operational": self.bm25_operational,
            "entity_extraction_operational": self.entity_extraction_operational,
        }


MemoryLedger: TypeAlias = Mapping[ScopedMemoryKey, tuple[SourceRef, ...]]


@dataclass(frozen=True, slots=True)
class Mem0AdapterStats:
    """Cumulative local proxy accounting; provider usage is unavailable."""

    add_calls: int = 0
    add_attempted_calls: int = 0
    add_completed_calls: int = 0
    add_failed_calls: int = 0
    search_calls: int = 0
    add_latency_s: float = 0.0
    search_latency_s: float = 0.0
    add_raw_message_tokens: int = 0
    search_query_tokens: int = 0
    search_raw_memory_tokens: int = 0
    search_context_tokens: int = 0
    search_prompt_token_proxy: int = 0
    # Compatibility spelling. This is the same caller-supplied local proxy,
    # not an exact provider-token count.
    search_prompt_tokens: int = 0
    add_returned_memories: int = 0
    unique_ledger_memories: int = 0
    search_returned_memories: int = 0
    search_packed_memories: int = 0
    released_scopes: int = 0
    provider_prompt_tokens: int | None = None
    provider_completion_tokens: int | None = None
    provider_usage_status: str = MEM0_PROVIDER_USAGE_STATUS
    token_counter_identity: str = ""
    token_counter_identity_verified: bool = False


@dataclass(frozen=True, slots=True)
class Mem0IngestResult:
    """Identity, request-window attribution, and accounting for one sample."""

    sample_id: str
    user_scope: str
    batches_added: tuple[SourceRef, ...]
    returned_memory_ids: tuple[str, ...]
    ledger: MemoryLedger
    attribution_kind: str
    supports_exact_source_provenance: bool
    date_exposure_kind: str
    raw_pair_count: int
    skipped_empty_pair_count: int
    official_longmemeval_protocol: bool
    comparison_certified: bool
    runtime_identity: Mapping[str, Any]
    stats: Mem0AdapterStats

    @property
    def pairs_added(self) -> tuple[SourceRef, ...]:
        """Compatibility alias; batches may be singleton or assistant-first."""

        return self.batches_added


@dataclass(frozen=True, slots=True)
class Mem0Candidate:
    """One rank-preserving, scoped, audited Mem0 search row."""

    rank: int
    memory_id: str
    text: str
    score: float | None
    created_at: str | None
    metadata: Mapping[str, Any]
    request_window_attribution: tuple[SourceRef, ...]
    attribution_kind: str
    raw: Any


@dataclass(frozen=True, slots=True)
class Mem0PackDiagnostic:
    """Why a candidate was included in or excluded from final context."""

    candidate: Mem0Candidate
    rendered: str
    audit_rendered: str
    rendered_tokens: int
    selected: bool
    reason: str
    context_tokens_after: int
    prompt_token_proxy_after: int
    # Compatibility spelling for prompt_token_proxy_after.
    prompt_tokens_after: int


@dataclass(frozen=True, slots=True)
class Mem0SearchResult:
    """Budgeted context plus the complete raw-pool and prompt audit trail.

    ``prompt_token_proxy`` is counted by the caller-supplied counter. It is a
    hard local packing bound, not a claim about provider-token usage.
    """

    user_scope: str
    query: str
    context: str
    context_tokens: int
    prompt: str
    prompt_token_proxy: int
    max_prompt_token_proxy: int
    prompt_token_proxy_overhead: int
    empty_context_prompt_token_proxy: int
    residual_prompt_token_proxy: int
    prompt_token_proxy_budget_compliant: bool
    token_counter_identity: str
    token_counter_identity_verified: bool
    # Compatibility fields below mirror their ``*_proxy`` counterparts.
    # In particular, prompt_budget_certified certifies only deterministic
    # local packing under the declared counter; it is not provider usage.
    prompt_tokens: int
    max_prompt_tokens: int
    prompt_token_overhead: int
    empty_context_prompt_tokens: int
    residual_prompt_tokens: int
    prompt_budget_certified: bool
    packed: tuple[Mem0Candidate, ...]
    raw_pool: tuple[Mem0Candidate, ...]
    diagnostics: tuple[Mem0PackDiagnostic, ...]
    raw_response: Any
    attribution_kind: str
    supports_exact_source_provenance: bool
    rendering_mode: str
    certified_rendering: bool
    official_longmemeval_protocol: bool
    official_search_protocol: bool
    comparison_certified: bool
    runtime_identity: Mapping[str, Any]
    stats: Mem0AdapterStats


_SECRET_CONFIG_KEYS = {
    "api_key",
    "password",
    "secret",
    "secret_key",
    "token",
    "access_token",
    "refresh_token",
    "client_secret",
    "credentials",
}


def _installed_version(distribution: str) -> str:
    try:
        value = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise Mem0DependencyError(
            f"Certified Mem0 OSS comparison requires {distribution!r} to be "
            "installed in the frozen benchmark environment."
        ) from exc
    if not value:
        raise Mem0DependencyError(
            f"Could not identify installed distribution {distribution!r}."
        )
    return value


def _default_stack_preflight() -> Mem0StackIdentity:
    """Prove that the tagged hybrid/entity stack is locally available.

    ``local_files_only=True`` is mandatory for the FastEmbed model.  This makes
    a missing artifact a pre-run error instead of an implicit network download
    or Mem0's silent semantic-only fallback.
    """

    versions = {
        "mem0ai": _installed_version("mem0ai"),
        "qdrant-client": _installed_version("qdrant-client"),
        "fastembed": _installed_version("fastembed"),
        "spacy": _installed_version("spacy"),
        "en-core-web-sm": _installed_version("en-core-web-sm"),
    }
    if versions["mem0ai"] != MEM0AI_PIN:
        raise Mem0DependencyError(
            f"Expected mem0ai=={MEM0AI_PIN}, found mem0ai=={versions['mem0ai']}."
        )

    try:
        spacy_model = importlib.import_module(MEM0_SPACY_MODEL)
        nlp = spacy_model.load()
    except Exception as exc:
        raise Mem0DependencyError(
            f"Certified Mem0 requires the locally installed {MEM0_SPACY_MODEL} "
            "pipeline; runtime downloads are not permitted."
        ) from exc
    pipe_names = set(getattr(nlp, "pipe_names", ()))
    if not {"ner", "lemmatizer"}.issubset(pipe_names):
        raise Mem0DependencyError(
            f"{MEM0_SPACY_MODEL} must expose both NER and lemmatization."
        )
    probe_doc = nlp("Alice visited Seattle and remembered the visit.")
    if not any(getattr(token, "lemma_", "") for token in probe_doc) or not list(
        getattr(probe_doc, "ents", ())
    ):
        raise Mem0DependencyError(
            f"{MEM0_SPACY_MODEL} failed the entity/lemma operational probe."
        )

    try:
        fastembed_module = importlib.import_module("fastembed")
        encoder_type = getattr(fastembed_module, "SparseTextEmbedding")
        parameters = inspect.signature(encoder_type.__init__).parameters
        if "local_files_only" not in parameters:
            raise Mem0DependencyError(
                "Installed fastembed cannot prove an offline BM25 model load."
            )
        encoder = encoder_type(
            model_name=MEM0_BM25_MODEL,
            local_files_only=True,
        )
        rows = list(encoder.embed(["memory retrieval operational probe"]))
        if not rows or not len(rows[0].indices) or not len(rows[0].values):
            raise Mem0DependencyError("FastEmbed BM25 operational probe was empty.")
    except Mem0DependencyError:
        raise
    except Exception as exc:
        raise Mem0DependencyError(
            f"Certified Mem0 requires {MEM0_BM25_MODEL!r} in the local "
            "FastEmbed cache; runtime downloads and semantic-only fallback are "
            "not permitted."
        ) from exc

    return Mem0StackIdentity(
        dependency_versions=MappingProxyType(dict(sorted(versions.items()))),
        bm25_model=MEM0_BM25_MODEL,
        spacy_model=MEM0_SPACY_MODEL,
        bm25_operational=True,
        entity_extraction_operational=True,
    )


def _redacted_stable_config(
    value: Any,
    *,
    state_root: Path,
    key: str = "",
) -> Any:
    """Canonicalize config without secrets or per-run owned-state paths."""

    lowered = key.lower()
    if lowered in _SECRET_CONFIG_KEYS or lowered.endswith(("_password", "_secret", "_token")):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(item_key): _redacted_stable_config(
                item_value,
                state_root=state_root,
                key=str(item_key),
            )
            for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _redacted_stable_config(item, state_root=state_root, key=key)
            for item in value
        ]
    if isinstance(value, (str, os.PathLike)) and lowered in {
        "path",
        "history_db_path",
    }:
        resolved = _resolved(value)
        if _is_within(resolved, state_root):
            relative = resolved.relative_to(state_root).as_posix()
            return f"<owned_state>/{relative}"
    return value


def _sha256_json(value: Any) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _required_value(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            dotted = ".".join(path)
            raise Mem0ConfigurationError(
                f"Mem0 OSS config must explicitly set {dotted!r}."
            )
        value = value[part]
    return value


def _required_text(config: Mapping[str, Any], path: tuple[str, ...]) -> str:
    value = _required_value(config, path)
    if not isinstance(value, str) or not value.strip():
        dotted = ".".join(path)
        raise Mem0ConfigurationError(
            f"Mem0 OSS config must explicitly set a non-empty {dotted!r}."
        )
    return value.strip()


def _resolved(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _telemetry_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return value is True


def _assert_no_live_mem0_telemetry() -> None:
    for module_name in ("mem0.memory.telemetry", "mem0.memory.main"):
        module = sys.modules.get(module_name)
        if module is not None and _telemetry_enabled(
            getattr(module, "MEM0_TELEMETRY", False)
        ):
            raise Mem0DependencyError(
                "Mem0 was already imported with telemetry enabled; exact "
                "benchmark isolation requires a fresh process with "
                "MEM0_TELEMETRY=false before importing mem0."
            )
    telemetry_module = sys.modules.get("mem0.memory.telemetry")
    if telemetry_module is not None:
        clients = [getattr(telemetry_module, "client_telemetry", None)]
        oss_client = getattr(telemetry_module, "_oss_telemetry_instance", None)
        if oss_client is not None:
            clients.append(oss_client)
        if any(getattr(client, "posthog", None) is not None for client in clients):
            raise Mem0DependencyError(
                "Mem0 already owns a live telemetry client; exact benchmark "
                "isolation requires a fresh process."
            )


def _assert_mem0_state_binding(state_root: Path) -> None:
    """Reject cached Mem0 modules bound to a different global state path."""

    for module_name in ("mem0.configs.base", "mem0.memory.setup"):
        module = sys.modules.get(module_name)
        configured = getattr(module, "mem0_dir", None) if module is not None else None
        if configured is not None and _resolved(str(configured)) != state_root:
            raise Mem0DependencyError(
                "Mem0 was already imported with a different MEM0_DIR; use a "
                "fresh process for each owned-state comparison run."
            )


def _remove_owned_state(root: Path, token: str) -> None:
    if not root.exists():
        return
    if root.is_symlink() or (
        hasattr(root, "is_junction") and root.is_junction()  # type: ignore[attr-defined]
    ):
        raise Mem0ConfigurationError(
            f"Refusing to remove replaced owned-state path: {root}"
        )
    marker = root / _OWNERSHIP_MARKER
    try:
        marker_value = marker.read_text(encoding="utf-8")
    except OSError as exc:
        raise Mem0ConfigurationError(
            f"Refusing to remove unmarked owned-state directory: {root}"
        ) from exc
    if marker_value != token:
        raise Mem0ConfigurationError(
            f"Refusing to remove owned-state directory with a foreign marker: {root}"
        )
    shutil.rmtree(root)


def _raise_cleanup_errors(errors: list[BaseException], message: str) -> None:
    if not errors:
        return
    if len(errors) == 1:
        raise errors[0]
    raise BaseExceptionGroup(message, errors)


class _OwnedMem0Backend:
    """Own a Mem0 instance, its collections/clients, and its SQLite state."""

    def __init__(
        self,
        *,
        backend: Any,
        state_root: Path,
        ownership_token: str,
        collection_name: str,
        stable_config_fingerprint: str,
        effective_config_fingerprint: str,
        runtime_identity: Mapping[str, Any],
    ) -> None:
        self.backend = backend
        self.state_root = state_root
        self.collection_name = collection_name
        # Compatibility name now denotes the stable, redacted identity.  The
        # effective hash intentionally varies with the unique collection name.
        self.config_fingerprint = stable_config_fingerprint
        self.stable_config_fingerprint = stable_config_fingerprint
        self.effective_config_fingerprint = effective_config_fingerprint
        self.runtime_identity = MappingProxyType(copy.deepcopy(dict(runtime_identity)))
        self._ownership_token = ownership_token
        self._closed = False

    def add(self, *args: Any, **kwargs: Any) -> Any:
        return self.backend.add(*args, **kwargs)

    def search(self, *args: Any, **kwargs: Any) -> Any:
        return self.backend.search(*args, **kwargs)

    def delete_all(self, *args: Any, **kwargs: Any) -> Any:
        return self.backend.delete_all(*args, **kwargs)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []

        stores: list[Any] = []
        entity_store = getattr(self.backend, "_entity_store", None)
        vector_store = getattr(self.backend, "vector_store", None)
        for store in (entity_store, vector_store):
            if store is not None and all(store is not item for item in stores):
                stores.append(store)

        # Collections are unique to this factory invocation.  Drop them while
        # their clients are live, then close SQLite and every discoverable
        # vector client before removing the owned directory.
        for store in stores:
            delete_col = getattr(store, "delete_col", None)
            if callable(delete_col):
                try:
                    delete_col()
                except BaseException as exc:
                    errors.append(exc)

        close_backend = getattr(self.backend, "close", None)
        if callable(close_backend):
            try:
                close_backend()
            except BaseException as exc:
                errors.append(exc)

        clients: list[Any] = []
        for store in stores:
            client = getattr(store, "client", None)
            if client is not None and all(client is not item for item in clients):
                clients.append(client)
        for client in clients:
            close_client = getattr(client, "close", None)
            if callable(close_client):
                try:
                    close_client()
                except BaseException as exc:
                    errors.append(exc)

        try:
            _remove_owned_state(self.state_root, self._ownership_token)
        except BaseException as exc:
            errors.append(exc)
        _raise_cleanup_errors(errors, "Mem0 owned-state cleanup failed")


class Mem0OSSBackendFactory:
    """Lazily construct exactly pinned Mem0 OSS in per-run owned state.

    The certified factory accepts only on-disk embedded Qdrant beneath a fresh
    ``owned_state_dir``.  Remote stores are deliberately excluded because
    exclusive ownership and successful erasure cannot be certified.  A unique
    suffix is still added to the collection name as defense in depth.

    Underscored dependency injections exist only for fake-backed tests; normal
    use validates both distribution and imported-module versions.
    """

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        llm_model_id: str,
        embedder_model_id: str,
        owned_state_dir: str | os.PathLike[str],
        _version_reader: Callable[[str], str] | None = None,
        _module_importer: Callable[[str], Any] | None = None,
        _stack_preflight: StackPreflight | None = None,
    ) -> None:
        if not isinstance(config, Mapping):
            raise Mem0ConfigurationError("Mem0 OSS config must be a mapping.")

        _required_text(config, ("llm", "provider"))
        configured_llm = _required_text(config, ("llm", "config", "model"))
        _required_text(config, ("embedder", "provider"))
        configured_embedder = _required_text(
            config, ("embedder", "config", "model")
        )
        vector_provider = _required_text(config, ("vector_store", "provider"))
        if vector_provider.lower() != "qdrant":
            raise Mem0ConfigurationError(
                "The isolated Mem0 comparison currently requires the qdrant "
                "vector_store provider."
            )
        vector_config = _required_value(config, ("vector_store", "config"))
        if not isinstance(vector_config, Mapping):
            raise Mem0ConfigurationError(
                "Mem0 OSS config must explicitly set 'vector_store.config'."
            )
        _required_text(config, ("vector_store", "config", "collection_name"))
        dims = _required_value(
            config, ("vector_store", "config", "embedding_model_dims")
        )
        if isinstance(dims, bool) or not isinstance(dims, int) or dims <= 0:
            raise Mem0ConfigurationError(
                "vector_store.config.embedding_model_dims must be a positive int."
            )
        if vector_config.get("client") is not None:
            raise Mem0ConfigurationError(
                "Injected Qdrant clients are not permitted in the owned-state factory."
            )

        version = _required_text(config, ("version",))
        if version != MEM0_API_VERSION:
            raise Mem0ConfigurationError(
                f"Mem0 API version must be exactly {MEM0_API_VERSION!r}."
            )
        if "custom_instructions" not in config or not (
            config["custom_instructions"] is None
            or isinstance(config["custom_instructions"], str)
        ):
            raise Mem0ConfigurationError(
                "Mem0 config must explicitly freeze custom_instructions as str or null."
            )
        if "reranker" not in config or config["reranker"] is not None:
            raise Mem0ConfigurationError(
                "Mem0 config must explicitly freeze reranker as null."
            )

        if not isinstance(llm_model_id, str) or not llm_model_id.strip():
            raise Mem0ConfigurationError("A non-empty LLM model ID is required.")
        if not isinstance(embedder_model_id, str) or not embedder_model_id.strip():
            raise Mem0ConfigurationError("A non-empty embedder model ID is required.")
        if configured_llm != llm_model_id.strip():
            raise Mem0ConfigurationError(
                "llm_model_id does not match config['llm']['config']['model']."
            )
        if configured_embedder != embedder_model_id.strip():
            raise Mem0ConfigurationError(
                "embedder_model_id does not match "
                "config['embedder']['config']['model']."
            )

        state_root = _resolved(owned_state_dir)
        parent = state_root.parent
        if state_root == parent or not parent.exists() or not parent.is_dir():
            raise Mem0ConfigurationError(
                "owned_state_dir must be a fresh child of an existing directory."
            )
        if state_root.exists():
            raise Mem0ConfigurationError("owned_state_dir must not already exist.")

        history_path = _resolved(_required_text(config, ("history_db_path",)))
        if history_path == state_root or not _is_within(history_path, state_root):
            raise Mem0ConfigurationError(
                "history_db_path must name a file inside owned_state_dir."
            )

        has_path = isinstance(vector_config.get("path"), str) and bool(
            vector_config.get("path", "").strip()
        )
        if not has_path or any(
            vector_config.get(name) is not None
            for name in ("url", "host", "port", "api_key")
        ):
            raise Mem0ConfigurationError(
                "Certified Mem0 requires only a local Qdrant path; remote URL, "
                "host, port, API key, and injected-client targets are forbidden."
            )
        if vector_config.get("on_disk") is not True:
            raise Mem0ConfigurationError(
                "Certified Mem0 requires vector_store.config.on_disk=true."
            )
        qdrant_path = _resolved(str(vector_config["path"]))
        if qdrant_path == state_root or not _is_within(qdrant_path, state_root):
            raise Mem0ConfigurationError(
                "Local Qdrant path must be a directory inside owned_state_dir."
            )

        self._config = copy.deepcopy(dict(config))
        self._state_root = state_root
        self._version_reader = _version_reader
        self._module_importer = _module_importer
        self._stack_preflight = _stack_preflight or _default_stack_preflight
        self._ownership_token = uuid.uuid4().hex
        self._called = False

    @property
    def config(self) -> Mapping[str, Any]:
        return MappingProxyType(copy.deepcopy(self._config))

    @property
    def owned_state_dir(self) -> Path:
        return self._state_root

    def __call__(self) -> _OwnedMem0Backend:
        if self._called:
            raise Mem0ConfigurationError(
                "A Mem0 owned-state factory is single-use; create a fresh run factory."
            )
        self._called = True

        previous = os.environ.get("MEM0_TELEMETRY")
        had_previous = "MEM0_TELEMETRY" in os.environ
        previous_mem0_dir = os.environ.get("MEM0_DIR")
        had_previous_mem0_dir = "MEM0_DIR" in os.environ
        os.environ["MEM0_TELEMETRY"] = "false"
        os.environ["MEM0_DIR"] = str(self._state_root)
        created_state = False
        try:
            _assert_no_live_mem0_telemetry()
            _assert_mem0_state_binding(self._state_root)
            version_reader = self._version_reader or importlib.metadata.version
            try:
                installed = version_reader("mem0ai")
            except importlib.metadata.PackageNotFoundError as exc:
                raise Mem0DependencyError(
                    "Mem0 OSS comparison requires the optional dependency "
                    f"'mem0ai=={MEM0AI_PIN}'. Install that exact version in "
                    "the benchmark environment; importing memory_condense "
                    "does not install or load it automatically."
                ) from exc
            if installed != MEM0AI_PIN:
                raise Mem0DependencyError(
                    "Mem0 OSS comparison is version locked: expected "
                    f"mem0ai=={MEM0AI_PIN}, found mem0ai=={installed}."
                )

            stack_identity = self._stack_preflight()
            if not isinstance(stack_identity, Mem0StackIdentity):
                raise Mem0DependencyError(
                    "Mem0 stack preflight must return Mem0StackIdentity."
                )
            if not stack_identity.certified:
                raise Mem0DependencyError(
                    "Mem0 hybrid/entity stack did not pass its operational preflight."
                )
            preflight_mem0 = stack_identity.dependency_versions.get("mem0ai")
            if preflight_mem0 != MEM0AI_PIN:
                raise Mem0DependencyError(
                    "Mem0 stack identity does not match the pinned distribution."
                )

            if self._state_root.exists():
                raise Mem0ConfigurationError(
                    "owned_state_dir appeared before factory initialization; "
                    "refusing to adopt it."
                )
            self._state_root.mkdir(parents=False)
            (self._state_root / _OWNERSHIP_MARKER).write_text(
                self._ownership_token, encoding="utf-8"
            )
            created_state = True

            module_importer = self._module_importer or importlib.import_module
            try:
                mem0_module = module_importer("mem0")
            except (ImportError, ModuleNotFoundError) as exc:
                raise Mem0DependencyError(
                    f"The installed mem0ai=={MEM0AI_PIN} distribution could "
                    "not provide the 'mem0' module."
                ) from exc

            module_version = getattr(mem0_module, "__version__", None)
            if module_version is not None and str(module_version) != installed:
                raise Mem0DependencyError(
                    "Imported mem0 module version does not match the installed "
                    f"distribution: module={module_version!r}, distribution={installed!r}."
                )
            if self._module_importer is None and module_version is None:
                raise Mem0DependencyError(
                    "The imported mem0 module did not expose __version__; "
                    "the pinned distribution identity cannot be certified."
                )
            _assert_no_live_mem0_telemetry()

            memory_type = getattr(mem0_module, "Memory", None)
            from_config = getattr(memory_type, "from_config", None)
            if not callable(from_config):
                raise Mem0DependencyError(
                    f"mem0ai=={MEM0AI_PIN} does not expose Memory.from_config."
                )

            effective_config = copy.deepcopy(self._config)
            vector_config = effective_config["vector_store"]["config"]
            base_collection = str(vector_config["collection_name"]).strip()
            collection_name = (
                f"{base_collection}-{self._ownership_token[:12]}"
            )
            vector_config["collection_name"] = collection_name
            stable_payload = {
                "protocol": "mem0-oss-2.0.18-certified-local-v1",
                "config": _redacted_stable_config(
                    self._config,
                    state_root=self._state_root,
                ),
                "stack": stack_identity.as_dict(),
            }
            stable_fingerprint = _sha256_json(stable_payload)
            effective_fingerprint = _sha256_json(effective_config)
            runtime_identity = {
                **stable_payload,
                "stable_config_sha256": stable_fingerprint,
                "effective_config_sha256": effective_fingerprint,
                "local_owned_state": True,
                "on_disk": True,
                "certified": True,
            }
            backend = from_config(config_dict=effective_config)
            wrapped = _OwnedMem0Backend(
                backend=backend,
                state_root=self._state_root,
                ownership_token=self._ownership_token,
                collection_name=collection_name,
                stable_config_fingerprint=stable_fingerprint,
                effective_config_fingerprint=effective_fingerprint,
                runtime_identity=runtime_identity,
            )
            vector_store = getattr(backend, "vector_store", None)
            if vector_store is None or getattr(
                vector_store, "_has_bm25_slot", False
            ) is not True:
                error = Mem0DependencyError(
                    "Fresh Qdrant collection did not expose the required BM25 slot."
                )
                try:
                    wrapped.close()
                except BaseException as cleanup_exc:
                    error.add_note(
                        f"Failed stack-preflight cleanup also failed: {cleanup_exc!r}"
                    )
                raise error
            return wrapped
        except BaseException as exc:
            if created_state:
                try:
                    _remove_owned_state(self._state_root, self._ownership_token)
                except BaseException as cleanup_exc:
                    exc.add_note(
                        "Mem0 initialization cleanup also failed: "
                        f"{cleanup_exc!r}"
                    )
            raise
        finally:
            if had_previous:
                assert previous is not None
                os.environ["MEM0_TELEMETRY"] = previous
            else:
                os.environ.pop("MEM0_TELEMETRY", None)
            if had_previous_mem0_dir:
                assert previous_mem0_dir is not None
                os.environ["MEM0_DIR"] = previous_mem0_dir
            else:
                os.environ.pop("MEM0_DIR", None)


def _response_rows(response: Any, *, operation: str) -> list[Mapping[str, Any]]:
    """Strictly normalize documented list/``{"results": [...]}`` variants."""

    rows: Any
    if isinstance(response, Mapping) and "results" in response:
        rows = response["results"]
    elif isinstance(response, Sequence) and not isinstance(
        response, (str, bytes, bytearray)
    ):
        rows = response
    elif isinstance(response, Mapping) and (
        "id" in response or "memory_id" in response
    ):
        rows = [response]
    else:
        raise Mem0ProtocolError(
            f"Mem0 {operation} returned an unsupported response shape."
        )
    if not isinstance(rows, Sequence) or isinstance(
        rows, (str, bytes, bytearray)
    ):
        raise Mem0ProtocolError(
            f"Mem0 {operation} response 'results' must be a sequence."
        )
    normalized: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise Mem0ProtocolError(
                f"Mem0 {operation} result {index} is not a mapping."
            )
        normalized.append(row)
    return normalized


def _memory_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("id", row.get("memory_id"))
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _memory_text(row: Mapping[str, Any]) -> str:
    value = row.get("memory", row.get("text", ""))
    return value.strip() if isinstance(value, str) else ""


def _memory_created_at(row: Mapping[str, Any]) -> str | None:
    value = row.get("created_at")
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _official_date_label(value: str) -> str:
    """Render Mem0's returned timestamp as the official benchmark date label."""

    candidate = value.strip()
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.strftime("%A, %B %d, %Y")
    except ValueError:
        # The official runner tolerates provider timestamp variants by using
        # their date prefix.  Refuse values that cannot supply even that.
        prefix = candidate[:10]
        try:
            parsed = datetime.strptime(prefix, "%Y-%m-%d")
        except ValueError as exc:
            raise Mem0ProtocolError(
                f"Mem0 search returned an invalid created_at value: {value!r}."
            ) from exc
        return parsed.strftime("%A, %B %d, %Y")


def _memory_score(row: Mapping[str, Any]) -> float | None:
    value = row.get("score")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _safe_label_value(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _parse_session_date(value: str) -> datetime:
    cleaned = _WEEKDAY_RE.sub(" ", value.strip())
    for format_string in (
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(cleaned, format_string)
        except ValueError:
            continue
    raise Mem0ProtocolError(
        f"Unsupported haystack session date {value!r}; chronology cannot be certified."
    )


@dataclass(frozen=True, slots=True)
class _SessionBlock:
    source: str
    date: str
    turns: tuple[tuple[str, str], ...]
    original_index: int


def _session_blocks(sample: BenchmarkSample) -> tuple[_SessionBlock, ...]:
    source_ids = sample.turn_source_ids
    if source_ids and len(source_ids) != len(sample.turns):
        raise Mem0ProtocolError(
            "turn_source_ids must be empty or parallel to every benchmark turn."
        )

    blocks: list[_SessionBlock] = []
    current_source = ""
    current_date = ""
    current_turns: list[tuple[str, str]] = []

    def flush() -> None:
        nonlocal current_turns
        if not current_turns:
            return
        blocks.append(
            _SessionBlock(
                source=current_source or f"{sample.sample_id}:session_1",
                date=current_date,
                turns=tuple(current_turns),
                original_index=len(blocks) + 1,
            )
        )
        current_turns = []

    for index, (raw_role, text) in enumerate(sample.turns):
        declared = ""
        if source_ids and source_ids[index] is not None:
            declared = str(source_ids[index]).strip()
        match = _SESSION_DATE_RE.fullmatch(text.strip())
        if match:
            marker_source = match.group("session").strip()
            if declared and declared != marker_source:
                raise Mem0ProtocolError(
                    "Session marker and turn_source_ids disagree at turn "
                    f"{index}."
                )
            if current_turns:
                flush()
            current_source = marker_source
            current_date = match.group("date").strip()
            continue

        source = declared or current_source or f"{sample.sample_id}:session_1"
        if current_source and source != current_source:
            flush()
            current_date = ""
        current_source = source
        role = str(raw_role).strip().lower()
        current_turns.append((role, text))

    flush()

    def sort_key(block: _SessionBlock) -> tuple[int, datetime, int]:
        if not block.date:
            return (1, datetime.max, block.original_index)
        return (0, _parse_session_date(block.date), block.original_index)

    return tuple(sorted(blocks, key=sort_key))


def _prepared_batches(sample: BenchmarkSample) -> tuple[_PreparedBatch, ...]:
    batches: list[_PreparedBatch] = []
    for chronological_index, block in enumerate(_session_blocks(sample), start=1):
        for turn_start in range(0, len(block.turns), 2):
            messages = block.turns[turn_start : turn_start + 2]
            ref = SourceRef(
                sample_id=sample.sample_id,
                source=block.source,
                session=block.source,
                session_index=chronological_index,
                original_session_index=block.original_index,
                batch_index=(turn_start // 2) + 1,
                date=block.date,
                turn_start=turn_start,
                turn_count=len(messages),
                roles=tuple(role for role, _ in messages),
            )
            batches.append(_PreparedBatch(ref=ref, messages=messages))
    return tuple(batches)


def _prepared_sample(sample: BenchmarkSample) -> _PreparedCorpus:
    batches = _prepared_batches(sample)
    return _PreparedCorpus(
        sample_id=sample.sample_id,
        batches=batches,
        raw_pair_count=len(batches),
        skipped_empty_pair_count=0,
        official_longmemeval_protocol=False,
    )


def _prepared_longmemeval_record(record: Mapping[str, Any]) -> _PreparedCorpus:
    """Prepare the official lossless LongMemEval add sequence.

    The shared :class:`BenchmarkSample` loader intentionally drops empty text,
    so it cannot reproduce the official order for the handful of sessions that
    contain empty turns.  Certified Mem0 runs therefore consume the raw record:
    pair original consecutive turns first, then skip a whole pair if either
    message is empty.
    """

    sample_id_value = record.get("question_id")
    sample_id = str(sample_id_value).strip() if sample_id_value is not None else ""
    if not sample_id:
        raise Mem0ProtocolError(
            "Certified LongMemEval input requires a non-empty question_id."
        )
    sessions = record.get("haystack_sessions")
    session_ids = record.get("haystack_session_ids")
    dates = record.get("haystack_dates")
    if not all(isinstance(value, list) for value in (sessions, session_ids, dates)):
        raise Mem0ProtocolError(
            "Certified LongMemEval input requires list-valued sessions, IDs, and dates."
        )
    assert isinstance(sessions, list)
    assert isinstance(session_ids, list)
    assert isinstance(dates, list)
    if not (len(sessions) == len(session_ids) == len(dates)):
        raise Mem0ProtocolError(
            "LongMemEval sessions, session IDs, and dates must be parallel."
        )

    ordered: list[tuple[datetime, int, str, str, list[Any]]] = []
    for original_index, (source_value, date_value, session) in enumerate(
        zip(session_ids, dates, sessions),
        start=1,
    ):
        if not isinstance(source_value, str) or not isinstance(date_value, str):
            raise Mem0ProtocolError(
                "Every certified LongMemEval session ID and date must be a string."
            )
        source = source_value.strip()
        date = date_value.strip()
        if not source or not date or not isinstance(session, list):
            raise Mem0ProtocolError(
                "Every certified LongMemEval session needs an ID, date, and turn list."
            )
        parsed_date = _parse_session_date(date)
        ordered.append((parsed_date, original_index, source, date, session))
    ordered.sort(key=lambda item: (item[0], item[1]))

    batches: list[_PreparedBatch] = []
    raw_pair_count = 0
    skipped_empty_pair_count = 0
    for chronological_index, (
        _parsed_date,
        original_index,
        source,
        date,
        session,
    ) in enumerate(ordered, start=1):
        for turn_start in range(0, len(session), 2):
            raw_pair_count += 1
            raw_pair = session[turn_start : turn_start + 2]
            messages: list[tuple[str, str]] = []
            for turn in raw_pair:
                if not isinstance(turn, Mapping):
                    raise Mem0ProtocolError(
                        "Every certified LongMemEval turn must be a mapping."
                    )
                role_value = turn.get("role")
                content_value = turn.get("content")
                if not isinstance(role_value, str) or not isinstance(
                    content_value, str
                ):
                    raise Mem0ProtocolError(
                        "Every certified LongMemEval turn needs string role/content."
                    )
                role = role_value.strip().lower()
                if role not in {"user", "assistant"}:
                    raise Mem0ProtocolError(
                        f"Unsupported certified LongMemEval role: {role_value!r}."
                    )
                messages.append((role, content_value))
            # Match the official runner's truthiness check exactly: an empty
            # string suppresses the entire original pair; whitespace is still
            # a non-empty message and must not shift later pair boundaries.
            if any(not content for _role, content in messages):
                skipped_empty_pair_count += 1
                continue
            ref = SourceRef(
                sample_id=sample_id,
                source=source,
                session=source,
                session_index=chronological_index,
                original_session_index=original_index,
                batch_index=(turn_start // 2) + 1,
                date=date,
                turn_start=turn_start,
                turn_count=len(messages),
                roles=tuple(role for role, _content in messages),
            )
            batches.append(_PreparedBatch(ref=ref, messages=tuple(messages)))

    return _PreparedCorpus(
        sample_id=sample_id,
        batches=tuple(batches),
        raw_pair_count=raw_pair_count,
        skipped_empty_pair_count=skipped_empty_pair_count,
        official_longmemeval_protocol=True,
    )


def _merge_refs(
    existing: Sequence[SourceRef], incoming: Sequence[SourceRef]
) -> list[SourceRef]:
    merged = list(existing)
    for ref in incoming:
        if ref not in merged:
            merged.append(ref)
    return merged


def _validate_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("threshold must be numeric.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("threshold must be finite and within [0, 1].")
    return result


class Mem0LongMemEvalAdapter:
    """Fake-testable Mem0 ingestion, retrieval, packing, and cleanup."""

    def __init__(
        self,
        *,
        token_counter: TokenCounter,
        backend: Any | None = None,
        backend_factory: BackendFactory | None = None,
        clock: Clock = time.perf_counter,
        threshold: float = MEM0_OFFICIAL_THRESHOLD,
        top_k: int = MEM0_OFFICIAL_TOP_K,
        vector_client: _Closable | None = None,
        user_scope_factory: Callable[[str], str] | None = None,
        token_counter_identity: str | None = None,
    ) -> None:
        if (backend is None) == (backend_factory is None):
            raise ValueError("Provide exactly one of backend or backend_factory.")
        if not callable(token_counter):
            raise TypeError("token_counter must be callable.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive int.")

        self._backend = backend
        self._backend_factory = backend_factory
        self._clock = clock
        self._token_counter = token_counter
        explicit_counter_identity = (
            token_counter_identity.strip()
            if isinstance(token_counter_identity, str)
            else ""
        )
        if token_counter_identity is not None and not explicit_counter_identity:
            raise ValueError("token_counter_identity must be non-empty when set.")
        if token_counter is count_tokens:
            identity = tokenizer_proxy_identity()
            recognized_identity = (
                f"{identity['schema']}:{identity['encoding']}:"
                f"{identity['vocabulary_sha256']}"
            )
            if (
                explicit_counter_identity
                and explicit_counter_identity != recognized_identity
            ):
                raise ValueError(
                    "token_counter_identity disagrees with the recognized "
                    "memory_condense count_tokens identity."
                )
            self._token_counter_identity = recognized_identity
            self._token_counter_identity_verified = True
        elif explicit_counter_identity:
            # A caller can bind a custom counter to an experiment, but this
            # adapter cannot independently verify that declaration.
            self._token_counter_identity = explicit_counter_identity
            self._token_counter_identity_verified = False
        else:
            module = getattr(token_counter, "__module__", "unknown")
            name = getattr(
                token_counter,
                "__qualname__",
                getattr(token_counter, "__name__", type(token_counter).__name__),
            )
            self._token_counter_identity = f"callable:{module}.{name}:unverified"
            self._token_counter_identity_verified = False
        self._threshold = _validate_threshold(threshold)
        self._top_k = top_k
        self._vector_client = vector_client
        self._user_scope_factory = user_scope_factory or self._default_user_scope
        self._stats = Mem0AdapterStats(
            token_counter_identity=self._token_counter_identity,
            token_counter_identity_verified=self._token_counter_identity_verified,
        )
        self._ledger: dict[ScopedMemoryKey, list[SourceRef]] = {}
        self._scopes: list[str] = []
        self._scope_protocol: dict[str, bool] = {}
        self._active_scope: str | None = None
        self._poisoned_reason: str | None = None
        self._closed = False

    @staticmethod
    def _default_user_scope(sample_id: str) -> str:
        safe_sample = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("_")
        safe_sample = safe_sample or "sample"
        return f"longmemeval:{safe_sample}:{uuid.uuid4().hex}"

    @property
    def stats(self) -> Mem0AdapterStats:
        return self._stats

    @property
    def ledger(self) -> MemoryLedger:
        return MappingProxyType(
            {key: tuple(rows) for key, rows in self._ledger.items()}
        )

    @property
    def active_user_scope(self) -> str | None:
        return self._active_scope

    @property
    def supports_exact_source_provenance(self) -> bool:
        return False

    def require_exact_source_provenance(self) -> None:
        raise Mem0AttributionError(
            "Mem0 OSS 2.0.18 does not expose exact grounding for inferred "
            "memories. Only request-window attribution is available."
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise Mem0AdapterError("The Mem0 adapter is closed.")

    def _ensure_usable(self) -> None:
        self._ensure_open()
        if self._poisoned_reason is not None:
            raise Mem0PoisonedError(
                "The Mem0 adapter is poisoned after an ambiguous mutation; "
                f"only cleanup is safe ({self._poisoned_reason})."
            )

    def _get_backend(self) -> Any:
        self._ensure_usable()
        if self._backend is None:
            assert self._backend_factory is not None
            self._backend = self._backend_factory()
        return self._backend

    def _runtime_identity_snapshot(self) -> Mapping[str, Any]:
        value = getattr(self._backend, "runtime_identity", {})
        if not isinstance(value, Mapping):
            return MappingProxyType({})
        return MappingProxyType(copy.deepcopy(dict(value)))

    def _backend_is_certified(self) -> bool:
        if not isinstance(self._backend, _OwnedMem0Backend):
            return False
        identity = self._runtime_identity_snapshot()
        return (
            identity.get("certified") is True
            and identity.get("local_owned_state") is True
            and identity.get("on_disk") is True
            and identity.get("protocol")
            == "mem0-oss-2.0.18-certified-local-v1"
        )

    def _tokens(self, text: str) -> int:
        value = self._token_counter(text)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("token_counter must return a non-negative int.")
        return value

    @staticmethod
    def _render_add_input(messages: Sequence[tuple[str, str]]) -> str:
        return "\n".join(f"{role}: {content}" for role, content in messages)

    def _ledger_snapshot(self, scope: str | None = None) -> MemoryLedger:
        return MappingProxyType(
            {
                key: tuple(rows)
                for key, rows in self._ledger.items()
                if scope is None or key[0] == scope
            }
        )

    def _ingest_prepared(self, corpus: _PreparedCorpus) -> Mem0IngestResult:
        self._ensure_usable()
        user_scope = self._user_scope_factory(corpus.sample_id).strip()
        if not user_scope:
            raise ValueError("user_scope_factory returned an empty scope.")
        if user_scope in self._scopes:
            raise ValueError(
                "user_scope_factory returned a duplicate scope; every sample "
                "ingestion must be isolated."
            )
        returned_ids: list[str] = []
        recent_message_refs: deque[SourceRef] = deque(
            maxlen=MEM0_REQUEST_WINDOW_MESSAGES
        )
        backend = self._get_backend()
        self._scopes.append(user_scope)
        self._scope_protocol[user_scope] = corpus.official_longmemeval_protocol
        self._active_scope = user_scope

        for batch in corpus.batches:
            prior_refs: list[SourceRef] = []
            for ref in recent_message_refs:
                if ref not in prior_refs:
                    prior_refs.append(ref)
            request_window = tuple(_merge_refs(prior_refs, (batch.ref,)))
            messages = [
                {"role": role, "content": content}
                for role, content in batch.messages
            ]
            raw_message_tokens = self._tokens(
                self._render_add_input(batch.messages)
            )
            started = self._clock()
            response: Any = None
            operation_error: BaseException | None = None
            try:
                response = backend.add(
                    messages,
                    user_id=user_scope,
                    infer=True,
                )
                rows = _response_rows(response, operation="add")
                response_ids: list[str] = []
                for row_index, row in enumerate(rows):
                    memory_id = _memory_id(row)
                    if memory_id is None:
                        raise Mem0ProtocolError(
                            f"Mem0 add result {row_index} omitted its memory ID."
                        )
                    response_ids.append(memory_id)
            except BaseException as exc:
                operation_error = exc
                self._poisoned_reason = f"add request for {batch.ref.source!r} failed"
                raise
            finally:
                elapsed = max(0.0, self._clock() - started)
                self._stats = replace(
                    self._stats,
                    add_calls=self._stats.add_calls + 1,
                    add_attempted_calls=self._stats.add_attempted_calls + 1,
                    add_completed_calls=(
                        self._stats.add_completed_calls
                        + (0 if operation_error is not None else 1)
                    ),
                    add_failed_calls=(
                        self._stats.add_failed_calls
                        + (1 if operation_error is not None else 0)
                    ),
                    add_latency_s=self._stats.add_latency_s + elapsed,
                    add_raw_message_tokens=(
                        self._stats.add_raw_message_tokens + raw_message_tokens
                    ),
                )

            for memory_id in response_ids:
                returned_ids.append(memory_id)
                key = (user_scope, memory_id)
                self._ledger[key] = _merge_refs(
                    self._ledger.get(key, ()), request_window
                )
            for _role, _content in batch.messages:
                recent_message_refs.append(batch.ref)
            self._stats = replace(
                self._stats,
                add_returned_memories=(
                    self._stats.add_returned_memories + len(response_ids)
                ),
                unique_ledger_memories=len(self._ledger),
            )

        return Mem0IngestResult(
            sample_id=corpus.sample_id,
            user_scope=user_scope,
            batches_added=tuple(batch.ref for batch in corpus.batches),
            returned_memory_ids=tuple(returned_ids),
            ledger=self._ledger_snapshot(user_scope),
            attribution_kind=MEM0_ATTRIBUTION_KIND,
            supports_exact_source_provenance=False,
            date_exposure_kind=MEM0_DATE_EXPOSURE_KIND,
            raw_pair_count=corpus.raw_pair_count,
            skipped_empty_pair_count=corpus.skipped_empty_pair_count,
            official_longmemeval_protocol=corpus.official_longmemeval_protocol,
            comparison_certified=(
                corpus.official_longmemeval_protocol
                and self._backend_is_certified()
                and self._token_counter_identity_verified
            ),
            runtime_identity=self._runtime_identity_snapshot(),
            stats=self._stats,
        )

    def ingest_sample(self, sample: BenchmarkSample) -> Mem0IngestResult:
        """Ingest an already-normalized sample as a non-certifying ablation.

        ``BenchmarkSample`` cannot preserve LongMemEval's empty raw turns. Use
        :meth:`ingest_longmemeval_record` for official comparisons.
        """

        return self._ingest_prepared(_prepared_sample(sample))

    def ingest_longmemeval_record(
        self, record: Mapping[str, Any]
    ) -> Mem0IngestResult:
        """Ingest one raw record with official pairing and empty-pair parity."""

        return self._ingest_prepared(_prepared_longmemeval_record(record))

    ingest = ingest_sample

    def _normalize_pool(
        self, response: Any, *, scope: str
    ) -> tuple[Mem0Candidate, ...]:
        candidates: list[Mem0Candidate] = []
        for rank, row in enumerate(
            _response_rows(response, operation="search"), start=1
        ):
            memory_id = _memory_id(row)
            if memory_id is None:
                raise Mem0AttributionError(
                    f"Mem0 search result {rank} omitted its memory ID."
                )
            key = (scope, memory_id)
            attribution = self._ledger.get(key)
            if not attribution:
                raise Mem0AttributionError(
                    "Mem0 search returned an unaudited memory ID for this "
                    f"scope: {memory_id!r}."
                )
            metadata_value = row.get("metadata", {})
            metadata = (
                dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
            )
            candidates.append(
                Mem0Candidate(
                    rank=rank,
                    memory_id=memory_id,
                    text=_memory_text(row),
                    score=_memory_score(row),
                    created_at=_memory_created_at(row),
                    metadata=MappingProxyType(metadata),
                    request_window_attribution=tuple(attribution),
                    attribution_kind=MEM0_ATTRIBUTION_KIND,
                    raw=copy.deepcopy(row),
                )
            )
        return tuple(candidates)

    @staticmethod
    def _candidate_label(candidate: Mem0Candidate) -> str:
        refs = candidate.request_window_attribution

        def values(name: str) -> str:
            found: list[str] = []
            for ref in refs:
                value = _safe_label_value(getattr(ref, name))
                if value and value not in found:
                    found.append(value)
            return ",".join(found)

        return (
            f"[Memory {candidate.rank} | id={_safe_label_value(candidate.memory_id)} | "
            f"attribution={MEM0_ATTRIBUTION_KIND} | "
            f"source={values('source')} | session={values('session')} | "
            f"batch={values('batch_index')} | date={values('date')}]"
        )

    @classmethod
    def _render_enriched_candidate(cls, candidate: Mem0Candidate) -> str:
        return f"{cls._candidate_label(candidate)}\n{candidate.text}"

    @staticmethod
    def _render_official_context(
        candidates: Sequence[Mem0Candidate],
    ) -> str:
        """Match the official Mem0 benchmark's dated memory-only rendering."""

        grouped: dict[str, list[Mem0Candidate]] = {}
        for candidate in sorted(
            candidates,
            key=lambda item: (item.created_at or "", item.rank),
        ):
            if candidate.created_at is None:
                raise Mem0ProtocolError(
                    "Official Mem0 rendering requires created_at on every "
                    f"returned memory ({candidate.memory_id!r})."
                )
            label = _official_date_label(candidate.created_at)
            grouped.setdefault(label, []).append(candidate)
        sections = [
            f"--- {label} ---\n"
            + "\n".join(f"- {candidate.text}" for candidate in rows)
            for label, rows in grouped.items()
        ]
        return "\n".join(sections)

    @classmethod
    def _render_context(
        cls,
        candidates: Sequence[Mem0Candidate],
        *,
        rendering_mode: str,
    ) -> str:
        if rendering_mode == MEM0_CERTIFIED_RENDERING:
            return cls._render_official_context(candidates)
        if rendering_mode == MEM0_ENRICHED_RENDERING:
            return MEM0_CONTEXT_SEPARATOR.join(
                cls._render_enriched_candidate(candidate)
                for candidate in candidates
            )
        raise ValueError(
            "rendering_mode must be MEM0_CERTIFIED_RENDERING or "
            "MEM0_ENRICHED_RENDERING."
        )

    def _render_prompt(
        self, renderer: PromptRenderer, query: str, context: str
    ) -> str:
        prompt = renderer(query, context)
        if not isinstance(prompt, str):
            raise TypeError("prompt_renderer must return str.")
        return prompt

    def search(
        self,
        query: str,
        *,
        max_prompt_tokens: int,
        prompt_renderer: PromptRenderer,
        prompt_token_overhead: int = 0,
        context_token_budget: int | None = None,
        user_scope: str | None = None,
        threshold: float | None = None,
        rendering_mode: str = MEM0_CERTIFIED_RENDERING,
    ) -> Mem0SearchResult:
        """Search under a complete rendered responder prompt-proxy budget.

        ``max_prompt_tokens`` and ``prompt_token_overhead`` are retained API
        spellings. Both refer to the declared local token-count proxy. Callers
        should include any chat framing allowance in the overhead.
        """

        self._ensure_usable()
        if (
            isinstance(max_prompt_tokens, bool)
            or not isinstance(max_prompt_tokens, int)
            or max_prompt_tokens < 0
        ):
            raise ValueError("max_prompt_tokens must be a non-negative int.")
        if (
            isinstance(prompt_token_overhead, bool)
            or not isinstance(prompt_token_overhead, int)
            or prompt_token_overhead < 0
        ):
            raise ValueError("prompt_token_overhead must be a non-negative int.")
        if context_token_budget is not None and (
            isinstance(context_token_budget, bool)
            or not isinstance(context_token_budget, int)
            or context_token_budget < 0
        ):
            raise ValueError("context_token_budget must be a non-negative int or null.")
        if not callable(prompt_renderer):
            raise TypeError("prompt_renderer must be callable.")
        if rendering_mode not in {
            MEM0_CERTIFIED_RENDERING,
            MEM0_ENRICHED_RENDERING,
        }:
            raise ValueError(
                "rendering_mode must be MEM0_CERTIFIED_RENDERING or "
                "MEM0_ENRICHED_RENDERING."
            )
        scope = user_scope or self._active_scope
        if not scope or scope not in self._scopes:
            raise ValueError("Search requires a user scope returned by ingest_sample().")
        effective_threshold = _validate_threshold(
            self._threshold if threshold is None else threshold
        )
        official_search_protocol = (
            self._top_k == MEM0_OFFICIAL_TOP_K
            and effective_threshold == MEM0_OFFICIAL_THRESHOLD
        )

        empty_prompt = self._render_prompt(prompt_renderer, query, "")
        empty_prompt_tokens = self._tokens(empty_prompt) + prompt_token_overhead
        if empty_prompt_tokens > max_prompt_tokens:
            raise Mem0PromptBudgetError(
                "The responder prompt without retrieved context already exceeds "
                f"the cap ({empty_prompt_tokens} > {max_prompt_tokens})."
            )

        query_tokens = self._tokens(query)
        backend = self._get_backend()
        started = self._clock()
        try:
            response = backend.search(
                query,
                top_k=self._top_k,
                filters={"user_id": scope},
                threshold=effective_threshold,
                rerank=False,
                explain=False,
            )
        finally:
            elapsed = max(0.0, self._clock() - started)
            self._stats = replace(
                self._stats,
                search_calls=self._stats.search_calls + 1,
                search_latency_s=self._stats.search_latency_s + elapsed,
                search_query_tokens=self._stats.search_query_tokens + query_tokens,
            )

        raw_pool = self._normalize_pool(response, scope=scope)
        raw_tokens = sum(self._tokens(candidate.text) for candidate in raw_pool)
        packed: list[Mem0Candidate] = []
        diagnostics: list[Mem0PackDiagnostic] = []
        context = ""
        context_tokens = 0
        prompt = empty_prompt
        prompt_tokens = empty_prompt_tokens

        for candidate in raw_pool:
            rendered = self._render_context(
                (candidate,), rendering_mode=rendering_mode
            )
            audit_rendered = self._render_enriched_candidate(candidate)
            rendered_tokens = self._tokens(rendered)
            if not candidate.text:
                selected = False
                reason = "empty_memory"
                proposed_context_tokens = context_tokens
                proposed_prompt_tokens = prompt_tokens
                proposed = context
                proposed_prompt = prompt
            else:
                proposed_candidates = [*packed, candidate]
                proposed = self._render_context(
                    proposed_candidates,
                    rendering_mode=rendering_mode,
                )
                proposed_context_tokens = self._tokens(proposed)
                proposed_prompt = self._render_prompt(
                    prompt_renderer, query, proposed
                )
                proposed_prompt_tokens = (
                    self._tokens(proposed_prompt) + prompt_token_overhead
                )
                if (
                    context_token_budget is not None
                    and proposed_context_tokens > context_token_budget
                ):
                    selected = False
                    reason = "context_token_budget"
                elif proposed_prompt_tokens > max_prompt_tokens:
                    selected = False
                    reason = "prompt_token_budget"
                else:
                    selected = True
                    reason = "selected"

            if selected:
                packed.append(candidate)
                context = proposed
                context_tokens = proposed_context_tokens
                prompt = proposed_prompt
                prompt_tokens = proposed_prompt_tokens
            diagnostics.append(
                Mem0PackDiagnostic(
                    candidate=candidate,
                    rendered=rendered,
                    audit_rendered=audit_rendered,
                    rendered_tokens=rendered_tokens,
                    selected=selected,
                    reason=reason,
                    context_tokens_after=context_tokens,
                    prompt_token_proxy_after=prompt_tokens,
                    prompt_tokens_after=prompt_tokens,
                )
            )

        # Re-render and recount the exact strings returned to the caller; BPE
        # boundaries can change across labels, separators, and prompt framing.
        context = self._render_context(packed, rendering_mode=rendering_mode)
        context_tokens = self._tokens(context)
        prompt = self._render_prompt(prompt_renderer, query, context)
        prompt_tokens = self._tokens(prompt) + prompt_token_overhead
        if prompt_tokens > max_prompt_tokens:
            raise Mem0PromptBudgetError(
                "Final prompt recount exceeded the declared cap; the renderer "
                "must be deterministic during one search call."
            )

        self._stats = replace(
            self._stats,
            search_raw_memory_tokens=(
                self._stats.search_raw_memory_tokens + raw_tokens
            ),
            search_context_tokens=(
                self._stats.search_context_tokens + context_tokens
            ),
            search_prompt_token_proxy=(
                self._stats.search_prompt_token_proxy + prompt_tokens
            ),
            search_prompt_tokens=self._stats.search_prompt_tokens + prompt_tokens,
            search_returned_memories=(
                self._stats.search_returned_memories + len(raw_pool)
            ),
            search_packed_memories=(
                self._stats.search_packed_memories + len(packed)
            ),
        )
        return Mem0SearchResult(
            user_scope=scope,
            query=query,
            context=context,
            context_tokens=context_tokens,
            prompt=prompt,
            prompt_token_proxy=prompt_tokens,
            max_prompt_token_proxy=max_prompt_tokens,
            prompt_token_proxy_overhead=prompt_token_overhead,
            empty_context_prompt_token_proxy=empty_prompt_tokens,
            residual_prompt_token_proxy=max_prompt_tokens - prompt_tokens,
            prompt_token_proxy_budget_compliant=True,
            token_counter_identity=self._token_counter_identity,
            token_counter_identity_verified=self._token_counter_identity_verified,
            prompt_tokens=prompt_tokens,
            max_prompt_tokens=max_prompt_tokens,
            prompt_token_overhead=prompt_token_overhead,
            empty_context_prompt_tokens=empty_prompt_tokens,
            residual_prompt_tokens=max_prompt_tokens - prompt_tokens,
            prompt_budget_certified=True,
            packed=tuple(packed),
            raw_pool=raw_pool,
            diagnostics=tuple(diagnostics),
            raw_response=copy.deepcopy(response),
            attribution_kind=MEM0_ATTRIBUTION_KIND,
            supports_exact_source_provenance=False,
            rendering_mode=rendering_mode,
            certified_rendering=(rendering_mode == MEM0_CERTIFIED_RENDERING),
            official_longmemeval_protocol=self._scope_protocol.get(scope, False),
            official_search_protocol=official_search_protocol,
            comparison_certified=(
                rendering_mode == MEM0_CERTIFIED_RENDERING
                and self._scope_protocol.get(scope, False)
                and official_search_protocol
                and self._backend_is_certified()
                and self._token_counter_identity_verified
            ),
            runtime_identity=self._runtime_identity_snapshot(),
            stats=self._stats,
        )

    retrieve = search

    def release_scope(self, user_scope: str) -> None:
        """Delete and forget one completed sample before processing the next."""

        self._ensure_usable()
        if not isinstance(user_scope, str) or user_scope not in self._scopes:
            raise ValueError("release_scope requires a live scope from ingest.")
        backend = self._get_backend()
        try:
            backend.delete_all(user_id=user_scope)
        except BaseException:
            self._poisoned_reason = f"scope release for {user_scope!r} failed"
            raise
        self._scopes.remove(user_scope)
        self._scope_protocol.pop(user_scope, None)
        for key in [key for key in self._ledger if key[0] == user_scope]:
            del self._ledger[key]
        if self._active_scope == user_scope:
            self._active_scope = self._scopes[-1] if self._scopes else None
        self._stats = replace(
            self._stats,
            released_scopes=self._stats.released_scopes + 1,
            unique_ledger_memories=len(self._ledger),
        )

    release_sample = release_scope

    def cleanup(self) -> None:
        """Delete scopes, close owned resources, and clear local attribution."""

        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        backend = self._backend
        if backend is not None:
            delete_all = getattr(backend, "delete_all", None)
            if callable(delete_all):
                for scope in self._scopes:
                    try:
                        delete_all(user_id=scope)
                    except BaseException as exc:
                        errors.append(exc)

            close_backend = getattr(backend, "close", None)
            if callable(close_backend):
                try:
                    close_backend()
                except BaseException as exc:
                    errors.append(exc)

        if self._vector_client is not None:
            close_vector = getattr(self._vector_client, "close", None)
            if callable(close_vector):
                try:
                    close_vector()
                except BaseException as exc:
                    errors.append(exc)

        self._ledger.clear()
        self._scopes.clear()
        self._scope_protocol.clear()
        self._active_scope = None
        _raise_cleanup_errors(errors, "Mem0 adapter cleanup failed")

    close = cleanup

    def __enter__(self) -> Mem0LongMemEvalAdapter:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        try:
            self.cleanup()
        except BaseException as cleanup_exc:
            if exc is None:
                raise
            exc.add_note(f"Mem0 cleanup also failed: {cleanup_exc!r}")
        return False


Mem0Adapter = Mem0LongMemEvalAdapter


__all__ = [
    "MEM0AI_PIN",
    "MEM0_API_VERSION",
    "MEM0_ATTRIBUTION_KIND",
    "MEM0_BM25_MODEL",
    "MEM0_CERTIFIED_RENDERING",
    "MEM0_CONTEXT_SEPARATOR",
    "MEM0_DATE_EXPOSURE_KIND",
    "MEM0_ENRICHED_RENDERING",
    "MEM0_OFFICIAL_THRESHOLD",
    "MEM0_OFFICIAL_TOP_K",
    "MEM0_PROVIDER_USAGE_STATUS",
    "MEM0_REQUEST_WINDOW_MESSAGES",
    "MEM0_SPACY_MODEL",
    "Mem0Adapter",
    "Mem0AdapterError",
    "Mem0AdapterStats",
    "Mem0AttributionError",
    "Mem0Candidate",
    "Mem0ConfigurationError",
    "Mem0DependencyError",
    "Mem0IngestResult",
    "Mem0LongMemEvalAdapter",
    "Mem0OSSBackendFactory",
    "Mem0PackDiagnostic",
    "Mem0PoisonedError",
    "Mem0PromptBudgetError",
    "Mem0ProtocolError",
    "Mem0SearchResult",
    "Mem0StackIdentity",
    "SourcePair",
    "SourceRef",
]
