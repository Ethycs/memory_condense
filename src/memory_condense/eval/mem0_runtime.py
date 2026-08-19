"""Pinned Mem0 stack preflight, isolated state, and backend construction."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import shutil
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from memory_condense.eval.mem0_models import (
    MEM0AI_PIN,
    MEM0_API_VERSION,
    MEM0_BM25_MODEL,
    MEM0_SPACY_MODEL,
    Mem0ConfigurationError,
    Mem0DependencyError,
    Mem0StackIdentity,
    StackPreflight,
)


_OWNERSHIP_MARKER = ".memory-condense-owned-state"


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
