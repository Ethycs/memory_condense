"""Exact Mem0 runtime seams used only by the resumable shard coordinator.

The frozen historical adapter source remains byte-for-byte unchanged.  This
tool-owned module adds two capabilities beside (never inside) the one-shot
factory: reopening a hash-verified owned state and suspending every live local
handle without deleting that state.  Provider calls remain owned by the exact
Terra transport in :mod:`tools.mem0_eval.production_binding`.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping

from . import production_binding as binding
from .policy import (
    MEM0_EXTRACTION_GATEWAY_URL,
    MEM0_EXTRACTION_MODEL,
    MEM0_EXTRACTION_PROVIDER,
    MEM0_EXTRACTION_REVISION,
    MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
    canonical_json_sha256,
)
from .resumable import OWNERSHIP_MARKER, ResumableShardError, state_tree_receipt
from .source_compat import (
    DEFAULT_MODEL_NAME,
    SOURCE_LAYOUT,
    _OwnedMem0Backend,
    _assert_mem0_state_binding,
    _assert_no_live_mem0_telemetry,
    _redacted_stable_config,
    _remove_owned_state,
    _sha256_json,
    count_tokens,
)


RESUMABLE_FACTORY_FORMAT = "memory-condense-mem0-resumable-factory-v1"
RESUMABLE_SUSPEND_FORMAT = "memory-condense-mem0-resumable-suspend-v1"
RESUMABLE_TRANSPORT_CLOSURE_FORMAT = (
    "memory-condense-mem0-resumable-transport-closure-v1"
)
RESUMABLE_WRITE_ACTIVITY_FORMAT = (
    "memory-condense-mem0-resumable-write-activity-v1"
)
RESUMABLE_PROVIDER_FREE_GATE_FORMAT = (
    "memory-condense-mem0-resumable-provider-free-factory-gate-v1"
)


class ZeroCallExtractionTransport:
    """Provider-free LLM seam for a sealed full-prefix search-only process."""

    production_eligible = True

    def __init__(self) -> None:
        self._closed = False
        self._rejected = 0

    def generate_response(self, *_args: Any, **_kwargs: Any) -> str:
        self._rejected += 1
        raise binding.TransportAttemptLimitExceeded(
            "sealed full-prefix runtime authorizes zero extraction sends"
        )

    def request_token_state_receipt(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "contract": "stateless-request-token-state-v1",
                "persisted_request_token_state": False,
                "retained_request_token_state_bytes": 0,
                "request_token_state_evidence_kind": (
                    "local_injected_request_token_state_contract"
                ),
                "external_provider_persistence_certified": False,
            }
        )

    def assert_call_budget_closed(self) -> None:
        if self._rejected:
            raise binding.ProductionBindingError(
                "zero-call extraction transport observed a rejected send"
            )

    def transport_receipt(self) -> dict[str, Any]:
        return {
            "kind": "local_transport_send_cap",
            "role": "extraction",
            "authorized": 0,
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "rejected": self._rejected,
            "retries_authorized": 0,
            "provider_usage_status": "not_applicable_zero_authorized",
            "provider_usage_records": 0,
            "provider_input_tokens": 0,
            "provider_output_tokens": 0,
            "provider_total_tokens": 0,
            "provider_latency_s": 0.0,
            "production_eligible": True,
            "provider": MEM0_EXTRACTION_PROVIDER,
            "model": MEM0_EXTRACTION_MODEL,
            "revision": MEM0_EXTRACTION_REVISION,
            "route_identity_sha256": MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
            "gateway_url": MEM0_EXTRACTION_GATEWAY_URL,
            "sdk_retries": 0,
            "http_transport_retries": 0,
            "cap_boundary": "deny_before_provider_transport",
            "external_http_attempts_certified": True,
            "external_provider_persistence_certified": False,
        }

    def close(self) -> None:
        self._closed = True


class Mem0WriteActivityMeter:
    """Observe every local embedding and persisted-state mutation.

    The meter is installed on the exact live Mem0 objects after construction
    and removed before suspension.  It records both attempts and failures so
    a successful checkpoint cannot silently omit a partially observed write.
    """

    def __init__(self, *, token_counter: Callable[[str], int] = count_tokens) -> None:
        if not callable(token_counter):
            raise TypeError("write-activity token counter must be callable")
        self._tokens = token_counter
        self._lock = threading.Lock()
        self._bindings: list[tuple[Any, str, Any, Any]] = []
        self._installed = False
        self._restored = False
        self.embedding_attempted = 0
        self.embedding_completed = 0
        self.embedding_failed = 0
        self.embedding_input_token_proxy = 0
        self.embedding_latency_s = 0.0
        self.storage_attempted = 0
        self.storage_completed = 0
        self.storage_failed = 0
        self.storage_latency_s = 0.0

    def _embedding_wrapper(self, original: Any, *, batch: bool) -> Any:
        def wrapped(texts: Any, *args: Any, **kwargs: Any) -> Any:
            values = list(texts) if batch else [texts]
            if not values or any(not isinstance(value, str) for value in values):
                raise binding.ProductionBindingError(
                    "Mem0 embedding input is not exact text"
                )
            operations = len(values)
            token_proxy = sum(int(self._tokens(value)) for value in values)
            started = time.perf_counter()
            with self._lock:
                self.embedding_attempted += operations
                self.embedding_input_token_proxy += token_proxy
            try:
                result = original(values if batch else texts, *args, **kwargs)
            except BaseException:
                elapsed = max(0.0, time.perf_counter() - started)
                with self._lock:
                    self.embedding_failed += operations
                    self.embedding_latency_s += elapsed
                raise
            elapsed = max(0.0, time.perf_counter() - started)
            with self._lock:
                self.embedding_completed += operations
                self.embedding_latency_s += elapsed
            return result

        return wrapped

    def _storage_wrapper(self, original: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            with self._lock:
                self.storage_attempted += 1
            try:
                result = original(*args, **kwargs)
            except BaseException:
                elapsed = max(0.0, time.perf_counter() - started)
                with self._lock:
                    self.storage_failed += 1
                    self.storage_latency_s += elapsed
                raise
            elapsed = max(0.0, time.perf_counter() - started)
            with self._lock:
                self.storage_completed += 1
                self.storage_latency_s += elapsed
            return result

        return wrapped

    def _bind(self, target: Any, name: str, wrapper: Any) -> None:
        original = getattr(target, name, None)
        if not callable(original):
            raise binding.ProductionBindingError(
                f"Mem0 write meter target omitted {name}"
            )
        wrapped = wrapper(original)
        setattr(target, name, wrapped)
        if getattr(target, name, None) is not wrapped:
            raise binding.ProductionBindingError(
                f"Mem0 write meter could not bind {name}"
            )
        self._bindings.append((target, name, original, wrapped))

    def install(self, adapter: Any) -> Callable[[], None]:
        if self._installed:
            raise binding.ProductionBindingError(
                "Mem0 write activity meter is single-install"
            )
        owned = getattr(adapter, "_backend", None)
        memory = getattr(owned, "backend", None)
        if memory is None:
            raise binding.ProductionBindingError(
                "Mem0 write activity meter omitted Memory"
            )
        embedder = getattr(memory, "embedding_model", None)
        stores = binding._materialize_exact_qdrant_stores(memory)
        history = getattr(memory, "db", None)
        if embedder is None or history is None:
            raise binding.ProductionBindingError(
                "Mem0 write activity targets are incomplete"
            )
        try:
            self._bind(
                embedder,
                "embed",
                lambda original: self._embedding_wrapper(original, batch=False),
            )
            self._bind(
                embedder,
                "embed_batch",
                lambda original: self._embedding_wrapper(original, batch=True),
            )
            for store in stores:
                for name in ("insert", "update", "delete"):
                    self._bind(store, name, self._storage_wrapper)
            for name in ("save_messages", "add_history", "batch_add_history"):
                self._bind(history, name, self._storage_wrapper)
        except BaseException:
            self._restore_bound(allow_uninstalled=True)
            raise
        self._installed = True

        def restore() -> None:
            self.restore()

        return restore

    def _restore_bound(self, *, allow_uninstalled: bool = False) -> None:
        errors: list[str] = []
        for target, name, original, wrapped in reversed(self._bindings):
            if getattr(target, name, None) is not wrapped:
                errors.append(f"{name} wrapper changed before restoration")
                continue
            try:
                setattr(target, name, original)
            except BaseException as exc:
                errors.append(f"{name} restoration failed: {type(exc).__name__}")
        self._bindings.clear()
        if errors and not allow_uninstalled:
            raise binding.ProductionBindingError("; ".join(errors))

    def restore(self) -> None:
        if not self._installed or self._restored:
            raise binding.ProductionBindingError(
                "Mem0 write activity meter restoration is out of order"
            )
        self._restore_bound()
        self._restored = True

    def receipt(self) -> Mapping[str, Any]:
        with self._lock:
            body = {
                "format": RESUMABLE_WRITE_ACTIVITY_FORMAT,
                "embedding_attempted": self.embedding_attempted,
                "embedding_completed": self.embedding_completed,
                "embedding_failed": self.embedding_failed,
                "embedding_input_token_proxy": self.embedding_input_token_proxy,
                "embedding_latency_s": self.embedding_latency_s,
                "storage_attempted": self.storage_attempted,
                "storage_completed": self.storage_completed,
                "storage_failed": self.storage_failed,
                "storage_latency_s": self.storage_latency_s,
                "wrappers_installed": self._installed,
                "wrappers_restored": self._restored,
            }
        return MappingProxyType(
            {**body, "receipt_sha256": canonical_json_sha256(body)}
        )

    def assert_closed(self) -> None:
        receipt = self.receipt()
        if (
            receipt["wrappers_installed"] is not True
            or receipt["wrappers_restored"] is not True
            or receipt["embedding_attempted"] != receipt["embedding_completed"]
            or receipt["embedding_failed"] != 0
            or receipt["storage_attempted"] != receipt["storage_completed"]
            or receipt["storage_failed"] != 0
        ):
            raise binding.ProductionBindingError(
                "Mem0 write activity receipt did not close exactly"
            )


def _ownership_token(state_root: Path) -> str:
    marker = state_root / OWNERSHIP_MARKER
    if marker.is_symlink() or not marker.is_file():
        raise ResumableShardError("adopted Mem0 state omitted its ownership marker")
    token = marker.read_text(encoding="utf-8")
    if len(token) != 32 or any(char not in "0123456789abcdef" for char in token):
        raise ResumableShardError("adopted Mem0 ownership token is invalid")
    return token


@contextmanager
def _mem0_state_environment(state_root: Path) -> Iterator[None]:
    names = ("MEM0_TELEMETRY", "MEM0_DIR")
    before = {name: os.environ.get(name) for name in names}
    os.environ["MEM0_TELEMETRY"] = "false"
    os.environ["MEM0_DIR"] = str(state_root)
    try:
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _adopt_owned_backend(authorization: Any, state_root: Path) -> Any:
    """Reopen an authenticated closed state using the historical types."""

    from memory_condense.eval import mem0_adapter

    token = _ownership_token(state_root)
    installed = importlib.metadata.version("mem0ai")
    if installed != mem0_adapter.MEM0AI_PIN:
        raise binding.ProductionBindingError(
            "adopted state requires the exact historical mem0ai pin"
        )
    stack_identity = binding._exact_mem0_stack_preflight()
    if not isinstance(stack_identity, mem0_adapter.Mem0StackIdentity):
        raise binding.ProductionBindingError("Mem0 stack identity type changed")
    if not stack_identity.certified:
        raise binding.ProductionBindingError("Mem0 stack is not certified")

    config = binding._materialize_mem0_config(authorization, state_root)
    effective_config = copy.deepcopy(config)
    vector_config = effective_config["vector_store"]["config"]
    base_collection = str(vector_config["collection_name"]).strip()
    collection_name = f"{base_collection}-{token[:12]}"
    vector_config["collection_name"] = collection_name
    stable_payload = {
        "protocol": "mem0-oss-2.0.18-certified-local-v1",
        "config": _redacted_stable_config(config, state_root=state_root),
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
    if stable_fingerprint != authorization.mem0_stable_config_sha256:
        raise binding.ProductionBindingError(
            "adopted state stable config differs from authorization"
        )

    with _mem0_state_environment(state_root):
        _assert_no_live_mem0_telemetry()
        _assert_mem0_state_binding(state_root)
        module = importlib.import_module("mem0")
        if str(getattr(module, "__version__", "")) != mem0_adapter.MEM0AI_PIN:
            raise binding.ProductionBindingError("imported Mem0 version changed")
        memory_type = getattr(module, "Memory", None)
        from_config = getattr(memory_type, "from_config", None)
        if not callable(from_config):
            raise binding.ProductionBindingError("Mem0 omitted Memory.from_config")
        with binding._mem0_default_llm_constructor_environment():
            memory = from_config(config_dict=effective_config)
    return _OwnedMem0Backend(
        backend=memory,
        state_root=state_root,
        ownership_token=token,
        collection_name=collection_name,
        stable_config_fingerprint=stable_fingerprint,
        effective_config_fingerprint=effective_fingerprint,
        runtime_identity=runtime_identity,
    )


class ResumableExactMem0AdapterFactory:
    """Fresh/adopt factory with a per-segment exact Terra transport budget."""

    production_eligible = True

    def __init__(
        self,
        authorization: Any,
        *,
        segment_authorized_calls: int,
        adopt_existing_state: bool,
        user_scope: str,
        before_http_send: Callable[[Any], Any] | None,
        expected_ownership_token_sha256: str | None = None,
    ) -> None:
        binding._require_exact_authorization(authorization, stage="retrieval")
        binding.validate_production_mem0_config(authorization)
        binding.validate_local_bge_m3_contract(authorization)
        if (
            isinstance(segment_authorized_calls, bool)
            or not isinstance(segment_authorized_calls, int)
            or segment_authorized_calls < 0
            or segment_authorized_calls > authorization.authorized_extraction_calls
        ):
            raise binding.ProductionBindingError("segment call budget is invalid")
        if segment_authorized_calls and not callable(before_http_send):
            raise binding.ProductionBindingError(
                "a positive resumable segment requires its durable send boundary"
            )
        if not segment_authorized_calls and before_http_send is not None:
            raise binding.ProductionBindingError(
                "zero-call resumable mode cannot bind a send callback"
            )
        if not isinstance(adopt_existing_state, bool):
            raise TypeError("adopt_existing_state must be boolean")
        if not isinstance(user_scope, str) or not user_scope.strip():
            raise binding.ProductionBindingError("resumable user scope is empty")
        if expected_ownership_token_sha256 is not None and (
            not isinstance(expected_ownership_token_sha256, str)
            or len(expected_ownership_token_sha256) != 64
        ):
            raise binding.ProductionBindingError(
                "expected ownership-token digest is invalid"
            )
        self._authorization = authorization
        self._segment_authorized = segment_authorized_calls
        self._adopt = adopt_existing_state
        self._scope = user_scope.strip()
        self._before_http_send = before_http_send
        self._expected_token_sha = expected_ownership_token_sha256
        self._called = False
        self._transport: Any | None = None
        self._bound_bm25: Mapping[str, Any] | None = None
        self._bound_embedder: Mapping[str, Any] | None = None
        self._mode: str | None = None

    def __call__(self, owned_state_dir: Path) -> Any:
        if self._called:
            raise binding.ProductionBindingError(
                "resumable exact Mem0 factory is single-use"
            )
        self._called = True
        state_root = Path(owned_state_dir).resolve(strict=False)
        if self._adopt != state_root.exists():
            raise binding.ProductionBindingError(
                "resumable factory fresh/adopt state expectation mismatch"
            )
        if self._adopt:
            token = _ownership_token(state_root)
            token_sha = hashlib.sha256(token.encode("ascii")).hexdigest()
            if token_sha != self._expected_token_sha:
                raise binding.ProductionBindingError(
                    "adopted ownership token differs from checkpoint"
                )
        transport: Any
        if self._segment_authorized:
            transport = binding.LiteLLMTerraExtractionTransport(
                authorized=self._segment_authorized,
                _before_http_send=self._before_http_send,
            )
        else:
            transport = ZeroCallExtractionTransport()
        self._transport = transport
        owned: Any | None = None
        try:
            from memory_condense.eval import mem0_adapter

            config = binding._materialize_mem0_config(
                self._authorization, state_root
            )
            if self._adopt:
                owned = _adopt_owned_backend(self._authorization, state_root)
                self._mode = "adopt"
            else:
                backend_factory = mem0_adapter.Mem0OSSBackendFactory(
                    config=config,
                    llm_model_id=MEM0_EXTRACTION_MODEL,
                    embedder_model_id=DEFAULT_MODEL_NAME,
                    owned_state_dir=state_root,
                    _stack_preflight=binding._exact_mem0_stack_preflight,
                )
                with binding._mem0_default_llm_constructor_environment():
                    owned = backend_factory()
                self._mode = "fresh"
            memory = getattr(owned, "backend", None)
            if memory is None:
                raise binding.ProductionBindingError(
                    "resumable owned backend omitted Memory"
                )
            binding._materialize_exact_qdrant_stores(memory)
            bound_bm25 = binding._bind_exact_bm25_encoders(memory)
            bound_embedder = binding._bound_embedder_receipt(
                memory, self._authorization
            )
            binding._bind_memory_transport(memory, transport)
            adapter = mem0_adapter.Mem0LongMemEvalAdapter(
                backend=owned,
                token_counter=count_tokens,
                user_scope_factory=lambda _sample_id: self._scope,
            )
            adapter._production_extraction_transport = transport
            adapter._bound_embedder_receipt = MappingProxyType(bound_embedder)
            adapter._bound_bm25_receipt = MappingProxyType(bound_bm25)
            adapter._resumable_factory = self
            self._bound_bm25 = MappingProxyType(bound_bm25)
            self._bound_embedder = MappingProxyType(bound_embedder)
            return adapter
        except BaseException:
            if owned is not None:
                try:
                    binding._harden_owned_qdrant_cleanup(owned)
                    owned.close()
                except BaseException:
                    pass
            transport.close()
            raise

    def transport_receipt(self) -> Mapping[str, Any]:
        if self._transport is None:
            raise binding.ProductionBindingError(
                "resumable transport has not been constructed"
            )
        return MappingProxyType(dict(self._transport.transport_receipt()))

    def binding_receipt(self) -> Mapping[str, Any]:
        body = {
            "format": RESUMABLE_FACTORY_FORMAT,
            "mode": self._mode,
            "segment_authorized_calls": self._segment_authorized,
            "full_authorized_calls": (
                self._authorization.authorized_extraction_calls
            ),
            "user_scope_sha256": hashlib.sha256(
                self._scope.encode("utf-8")
            ).hexdigest(),
            "bound_embedder": (
                dict(self._bound_embedder)
                if self._bound_embedder is not None
                else None
            ),
            "bound_bm25": (
                dict(self._bound_bm25) if self._bound_bm25 is not None else None
            ),
            "transport": (
                dict(self._transport.transport_receipt())
                if self._transport is not None
                else None
            ),
        }
        return MappingProxyType(
            {**body, "receipt_sha256": canonical_json_sha256(body)}
        )

    def transport_closure_receipt(self) -> Mapping[str, Any]:
        transport = self._transport
        if transport is None:
            raise binding.ProductionBindingError(
                "resumable transport has not been constructed"
            )
        if getattr(transport, "_closed", None) is not True:
            raise binding.ProductionBindingError(
                "resumable transport closure was requested before close"
            )
        transport.assert_call_budget_closed()
        observed = dict(transport.transport_receipt())
        authorized = self._segment_authorized
        required = {
            "authorized": authorized,
            "attempted": authorized,
            "completed": authorized,
            "failed": 0,
            "rejected": 0,
            "sdk_retries": 0,
            "http_transport_retries": 0,
            "provider_usage_records": authorized,
        }
        for field, expected in required.items():
            if observed.get(field) != expected:
                raise binding.ProductionBindingError(
                    f"resumable transport closure {field} mismatch"
                )
        for field in (
            "provider_input_tokens",
            "provider_output_tokens",
            "provider_total_tokens",
        ):
            value = observed.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise binding.ProductionBindingError(
                    f"resumable transport closure {field} is invalid"
                )
        if observed["provider_total_tokens"] != (
            observed["provider_input_tokens"]
            + observed["provider_output_tokens"]
        ):
            raise binding.ProductionBindingError(
                "resumable transport closure provider tokens do not close"
            )
        latency = observed.get("provider_latency_s")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or float(latency) < 0
        ):
            raise binding.ProductionBindingError(
                "resumable transport closure latency is invalid"
            )
        expected_status = (
            "provider_reported_exact"
            if authorized
            else "not_applicable_zero_authorized"
        )
        if observed.get("provider_usage_status") != expected_status:
            raise binding.ProductionBindingError(
                "resumable transport closure usage status changed"
            )
        body = {
            "format": RESUMABLE_TRANSPORT_CLOSURE_FORMAT,
            "segment_authorized_calls": authorized,
            "transport_closed": True,
            "budget_closed_exactly": True,
            "provider_usage_complete": True,
            "sdk_retries": 0,
            "http_transport_retries": 0,
            "transport_receipt": observed,
            "transport_receipt_sha256": canonical_json_sha256(observed),
        }
        return MappingProxyType(
            {**body, "receipt_sha256": canonical_json_sha256(body)}
        )


def _namespace_persisted_memory_count(memory: Any, user_scope: str) -> int:
    if not isinstance(user_scope, str) or not user_scope:
        raise binding.ProductionBindingError(
            "persisted-memory count requires one namespace"
        )
    store = binding._materialize_exact_qdrant_stores(memory)[0]
    create_filter = getattr(store, "_create_filter", None)
    client = getattr(store, "client", None)
    count = getattr(client, "count", None)
    collection_name = getattr(store, "collection_name", None)
    if (
        not callable(create_filter)
        or not callable(count)
        or not isinstance(collection_name, str)
        or not collection_name
    ):
        raise binding.ProductionBindingError(
            "Mem0 persisted-memory count boundary changed"
        )
    result = count(
        collection_name=collection_name,
        count_filter=create_filter({"user_id": user_scope}),
        exact=True,
    )
    value = getattr(result, "count", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise binding.ProductionBindingError(
            "Mem0 persisted-memory count is invalid"
        )
    return value


def suspend_resumable_adapter(adapter: Any) -> Mapping[str, Any]:
    """Close every local handle without delete_col or owned-state erasure."""

    owned = getattr(adapter, "_backend", None)
    if not isinstance(owned, _OwnedMem0Backend):
        raise binding.ProductionBindingError(
            "resumable suspension requires the historical owned backend"
        )
    memory = getattr(owned, "backend", None)
    if memory is None:
        raise binding.ProductionBindingError("resumable Memory is absent")
    stores = binding._materialize_exact_qdrant_stores(memory)
    factory = getattr(adapter, "_resumable_factory", None)
    user_scope = getattr(adapter, "active_user_scope", None) or getattr(
        factory, "_scope", None
    )
    persisted_memory_count = _namespace_persisted_memory_count(
        memory, user_scope
    )
    clients: list[Any] = []
    locals_: list[Any] = []
    collections: list[Any] = []
    for store in stores:
        client = getattr(store, "client", None)
        if client is None:
            raise binding.ProductionBindingError("Qdrant store omitted client")
        if all(client is not row for row in clients):
            clients.append(client)
        local = getattr(client, "_client", None)
        if local is None:
            raise binding.ProductionBindingError("Qdrant client is not local")
        if all(local is not row for row in locals_):
            locals_.append(local)
    for local in locals_:
        registry = getattr(local, "collections", None)
        if not isinstance(registry, Mapping) or not registry:
            raise binding.ProductionBindingError(
                "local Qdrant registry is empty before suspension"
            )
        for collection in registry.values():
            if all(collection is not row for row in collections):
                collections.append(collection)
    errors: list[BaseException] = []
    for collection in collections:
        close = getattr(collection, "close", None)
        if not callable(close):
            errors.append(
                binding.ProductionBindingError(
                    "local Qdrant collection omitted close"
                )
            )
            continue
        try:
            close()
        except BaseException as exc:
            errors.append(exc)
    close_memory = getattr(memory, "close", None)
    if not callable(close_memory):
        errors.append(binding.ProductionBindingError("Memory omitted close"))
    else:
        try:
            close_memory()
        except BaseException as exc:
            errors.append(exc)
    for client in clients:
        close = getattr(client, "close", None)
        if not callable(close):
            errors.append(binding.ProductionBindingError("Qdrant client omitted close"))
            continue
        try:
            close()
        except BaseException as exc:
            errors.append(exc)
    transport = getattr(adapter, "_production_extraction_transport", None)
    close_transport = getattr(transport, "close", None)
    if transport is None or not callable(close_transport):
        errors.append(
            binding.ProductionBindingError(
                "resumable adapter omitted its extraction transport"
            )
        )
    else:
        try:
            close_transport()
        except BaseException as exc:
            errors.append(exc)
    if errors:
        if len(errors) == 1:
            raise errors[0]
        raise BaseExceptionGroup("resumable Mem0 suspension failed", errors)
    history = getattr(memory, "db", None)
    if history is not None:
        raise binding.ProductionBindingError(
            "resumable suspension retained Memory history manager"
        )
    if any(getattr(local, "closed", None) is not True for local in locals_):
        raise binding.ProductionBindingError(
            "resumable suspension retained a Qdrant local client"
        )
    if getattr(transport, "_closed", None) is not True:
        raise binding.ProductionBindingError(
            "resumable suspension retained its extraction transport"
        )
    adapter._closed = True
    closure_reader = getattr(factory, "transport_closure_receipt", None)
    if not callable(closure_reader):
        raise binding.ProductionBindingError(
            "resumable adapter omitted its transport-closure authority"
        )
    transport_closure = dict(closure_reader())
    receipt_before = {
        "format": RESUMABLE_SUSPEND_FORMAT,
        "history_sqlite_closed": True,
        "qdrant_local_collections_closed": len(collections),
        "qdrant_clients_closed": len(clients),
        "qdrant_local_registries_closed": len(locals_),
        "transport_closed": True,
        "transport_closure": transport_closure,
        "transport_closure_sha256": transport_closure["receipt_sha256"],
        "delete_col_calls": 0,
        "owned_state_retained": owned.state_root.is_dir(),
        "owned_state_tree": state_tree_receipt(owned.state_root),
        "namespace_persisted_memory_count": persisted_memory_count,
    }
    if receipt_before["owned_state_retained"] is not True:
        raise binding.ProductionBindingError(
            "resumable suspension erased owned state"
        )
    return MappingProxyType(
        {
            **receipt_before,
            "receipt_sha256": canonical_json_sha256(receipt_before),
        }
    )


def run_resumable_factory_provider_free_gate(
    *, owned_state_dir: str | os.PathLike[str]
) -> Mapping[str, Any]:
    """Construct, suspend, and erase the real zero-call resumable factory.

    The gate exercises both local Qdrant stores, BGE-M3, BM25, history SQLite,
    and the source-layout private seams while sockets are denied.  It cannot
    authorize an extraction call and its state is removed only after the
    non-destructive suspension receipt proves every handle closed.
    """

    state = binding._factory_canary_state(owned_state_dir)
    record = binding._factory_canary_record()
    authorization = binding._factory_canary_authorization(record)
    authorization_sha256 = canonical_json_sha256(asdict(authorization))
    scope = f"longmemeval:resumable:{authorization_sha256[:32]}"
    factory = ResumableExactMem0AdapterFactory(
        authorization,
        segment_authorized_calls=0,
        adopt_existing_state=False,
        user_scope=scope,
        before_http_send=None,
    )
    adapter: Any | None = None
    suspended: Mapping[str, Any] | None = None
    operation_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    with binding._blocked_network_probe() as network_attempts:
        try:
            adapter = factory(state)
            suspended = suspend_resumable_adapter(adapter)
        except BaseException as exc:
            operation_error = exc
        finally:
            if adapter is not None and suspended is None:
                try:
                    adapter.cleanup()
                except BaseException as exc:
                    cleanup_errors.append(exc)
    if operation_error is not None:
        for exc in cleanup_errors:
            operation_error.add_note(
                f"resumable provider-free cleanup: {type(exc).__name__}: {exc}"
            )
        raise operation_error
    if cleanup_errors:
        raise BaseExceptionGroup(
            "resumable provider-free gate cleanup failed", cleanup_errors
        )
    assert suspended is not None
    if network_attempts:
        raise binding.ProductionBindingError(
            "resumable provider-free factory attempted network access"
        )
    factory_receipt = dict(factory.binding_receipt())
    transport = factory_receipt.get("transport")
    if not isinstance(transport, Mapping) or any(
        transport.get(field) != 0
        for field in ("authorized", "attempted", "completed", "failed", "rejected")
    ):
        raise binding.ProductionBindingError(
            "resumable provider-free transport receipt did not close at zero"
        )
    marker = state / OWNERSHIP_MARKER
    if not marker.is_file():
        raise binding.ProductionBindingError(
            "resumable provider-free state lost its ownership marker"
        )
    token = marker.read_text(encoding="utf-8").strip()
    _remove_owned_state(state, token)
    if state.exists():
        raise binding.ProductionBindingError(
            "resumable provider-free state remained after verified removal"
        )
    body = {
        "format": RESUMABLE_PROVIDER_FREE_GATE_FORMAT,
        "campaign_authority": False,
        "source_layout": SOURCE_LAYOUT,
        "authorization_sha256": authorization_sha256,
        "provider_calls_authorized": 0,
        "network_attempts": 0,
        "owned_state_removed": True,
        "factory_receipt": factory_receipt,
        "suspend_receipt": dict(suspended),
        "environment_lock_sha256": authorization.mem0_environment_lock_sha256,
        "tool_implementation_sha256": (
            authorization.mem0_tool_implementation_sha256
        ),
    }
    return MappingProxyType(
        {**body, "receipt_sha256": canonical_json_sha256(body)}
    )


__all__ = [
    "RESUMABLE_FACTORY_FORMAT",
    "RESUMABLE_PROVIDER_FREE_GATE_FORMAT",
    "RESUMABLE_SUSPEND_FORMAT",
    "RESUMABLE_TRANSPORT_CLOSURE_FORMAT",
    "RESUMABLE_WRITE_ACTIVITY_FORMAT",
    "Mem0WriteActivityMeter",
    "ResumableExactMem0AdapterFactory",
    "ZeroCallExtractionTransport",
    "run_resumable_factory_provider_free_gate",
    "suspend_resumable_adapter",
]
