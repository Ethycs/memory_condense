"""Concrete, fail-closed production-binding seam for the Mem0 campaign.

Importing this module installs no package, loads no model, opens no socket, and
makes no provider call.  The concrete extraction transport and Mem0 adapter
factory are code-owned and lazy: sockets and optional runtime imports occur only
when the exact factory or transport is explicitly constructed.  The responder
and judge model names are frozen, but their concrete single-attempt transports
are not yet implemented in this repository, so positive end-to-end production
binding issuance remains closed.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import ssl
import socket
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence

from .source_compat import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    count_tokens,
    verify_bge_m3_checkpoint,
)

from .preflight import tool_implementation_sha256
from .policy import (
    MEM0_EXTRACTION_GATEWAY_URL,
    MEM0_EXTRACTION_HTTPX_VERSION,
    MEM0_EXTRACTION_MODEL,
    MEM0_EXTRACTION_OPENAI_VERSION,
    MEM0_EXTRACTION_PROVIDER,
    MEM0_EXTRACTION_RESPONSE_MODEL,
    MEM0_EXTRACTION_REVISION,
    MEM0_EXTRACTION_ROUTE_IDENTITY,
    MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
    MEM0_EXTRACTION_TRUSTSTORE_VERSION,
    canonical_json_sha256,
)

if TYPE_CHECKING:
    from .run_shard import (
        ProviderCallResult,
        RetrievalStageAuthorization,
        ScoringStageAuthorization,
    )


PRODUCTION_BINDING_FORMAT = "memory-condense-mem0-production-binding-v1"
PRODUCTION_BINDING_KIND = "frozen_concrete_production_launcher_v1"
LOCAL_BGE_PROBE_FORMAT = "memory-condense-local-bge-m3-runtime-probe-v1"
PROVIDER_FREE_STATUS_FORMAT = "memory-condense-mem0-production-readiness-v2"

# These are code-owned slots, not configuration inputs.  Policy JSON or an
# injected object can never populate them.  Concrete extraction types are
# assigned below their final class definitions; scoring stays closed.
_EXACT_EXTRACTION_TRANSPORT_TYPE: type[Any] | None = None
_EXACT_MEM0_ADAPTER_FACTORY_TYPE: type[Any] | None = None
_EXACT_RESPONDER_TRANSPORT_TYPE: type[Any] | None = None
_EXACT_JUDGE_TRANSPORT_TYPE: type[Any] | None = None

_MEM0_EXTRACTION_API_KEY_ENV = "LITELLM_KEY"
_MEM0_EXTRACTION_MAX_COMPLETION_TOKENS = 2_000
_MEM0_EXTRACTION_TIMEOUT_SECONDS = 600.0
_MEM0_CONSTRUCTOR_ONLY_API_KEY = "mem0-constructor-only-no-network"
_OPENAI_CONSTRUCTION_ENV_LOCK = threading.Lock()
_MEM0_STACK_DEPENDENCY_VERSIONS = MappingProxyType(
    {
        "mem0ai": "2.0.18",
        "qdrant-client": "1.15.1",
        "fastembed": "0.7.3",
        "spacy": "3.8.7",
        "en-core-web-sm": "3.8.0",
    }
)
_MEM0_SPACY_TRANSITIVE_VERSIONS = MappingProxyType({"click": "8.4.2"})
_FASTEMBED_BM25_REVISION = "22b8d2af71a76161e18dd432d2cee0eefa66e412"
_FASTEMBED_BM25_ASSET_SHA256 = (
    "6776662741625645eeeee8ca293f5ce650d194bcdd3f2f8a46675da4cd48e0f1"
)
_MEM0_EXTRACTION_REQUEST_IDENTITY = MappingProxyType(
    {
        "format": "memory-condense-mem0-extraction-request-v1",
        "route_identity_sha256": MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
        "response_format": {"type": "json_object"},
        "max_completion_tokens": _MEM0_EXTRACTION_MAX_COMPLETION_TOKENS,
        "sampling_parameters": "omitted",
        "sdk_retries": 0,
        "http_transport_retries": 0,
        "follow_redirects": False,
        "trust_env": False,
        "timeout_seconds": _MEM0_EXTRACTION_TIMEOUT_SECONDS,
        "connect_timeout_seconds": 30.0,
        "cap_boundary": "httpx.BaseTransport.handle_request",
    }
)
_MEM0_EXTRACTION_REQUEST_IDENTITY_SHA256 = canonical_json_sha256(
    dict(_MEM0_EXTRACTION_REQUEST_IDENTITY)
)

_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "LITELLM_LOCAL_MODEL_COST_MAP": "true",
    "MEM0_TELEMETRY": "false",
}
# Local-only probes compose (the factory gate contains the BGE/FastEmbed
# probes), so the process-global socket guard must be re-entrant on one thread.
_NETWORK_PROBE_LOCK = threading.RLock()
_FORBIDDEN_SECRET_KEYS = {
    "api_key",
    "authorization",
    "auth",
    "client_key",
    "client_secret",
    "connection_string",
    "cookie",
    "credentials",
    "password",
    "private_key",
    "proxy_authorization",
    "refresh_token",
    "sas_token",
    "secret",
    "secret_key",
    "set_cookie",
    "signing_key",
    "token",
    "access_token",
    "x_api_key",
}
_SECRET_KEY_SUFFIXES = (
    "_api_key",
    "_credential",
    "_credentials",
    "_password",
    "_private_key",
    "_secret",
    "_token",
)
_SECRET_VALUE_RE = re.compile(
    r"(?:^\s*(?:bearer|basic)\s+\S+|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\b(?:sk|ghp|github_pat|xox[baprs]|AIza)[-_][A-Za-z0-9_-]{8,}|"
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})",
    re.IGNORECASE,
)


class ProductionBindingError(RuntimeError):
    """A claimed production boundary was incomplete or changed."""


class ProductionBindingBlocked(ProductionBindingError):
    """A required concrete provider transport has not been frozen."""


class TransportAttemptLimitExceeded(ProductionBindingError):
    """A local send-boundary cap rejected an extra attempt before dispatch."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[str, bytes]:
    payload = path.read_bytes()
    return _sha256_bytes(payload), payload


def _require_exact_authorization(
    value: RetrievalStageAuthorization | ScoringStageAuthorization,
    *,
    stage: str,
) -> None:
    # Keep the standalone Terra canary independent of the full benchmark
    # runtime.  The authorization contracts are imported only when a launcher
    # or adapter factory actually consumes one.
    from .run_shard import RetrievalStageAuthorization, ScoringStageAuthorization

    expected = (
        RetrievalStageAuthorization
        if stage == "retrieval"
        else ScoringStageAuthorization
    )
    if type(value) is not expected:
        raise ProductionBindingError(
            f"{stage} production binding requires exact {expected.__name__}"
        )


def _authorization_sha256(
    value: RetrievalStageAuthorization | ScoringStageAuthorization,
) -> str:
    return canonical_json_sha256(asdict(value))


def _plain_json(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProductionBindingError(f"{label} is not strict JSON") from exc


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionBindingError(f"{label} must be a mapping")
    result = _plain_json(dict(value), label=label)
    if not isinstance(result, dict):  # pragma: no cover - guarded above
        raise ProductionBindingError(f"{label} must be a JSON object")
    _reject_secret_material(result, label=label)
    return result


def _reject_secret_material(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).casefold().replace("-", "_")
            if (
                key in _FORBIDDEN_SECRET_KEYS
                or key.endswith(_SECRET_KEY_SUFFIXES)
            ) and item not in (None, "<redacted>"):
                raise ProductionBindingError(
                    f"{label} contains forbidden secret field {raw_key!r}"
                )
            _reject_secret_material(item, label=f"{label}.{raw_key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secret_material(item, label=f"{label}[{index}]")
        return
    if isinstance(value, str) and value != "<redacted>" and _SECRET_VALUE_RE.search(value):
        raise ProductionBindingError(
            f"{label} contains credential-shaped secret material"
        )


def _must_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ProductionBindingError(
            f"{label} mismatch: observed={observed!r}, expected={expected!r}"
        )


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ProductionBindingError(
            f"{label} fields mismatch: missing={sorted(expected - set(value))!r}, "
            f"extra={sorted(set(value) - expected)!r}"
        )


def _owned_state_path(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ProductionBindingError(f"{label} must be an owned-state placeholder")
    prefix = "<owned_state>/"
    if not value.startswith(prefix):
        raise ProductionBindingError(f"{label} must begin with {prefix!r}")
    relative = value[len(prefix) :]
    if (
        not relative
        or "\\" in relative
        or relative.startswith("/")
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ProductionBindingError(f"{label} contains unsafe path traversal")
    normalized = PurePosixPath(relative)
    if normalized.is_absolute() or normalized.as_posix() != relative:
        raise ProductionBindingError(f"{label} is not a normalized relative path")
    return value


def validate_production_mem0_config(
    authorization: RetrievalStageAuthorization | ScoringStageAuthorization,
) -> dict[str, Any]:
    """Validate the complete redacted Mem0 config before any constructor runs.

    The extraction route is code-owned, and the only admissible redacted LLM
    config field is its exact gateway model identifier.  A future provider
    integration must change this allowlist deliberately; arbitrary policy keys
    such as retries, temperature, base URLs, or proxy settings are never
    inherited into production trust.
    """

    extraction = _mapping(
        authorization.extraction_model_identity,
        "extraction_model_identity",
    )
    _exact_keys(
        extraction,
        {
            "provider",
            "model",
            "revision",
            "model_identity_sha256",
            "provider_retries",
            "logical_call_boundary",
            "logical_calls_per_add",
            "http_attempts_certified",
        },
        "extraction_model_identity",
    )
    extraction_body = dict(extraction)
    extraction_sha = extraction_body.pop("model_identity_sha256")
    _must_equal(
        extraction_sha,
        canonical_json_sha256(extraction_body),
        "extraction internal identity SHA-256",
    )
    _must_equal(
        canonical_json_sha256(extraction),
        authorization.extraction_model_identity_sha256,
        "extraction authorization SHA-256",
    )
    for field, expected in {
        "provider_retries": 0,
        "logical_call_boundary": "Memory.llm.generate_response",
        "logical_calls_per_add": 1,
        "http_attempts_certified": False,
    }.items():
        _must_equal(extraction.get(field), expected, f"extraction identity {field}")
    for field in ("provider", "model", "revision"):
        if not isinstance(extraction.get(field), str) or not extraction[field].strip():
            raise ProductionBindingError(f"extraction identity {field} is empty")
    for field, expected in {
        "provider": MEM0_EXTRACTION_PROVIDER,
        "model": MEM0_EXTRACTION_MODEL,
        "revision": MEM0_EXTRACTION_REVISION,
    }.items():
        _must_equal(
            extraction.get(field),
            expected,
            f"frozen extraction identity {field}",
        )

    stable = _mapping(authorization.mem0_stable_payload, "mem0_stable_payload")
    _exact_keys(stable, {"protocol", "config", "stack"}, "mem0_stable_payload")
    _must_equal(
        stable.get("protocol"),
        "mem0-oss-2.0.18-certified-local-v1",
        "Mem0 runtime protocol",
    )
    _must_equal(
        canonical_json_sha256(stable),
        authorization.mem0_stable_config_sha256,
        "Mem0 stable payload SHA-256",
    )
    config = _mapping(stable.get("config"), "Mem0 stable config")
    _exact_keys(
        config,
        {
            "version",
            "custom_instructions",
            "reranker",
            "llm",
            "embedder",
            "vector_store",
            "history_db_path",
        },
        "Mem0 stable config",
    )
    for field, expected in {
        "version": "v1.1",
        "custom_instructions": None,
        "reranker": None,
    }.items():
        _must_equal(config.get(field), expected, f"Mem0 config {field}")

    llm = _mapping(config.get("llm"), "Mem0 LLM config")
    _exact_keys(llm, {"provider", "config"}, "Mem0 LLM config")
    _must_equal(llm.get("provider"), extraction["provider"], "Mem0 LLM provider")
    llm_config = _mapping(llm.get("config"), "Mem0 LLM config.config")
    _exact_keys(llm_config, {"model"}, "Mem0 LLM config.config")
    _must_equal(llm_config.get("model"), extraction["model"], "Mem0 LLM model")

    vector = _mapping(config.get("vector_store"), "Mem0 vector-store config")
    _exact_keys(vector, {"provider", "config"}, "Mem0 vector-store config")
    _must_equal(vector.get("provider"), "qdrant", "Mem0 vector-store provider")
    vector_config = _mapping(
        vector.get("config"), "Mem0 vector-store config.config"
    )
    _exact_keys(
        vector_config,
        {
            "embedding_model_dims",
            "collection_name",
            "path",
            "on_disk",
        },
        "Mem0 vector-store config.config",
    )
    _must_equal(
        vector_config.get("embedding_model_dims"),
        DEFAULT_MODEL_DIM,
        "Mem0 vector-store embedding dimension",
    )
    _must_equal(vector_config.get("on_disk"), True, "Mem0 Qdrant on_disk")
    _must_equal(
        vector_config.get("collection_name"),
        "longmemeval",
        "Mem0 collection_name",
    )
    _owned_state_path(vector_config.get("path"), "Mem0 Qdrant path")
    _must_equal(
        vector_config.get("path"),
        "<owned_state>/qdrant",
        "Mem0 Qdrant path",
    )
    _owned_state_path(config.get("history_db_path"), "Mem0 history path")
    _must_equal(
        config.get("history_db_path"),
        "<owned_state>/history.sqlite",
        "Mem0 history path",
    )

    stack = _mapping(stable.get("stack"), "Mem0 runtime stack")
    _exact_keys(
        stack,
        {
            "dependency_versions",
            "bm25_model",
            "spacy_model",
            "bm25_operational",
            "entity_extraction_operational",
        },
        "Mem0 runtime stack",
    )
    _must_equal(
        _mapping(stack.get("dependency_versions"), "Mem0 dependency versions"),
        {
            "mem0ai": "2.0.18",
            "qdrant-client": "1.15.1",
            "fastembed": "0.7.3",
            "spacy": "3.8.7",
            "en-core-web-sm": "3.8.0",
        },
        "Mem0 dependency versions",
    )
    for field, expected in {
        "bm25_model": "Qdrant/bm25",
        "spacy_model": "en_core_web_sm",
        "bm25_operational": True,
        "entity_extraction_operational": True,
    }.items():
        _must_equal(stack.get(field), expected, f"Mem0 runtime stack {field}")

    return {
        "format": "memory-condense-mem0-production-config-v1",
        "stable_config_sha256": authorization.mem0_stable_config_sha256,
        "extraction_model_identity_sha256": (
            authorization.extraction_model_identity_sha256
        ),
        "llm_config_fields": ["model"],
        "provider_retries": 0,
        "qdrant_remote_fields_disabled": True,
        "owned_state_paths_normalized": True,
    }


@dataclass(frozen=True, slots=True)
class FrozenArtifactBindingReceipt:
    """Direct byte identities rechecked by the concrete launcher."""

    policy_path: str
    policy_sha256: str
    mem0_environment_lock_path: str
    mem0_environment_lock_sha256: str
    tool_root: str
    tool_implementation_sha256: str
    source_environment_lock_path: str | None = None
    source_environment_lock_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_frozen_artifact_binding(
    authorization: RetrievalStageAuthorization | ScoringStageAuthorization,
    *,
    stage: str,
    policy_path: str | os.PathLike[str],
    mem0_environment_lock_path: str | os.PathLike[str],
    tool_root: str | os.PathLike[str],
    source_environment_lock_path: str | os.PathLike[str] | None = None,
) -> FrozenArtifactBindingReceipt:
    """Hash policy, lock, and tool bytes twice without trusting a caller receipt."""

    if stage not in {"retrieval", "scoring"}:
        raise ProductionBindingError("artifact binding stage is invalid")
    _require_exact_authorization(authorization, stage=stage)
    policy = Path(policy_path).resolve(strict=True)
    mem0_lock = Path(mem0_environment_lock_path).resolve(strict=True)
    tools = Path(tool_root).resolve(strict=True)
    if not policy.is_file() or not mem0_lock.is_file() or not tools.is_dir():
        raise ProductionBindingError("policy, lock, and tool paths have wrong kinds")

    policy_sha, policy_bytes = _sha256_file(policy)
    lock_sha, lock_bytes = _sha256_file(mem0_lock)
    tool_sha = tool_implementation_sha256(tools)
    for observed, expected, label in (
        (policy_sha, authorization.mem0_policy_sha256, "Mem0 policy SHA-256"),
        (
            lock_sha,
            authorization.mem0_environment_lock_sha256,
            "Mem0 environment-lock SHA-256",
        ),
        (
            tool_sha,
            authorization.mem0_tool_implementation_sha256,
            "Mem0 tool implementation SHA-256",
        ),
    ):
        _must_equal(observed, expected, label)

    source_path: Path | None = None
    source_sha: str | None = None
    source_bytes: bytes | None = None
    if stage == "scoring":
        if source_environment_lock_path is None:
            raise ProductionBindingError(
                "scoring production binding requires the frozen source lock path"
            )
        source_path = Path(source_environment_lock_path).resolve(strict=True)
        if not source_path.is_file():
            raise ProductionBindingError("source environment lock is not a file")
        source_sha, source_bytes = _sha256_file(source_path)
        _must_equal(
            source_sha,
            authorization.source_environment_lock_sha256,
            "source environment-lock SHA-256",
        )
    elif source_environment_lock_path is not None:
        raise ProductionBindingError(
            "retrieval binding must not accept an unrelated source-lock path"
        )

    # Repeat every external read after the tree walk.  A replacement while the
    # launcher is validating must not escape with a stale positive receipt.
    if policy.read_bytes() != policy_bytes:
        raise ProductionBindingError("Mem0 policy bytes changed during preflight")
    if mem0_lock.read_bytes() != lock_bytes:
        raise ProductionBindingError("Mem0 lock bytes changed during preflight")
    if tool_implementation_sha256(tools) != tool_sha:
        raise ProductionBindingError("Mem0 tool bytes changed during preflight")
    if source_path is not None and source_path.read_bytes() != source_bytes:
        raise ProductionBindingError("source lock bytes changed during preflight")

    return FrozenArtifactBindingReceipt(
        policy_path=policy.as_posix(),
        policy_sha256=policy_sha,
        mem0_environment_lock_path=mem0_lock.as_posix(),
        mem0_environment_lock_sha256=lock_sha,
        tool_root=tools.as_posix(),
        tool_implementation_sha256=tool_sha,
        source_environment_lock_path=(
            source_path.as_posix() if source_path is not None else None
        ),
        source_environment_lock_sha256=source_sha,
    )


def validate_local_bge_m3_contract(
    authorization: RetrievalStageAuthorization | ScoringStageAuthorization,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Independently validate the exact local/offline embedder configuration."""

    validate_production_mem0_config(authorization)
    identity = _mapping(
        authorization.embedder_model_identity,
        "embedder_model_identity",
    )
    exact_fields = {
        "provider",
        "model",
        "revision",
        "model_identity_sha256",
        "checkpoint_sha256",
        "dimension",
        "device",
        "dtype",
        "execution",
        "network_calls_authorized",
        "runtime_probe_required",
    }
    if set(identity) != exact_fields:
        raise ProductionBindingError("embedder identity fields mismatch")
    body = dict(identity)
    supplied_model_sha = body.pop("model_identity_sha256")
    _must_equal(
        supplied_model_sha,
        canonical_json_sha256(body),
        "embedder internal identity SHA-256",
    )
    _must_equal(
        canonical_json_sha256(identity),
        authorization.embedder_model_identity_sha256,
        "embedder authorization SHA-256",
    )
    for field, expected in {
        "provider": "huggingface",
        "model": DEFAULT_MODEL_NAME,
        "revision": DEFAULT_MODEL_REVISION,
        "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
        "dimension": DEFAULT_MODEL_DIM,
        "dtype": "float32",
        "execution": "local_offline",
        "network_calls_authorized": 0,
        "runtime_probe_required": True,
    }.items():
        _must_equal(identity.get(field), expected, f"embedder identity {field}")
    if identity.get("device") not in {"cpu", "cuda"}:
        raise ProductionBindingError("embedder device must resolve to cpu or cuda")

    stable = _mapping(authorization.mem0_stable_payload, "mem0_stable_payload")
    _must_equal(
        canonical_json_sha256(stable),
        authorization.mem0_stable_config_sha256,
        "Mem0 stable payload SHA-256",
    )
    config = _mapping(stable.get("config"), "Mem0 stable config")
    embedder = _mapping(config.get("embedder"), "Mem0 embedder config")
    if set(embedder) != {"provider", "config"}:
        raise ProductionBindingError("Mem0 embedder config fields mismatch")
    _must_equal(embedder.get("provider"), "huggingface", "embedder provider")
    embedded = _mapping(embedder.get("config"), "Mem0 embedder config.config")
    expected_embedded = {
        "model": DEFAULT_MODEL_NAME,
        "embedding_dims": DEFAULT_MODEL_DIM,
        "huggingface_base_url": None,
        "model_kwargs": {
            "revision": DEFAULT_MODEL_REVISION,
            "local_files_only": True,
            "trust_remote_code": False,
            "device": identity["device"],
        },
    }
    _must_equal(embedded, expected_embedded, "Mem0 local BGE-M3 config")
    vector = _mapping(config.get("vector_store"), "Mem0 vector-store config")
    vector_config = _mapping(
        vector.get("config"), "Mem0 vector-store config.config"
    )
    _must_equal(
        vector_config.get("embedding_model_dims"),
        DEFAULT_MODEL_DIM,
        "vector-store embedding dimension",
    )

    observed_environment = os.environ if environment is None else environment
    for name, expected in _OFFLINE_ENVIRONMENT.items():
        observed = observed_environment.get(name)
        if name in {"MEM0_TELEMETRY", "LITELLM_LOCAL_MODEL_COST_MAP"}:
            observed = observed.casefold() if isinstance(observed, str) else observed
        _must_equal(observed, expected, f"offline environment {name}")

    return {
        "format": "memory-condense-local-bge-m3-contract-v1",
        "model": identity["model"],
        "revision": identity["revision"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "dimension": identity["dimension"],
        "device": identity["device"],
        "dtype": identity["dtype"],
        "local_files_only": True,
        "trust_remote_code": False,
        "huggingface_base_url": None,
        "network_calls_authorized": 0,
        "offline_environment_sha256": canonical_json_sha256(
            dict(sorted(_OFFLINE_ENVIRONMENT.items()))
        ),
    }


@contextmanager
def _blocked_network_probe() -> Iterator[list[str]]:
    """Reject and count socket attempts while the local model is probed."""

    attempts: list[str] = []
    with _NETWORK_PROBE_LOCK:
        original_create = socket.create_connection
        original_connect = socket.socket.connect
        original_connect_ex = socket.socket.connect_ex

        def blocked(*args: Any, **_kwargs: Any) -> Any:
            target = args[-1] if args else "unknown"
            attempts.append(repr(target))
            raise ProductionBindingError(
                "local model probe attempted forbidden network access"
            )

        socket.create_connection = blocked  # type: ignore[assignment]
        socket.socket.connect = blocked  # type: ignore[method-assign]
        socket.socket.connect_ex = blocked  # type: ignore[method-assign]
        try:
            yield attempts
        finally:
            socket.create_connection = original_create  # type: ignore[assignment]
            socket.socket.connect = original_connect  # type: ignore[method-assign]
            socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]


def probe_local_bge_m3_runtime(
    authorization: RetrievalStageAuthorization,
    *,
    model_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Hash and execute the exact local checkpoint under a socket-deny guard.

    The function has no injectable loader or verifier seam.  Tests can exercise
    its pure contract validator without materializing the 2.3 GB checkpoint;
    a production launcher must execute this concrete function successfully.
    """

    _require_exact_authorization(authorization, stage="retrieval")
    contract = validate_local_bge_m3_contract(authorization)
    root = Path(model_dir).resolve(strict=True)
    if not root.is_dir():
        raise ProductionBindingError("BGE-M3 model_dir must be a directory")
    before_sha = verify_bge_m3_checkpoint(root)
    _must_equal(before_sha, BGE_M3_CHECKPOINT_SHA256, "BGE-M3 checkpoint")

    model: Any | None = None
    vector_sha256 = ""
    observed_dim = 0
    observed_device = ""
    observed_dtype = ""
    attempts: list[str] = []
    try:
        from sentence_transformers import SentenceTransformer

        parameters = inspect.signature(SentenceTransformer.__init__).parameters
        for required in ("device", "local_files_only", "trust_remote_code"):
            if required not in parameters:
                raise ProductionBindingError(
                    f"SentenceTransformer lacks required offline parameter {required}"
                )
        with _blocked_network_probe() as attempts:
            model = SentenceTransformer(
                str(root),
                device=contract["device"],
                local_files_only=True,
                trust_remote_code=False,
            )
            dimension_reader = getattr(
                model, "get_sentence_embedding_dimension", None
            ) or getattr(model, "get_embedding_dimension", None)
            if not callable(dimension_reader):
                raise ProductionBindingError(
                    "loaded BGE-M3 omitted an embedding-dimension probe"
                )
            observed_dim = int(dimension_reader())
            observed_device = str(getattr(model, "device", "")).casefold()
            vectors = model.encode(
                [
                    "memory-condense local embedding probe alpha",
                    "memory-condense local embedding probe beta",
                ],
                normalize_embeddings=False,
                convert_to_numpy=True,
            )
            shape = tuple(int(value) for value in getattr(vectors, "shape", ()))
            if shape != (2, DEFAULT_MODEL_DIM):
                raise ProductionBindingError(
                    f"BGE-M3 runtime output shape mismatch: {shape!r}"
                )
            observed_dtype = str(getattr(vectors, "dtype", "")).casefold()
            vector_sha256 = hashlib.sha256(vectors.tobytes(order="C")).hexdigest()
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()
        if model is not None:
            del model
        gc.collect()

    _must_equal(len(attempts), 0, "BGE-M3 runtime network attempts")
    _must_equal(observed_dim, DEFAULT_MODEL_DIM, "BGE-M3 runtime dimension")
    actual_device = observed_device.split(":", 1)[0]
    _must_equal(actual_device, contract["device"], "BGE-M3 runtime device")
    _must_equal(observed_dtype, "float32", "BGE-M3 runtime output dtype")
    after_sha = verify_bge_m3_checkpoint(root)
    _must_equal(after_sha, before_sha, "BGE-M3 checkpoint post-probe")

    body = {
        "format": LOCAL_BGE_PROBE_FORMAT,
        "model": DEFAULT_MODEL_NAME,
        "revision": DEFAULT_MODEL_REVISION,
        "checkpoint_sha256_before": before_sha,
        "checkpoint_sha256_after": after_sha,
        "checkpoint_unchanged": True,
        "dimension": observed_dim,
        "device": actual_device,
        "output_dtype": observed_dtype,
        "probe_vectors_sha256": vector_sha256,
        "local_files_only": True,
        "trust_remote_code": False,
        "network_attempts": 0,
        "network_calls_authorized": 0,
    }
    return {**body, "receipt_sha256": canonical_json_sha256(body)}


class HardTransportAttemptCap:
    """Thread-safe, one-dispatch-per-claim cap for an exact send boundary.

    This helper is usable with injected test delegates, but that use is always
    labelled nonproduction and cannot issue ``TrustedRuntimeBinding``.  A future
    concrete provider transport must place ``call`` at its actual HTTP send
    boundary; merely wrapping an SDK call cannot certify hidden SDK retries.
    """

    __slots__ = (
        "_authorized",
        "_completed",
        "_failed",
        "_lock",
        "_rejected",
        "_role",
        "_attempted",
    )

    def __init__(self, *, role: str, authorized: int) -> None:
        if role not in {"extraction", "responder", "judge"}:
            raise ValueError("transport cap role is invalid")
        if isinstance(authorized, bool) or not isinstance(authorized, int) or authorized < 1:
            raise ValueError("transport cap authorization must be a positive int")
        self._role = role
        self._authorized = authorized
        self._attempted = 0
        self._completed = 0
        self._failed = 0
        self._rejected = 0
        # The durable pre-send hook runs while the cap is held so eligibility,
        # journal publication, and counter advancement are one ordered
        # boundary.  A re-entrant lock lets the hook inspect the cap receipt
        # for an audit assertion without deadlocking that boundary.
        self._lock = threading.RLock()

    def call(
        self,
        send_once: Callable[..., Any],
        *args: Any,
        _before_increment: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if not callable(send_once):
            raise TypeError("transport send boundary must be callable")
        with self._lock:
            if self._attempted >= self._authorized:
                self._rejected += 1
                raise TransportAttemptLimitExceeded(
                    f"{self._role} transport attempt authorization exhausted"
                )
            if _before_increment is not None:
                if not callable(_before_increment):
                    raise TypeError("transport pre-send boundary must be callable")
                # This callback is the durable send marker for resumable runs.
                # It executes after cap eligibility, but before the counter can
                # advance or the inner HTTP transport can see the request.
                _before_increment()
            self._attempted += 1
        try:
            result = send_once(*args, **kwargs)
        except BaseException:
            with self._lock:
                self._failed += 1
            raise
        with self._lock:
            self._completed += 1
        return result

    def assert_closed(self) -> None:
        receipt = self.receipt()
        if (
            receipt["attempted"] != receipt["authorized"]
            or receipt["completed"] != receipt["authorized"]
            or receipt["failed"]
            or receipt["rejected"]
        ):
            raise ProductionBindingError(
                f"{self._role} transport attempt accounting did not close exactly"
            )

    def receipt(self) -> dict[str, Any]:
        with self._lock:
            return {
                "kind": "local_transport_send_cap",
                "role": self._role,
                "authorized": self._authorized,
                "attempted": self._attempted,
                "completed": self._completed,
                "failed": self._failed,
                "rejected": self._rejected,
                "retries_authorized": 0,
            }


def _require_exact_extraction_transport_dependencies() -> None:
    expected = {
        "openai": MEM0_EXTRACTION_OPENAI_VERSION,
        "httpx": MEM0_EXTRACTION_HTTPX_VERSION,
        "truststore": MEM0_EXTRACTION_TRUSTSTORE_VERSION,
    }
    for distribution, wanted in expected.items():
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ProductionBindingBlocked(
                f"exact extraction transport requires {distribution}=={wanted}"
            ) from exc
        _must_equal(
            observed,
            wanted,
            f"extraction transport dependency {distribution}",
        )


def _litellm_api_key() -> str:
    value = os.environ.get(_MEM0_EXTRACTION_API_KEY_ENV, "")
    if not isinstance(value, str) or not value.strip():
        raise ProductionBindingBlocked(
            f"exact extraction transport requires {_MEM0_EXTRACTION_API_KEY_ENV}"
        )
    return value.strip()


class _HardCappedSyncHTTPTransport:
    """Duck-typed httpx transport with the cap at ``handle_request`` itself."""

    def __init__(
        self,
        inner: Any,
        cap: HardTransportAttemptCap,
        *,
        before_http_send: Callable[[Any], Any] | None = None,
    ) -> None:
        if not callable(getattr(inner, "handle_request", None)):
            raise TypeError("inner HTTP transport omitted handle_request")
        self._inner = inner
        self._cap = cap
        self._before_http_send = before_http_send
        self._closed = False

    def handle_request(self, request: Any) -> Any:
        if self._closed:
            raise ProductionBindingError("extraction HTTP transport is closed")
        callback = self._before_http_send
        return self._cap.call(
            self._inner.handle_request,
            request,
            _before_increment=(
                (lambda: callback(request)) if callback is not None else None
            ),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()


def _extraction_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    if not isinstance(messages, Sequence) or isinstance(
        messages, (str, bytes, bytearray)
    ):
        raise TypeError("Mem0 extraction messages must be a sequence")
    normalized: list[dict[str, str]] = []
    for index, value in enumerate(messages):
        if not isinstance(value, Mapping) or set(value) != {"role", "content"}:
            raise ProductionBindingError(
                f"Mem0 extraction message {index} fields changed"
            )
        role = value.get("role")
        content = value.get("content")
        if not isinstance(role, str) or not isinstance(content, str) or not content:
            raise ProductionBindingError(
                f"Mem0 extraction message {index} is not non-empty text"
            )
        normalized.append({"role": role, "content": content})
    if tuple(row["role"] for row in normalized) != ("system", "user"):
        raise ProductionBindingError(
            "Mem0 extraction requires the exact system/user prompt boundary"
        )
    return normalized


class LiteLLMTerraExtractionTransport:
    """Exact Mem0 extraction LLM for the controlled central-dev Terra route.

    The OpenAI SDK and httpx each have retries disabled.  The hard attempt cap
    wraps ``httpx``'s concrete ``handle_request`` boundary, below the SDK, so an
    SDK regression cannot silently spend a second authorized call.
    """

    production_eligible = True

    def __init_subclass__(cls, **_kwargs: Any) -> None:
        raise TypeError("LiteLLMTerraExtractionTransport cannot be subclassed")

    def __init__(
        self,
        *,
        authorized: int,
        _before_http_send: Callable[[Any], Any] | None = None,
    ) -> None:
        _require_exact_extraction_transport_dependencies()
        api_key = _litellm_api_key()

        import httpx
        import truststore
        from openai import OpenAI

        cap = HardTransportAttemptCap(role="extraction", authorized=authorized)
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        raw_transport = httpx.HTTPTransport(
            verify=context,
            retries=0,
        )
        capped_transport = _HardCappedSyncHTTPTransport(
            raw_transport,
            cap,
            before_http_send=_before_http_send,
        )
        http_client = httpx.Client(
            transport=capped_transport,
            follow_redirects=False,
            trust_env=False,
            timeout=httpx.Timeout(
                _MEM0_EXTRACTION_TIMEOUT_SECONDS,
                connect=30.0,
            ),
        )
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=MEM0_EXTRACTION_GATEWAY_URL,
                http_client=http_client,
                max_retries=0,
            )
        except BaseException:
            http_client.close()
            raise
        if getattr(client, "max_retries", None) != 0:
            client.close()
            raise ProductionBindingError("OpenAI extraction retries are not zero")
        if str(getattr(client, "base_url", "")).rstrip("/") != (
            MEM0_EXTRACTION_GATEWAY_URL.rstrip("/")
        ):
            client.close()
            raise ProductionBindingError("OpenAI extraction base URL changed")

        self._cap = cap
        self._client = client
        self._closed = False
        self._usage_lock = threading.Lock()
        self._provider_input_tokens = 0
        self._provider_output_tokens = 0
        self._provider_total_tokens = 0
        self._provider_usage_records = 0
        self._provider_latency_s = 0.0

    def generate_response(
        self,
        messages: Sequence[Mapping[str, str]],
        response_format: Mapping[str, str] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> str:
        if self._closed:
            raise ProductionBindingError("extraction transport is closed")
        normalized = _extraction_messages(messages)
        if tools is not None:
            raise ProductionBindingError("Mem0 extraction tools are not authorized")
        if tool_choice != "auto":
            raise ProductionBindingError(
                "Mem0 extraction tool_choice changed without tools"
            )
        if kwargs or response_format != {"type": "json_object"}:
            raise ProductionBindingError(
                "Mem0 extraction requires the exact JSON-object response format"
            )

        # The controlled codex_sdk route rejects non-default sampling knobs.
        # Mem0 2.0.18's default output allowance remains fixed at 2,000 tokens.
        started = time.perf_counter()
        raw_response = self._client.chat.completions.with_raw_response.create(
            model=MEM0_EXTRACTION_MODEL,
            messages=normalized,
            response_format={"type": "json_object"},
            max_completion_tokens=_MEM0_EXTRACTION_MAX_COMPLETION_TOKENS,
        )
        try:
            raw_payload = json.loads(raw_response.content)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise ProductionBindingError(
                "extraction response is not exact JSON bytes"
            ) from exc
        raw_usage = raw_payload.get("usage") if isinstance(raw_payload, dict) else None
        if not isinstance(raw_usage, dict):
            raise ProductionBindingError(
                "extraction response omitted exact non-negative provider usage"
            )
        raw_prompt_tokens = raw_usage.get("prompt_tokens")
        raw_completion_tokens = raw_usage.get("completion_tokens")
        raw_total_tokens = raw_usage.get("total_tokens")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                raw_prompt_tokens,
                raw_completion_tokens,
                raw_total_tokens,
            )
        ):
            raise ProductionBindingError(
                "extraction response omitted exact non-negative provider usage"
            )
        if raw_total_tokens != raw_prompt_tokens + raw_completion_tokens:
            raise ProductionBindingError(
                "extraction provider usage total does not close"
            )
        response = raw_response.parse()
        elapsed = max(0.0, time.perf_counter() - started)
        observed_model = getattr(response, "model", None)
        _must_equal(
            observed_model,
            MEM0_EXTRACTION_RESPONSE_MODEL,
            "extraction response model",
        )
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise ProductionBindingError(
                "extraction response omitted its first assistant message"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise ProductionBindingError("extraction response content is empty")
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (prompt_tokens, completion_tokens, total_tokens)
        ):
            raise ProductionBindingError(
                "extraction response omitted exact non-negative provider usage"
            )
        if total_tokens != prompt_tokens + completion_tokens:
            raise ProductionBindingError(
                "extraction provider usage total does not close"
            )
        if (
            prompt_tokens != raw_prompt_tokens
            or completion_tokens != raw_completion_tokens
            or total_tokens != raw_total_tokens
        ):
            raise ProductionBindingError(
                "extraction parsed provider usage differs from raw response"
            )
        with self._usage_lock:
            self._provider_input_tokens += prompt_tokens
            self._provider_output_tokens += completion_tokens
            self._provider_total_tokens += total_tokens
            self._provider_usage_records += 1
            self._provider_latency_s += elapsed
        return content.strip()

    def request_token_state_receipt(self) -> Mapping[str, Any]:
        return {
            "contract": "stateless-request-token-state-v1",
            "persisted_request_token_state": False,
            "retained_request_token_state_bytes": 0,
            "request_token_state_evidence_kind": (
                "local_injected_request_token_state_contract"
            ),
            "external_provider_persistence_certified": False,
        }

    def assert_call_budget_closed(self) -> None:
        self._cap.assert_closed()
        cap = self._cap.receipt()
        with self._usage_lock:
            if self._provider_usage_records != cap["completed"]:
                raise ProductionBindingError(
                    "extraction provider usage records do not close the HTTP budget"
                )
            if self._provider_total_tokens != (
                self._provider_input_tokens + self._provider_output_tokens
            ):
                raise ProductionBindingError(
                    "extraction provider token usage does not close"
                )

    def transport_receipt(self) -> dict[str, Any]:
        with self._usage_lock:
            usage = {
                "provider_usage_status": "provider_reported_exact",
                "provider_usage_records": self._provider_usage_records,
                "provider_input_tokens": self._provider_input_tokens,
                "provider_output_tokens": self._provider_output_tokens,
                "provider_total_tokens": self._provider_total_tokens,
                "provider_latency_s": self._provider_latency_s,
            }
        return {
            **self._cap.receipt(),
            **usage,
            "production_eligible": True,
            "provider": MEM0_EXTRACTION_PROVIDER,
            "model": MEM0_EXTRACTION_MODEL,
            "revision": MEM0_EXTRACTION_REVISION,
            "route_identity_sha256": MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
            "request_identity_sha256": (
                _MEM0_EXTRACTION_REQUEST_IDENTITY_SHA256
            ),
            "gateway_url": MEM0_EXTRACTION_GATEWAY_URL,
            "max_completion_tokens": _MEM0_EXTRACTION_MAX_COMPLETION_TOKENS,
            "sampling_parameters_omitted": True,
            "sdk_retries": 0,
            "http_transport_retries": 0,
            "follow_redirects": False,
            "trust_env": False,
            "cap_boundary": "httpx.BaseTransport.handle_request",
            "external_http_attempts_certified": True,
            "external_provider_persistence_certified": False,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        client = self._client
        self._client = None
        client.close()


@contextmanager
def _mem0_default_llm_constructor_environment() -> Iterator[None]:
    """Let Memory.from_config build its unused default LLM, then restore env."""

    names = (
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_API_BASE",
    )
    with _OPENAI_CONSTRUCTION_ENV_LOCK:
        before = {name: os.environ.get(name) for name in names}
        os.environ["OPENAI_API_KEY"] = _MEM0_CONSTRUCTOR_ONLY_API_KEY
        for name in names[1:]:
            os.environ.pop(name, None)
        try:
            yield
        finally:
            for name, value in before.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _verified_fastembed_bm25_cache() -> tuple[Path, Path, dict[str, Any]]:
    cache_root = Path(tempfile.gettempdir()).resolve() / "fastembed_cache"
    repository = cache_root / "models--Qdrant--bm25"
    ref = repository / "refs" / "main"
    snapshot = repository / "snapshots" / _FASTEMBED_BM25_REVISION
    try:
        observed_revision = ref.read_text(encoding="utf-8").strip()
        resolved_snapshot = snapshot.resolve(strict=True)
    except OSError as exc:
        raise ProductionBindingError(
            "exact local Qdrant/bm25 snapshot is absent"
        ) from exc
    _must_equal(
        observed_revision,
        _FASTEMBED_BM25_REVISION,
        "FastEmbed Qdrant/bm25 main revision",
    )
    if not resolved_snapshot.is_dir():
        raise ProductionBindingError("Qdrant/bm25 snapshot is not a directory")
    files = {
        path.relative_to(resolved_snapshot).as_posix(): _sha256_bytes(
            path.read_bytes()
        )
        for path in sorted(
            resolved_snapshot.rglob("*"), key=lambda item: item.as_posix()
        )
        if path.is_file()
    }
    observed_sha = canonical_json_sha256(files)
    _must_equal(
        observed_sha,
        _FASTEMBED_BM25_ASSET_SHA256,
        "FastEmbed Qdrant/bm25 asset tree",
    )
    receipt = {
        "model": "Qdrant/bm25",
        "revision": _FASTEMBED_BM25_REVISION,
        "asset_tree_sha256": observed_sha,
        "cache_root": cache_root.as_posix(),
        "file_count": len(files),
        "local_files_only": True,
        "network_calls_authorized": 0,
    }
    return cache_root, resolved_snapshot, receipt


def _new_exact_bm25_encoder() -> tuple[Any, dict[str, Any]]:
    cache_root, snapshot, cache_receipt = _verified_fastembed_bm25_cache()
    attempts: list[str] = []
    retry_sleeps: list[float] = []
    encoder: Any | None = None
    model_management: Any | None = None
    original_time: Any | None = None
    try:
        fastembed_module = importlib.import_module("fastembed")
        model_management = importlib.import_module(
            "fastembed.common.model_management"
        )
        original_time = getattr(model_management, "time")

        class _NoRetryTime:
            def __getattr__(self, name: str) -> Any:
                return getattr(original_time, name)

            @staticmethod
            def sleep(seconds: float) -> None:
                retry_sleeps.append(float(seconds))
                raise ProductionBindingError(
                    "FastEmbed entered its forbidden download-retry sleep path"
                )

        model_management.time = _NoRetryTime()
        encoder_type = getattr(fastembed_module, "SparseTextEmbedding")
        with _blocked_network_probe() as attempts:
            encoder = encoder_type(
                model_name="Qdrant/bm25",
                cache_dir=str(cache_root),
                local_files_only=True,
                specific_model_path=str(snapshot),
            )
            rows = list(encoder.embed(["memory retrieval operational probe"]))
        if not rows or not len(rows[0].indices) or not len(rows[0].values):
            raise ProductionBindingError("FastEmbed BM25 probe was empty")
        model_dir = Path(getattr(getattr(encoder, "model", None), "_model_dir", ""))
        _must_equal(
            model_dir.resolve(strict=True),
            snapshot,
            "FastEmbed Qdrant/bm25 loaded snapshot",
        )
    except ProductionBindingError:
        raise
    except BaseException as exc:
        raise ProductionBindingError(
            "exact local Qdrant/bm25 operational probe failed"
        ) from exc
    finally:
        if model_management is not None and original_time is not None:
            model_management.time = original_time
    _must_equal(len(attempts), 0, "FastEmbed BM25 network attempts")
    _must_equal(len(retry_sleeps), 0, "FastEmbed BM25 retry sleeps")
    assert encoder is not None
    return encoder, {
        **cache_receipt,
        "specific_model_path": snapshot.as_posix(),
        "retry_sleep_attempts": 0,
    }


def _exact_mem0_stack_preflight() -> Any:
    """Execute the pinned hybrid/entity probes with all sockets denied.

    FastEmbed 0.7.3 deliberately accepts ``local_files_only`` through its
    ``**kwargs`` compatibility boundary.  The proof here therefore rests on
    the exact distribution version and the successful concrete local-only
    construction/embedding call, rather than a permissive signature guess.
    """

    from memory_condense.eval.mem0_adapter import Mem0StackIdentity

    for distribution, expected in {
        **dict(_MEM0_STACK_DEPENDENCY_VERSIONS),
        **dict(_MEM0_SPACY_TRANSITIVE_VERSIONS),
    }.items():
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ProductionBindingError(
                f"exact Mem0 stack dependency {distribution!r} is absent"
            ) from exc
        _must_equal(observed, expected, f"Mem0 stack dependency {distribution}")

    try:
        spacy_model = importlib.import_module("en_core_web_sm")
        nlp = spacy_model.load()
        pipe_names = set(getattr(nlp, "pipe_names", ()))
        if not {"ner", "lemmatizer"}.issubset(pipe_names):
            raise ProductionBindingError(
                "en_core_web_sm omitted NER or lemmatization"
            )
        probe_doc = nlp("Alice visited Seattle and remembered the visit.")
        if not any(getattr(token, "lemma_", "") for token in probe_doc):
            raise ProductionBindingError("spaCy lemmatization probe was empty")
        if not list(getattr(probe_doc, "ents", ())):
            raise ProductionBindingError("spaCy entity probe was empty")
    except ProductionBindingError:
        raise
    except BaseException as exc:
        raise ProductionBindingError(
            "exact local en_core_web_sm operational probe failed"
        ) from exc

    encoder: Any | None = None
    try:
        encoder, _cache_receipt = _new_exact_bm25_encoder()
    finally:
        close = getattr(encoder, "close", None)
        if callable(close):
            close()
        if encoder is not None:
            del encoder
        gc.collect()
    return Mem0StackIdentity(
        dependency_versions=MappingProxyType(
            dict(_MEM0_STACK_DEPENDENCY_VERSIONS)
        ),
        bm25_model="Qdrant/bm25",
        spacy_model="en_core_web_sm",
        bm25_operational=True,
        entity_extraction_operational=True,
    )


def _materialize_mem0_config(
    authorization: RetrievalStageAuthorization,
    state_root: Path,
) -> dict[str, Any]:
    stable = _mapping(authorization.mem0_stable_payload, "mem0_stable_payload")
    config = _mapping(stable.get("config"), "Mem0 stable config")
    vector = _mapping(config.get("vector_store"), "Mem0 vector-store config")
    vector_config = _mapping(
        vector.get("config"), "Mem0 vector-store config.config"
    )
    vector_config["path"] = str(state_root / "qdrant")
    vector["config"] = vector_config
    config["vector_store"] = vector
    config["history_db_path"] = str(state_root / "history.sqlite")
    return config


def _bound_embedder_receipt(
    memory: Any,
    authorization: RetrievalStageAuthorization,
) -> dict[str, Any]:
    identity = _mapping(
        authorization.embedder_model_identity,
        "embedder_model_identity",
    )
    embedder = getattr(memory, "embedding_model", None)
    if embedder is None:
        raise ProductionBindingError("Mem0 Memory omitted its embedding_model")
    embedder_type = type(embedder)
    if (
        embedder_type.__module__ != "mem0.embeddings.huggingface"
        or embedder_type.__name__ != "HuggingFaceEmbedding"
    ):
        raise ProductionBindingError("Mem0 bound embedder type changed")
    config = getattr(embedder, "config", None)
    expected_kwargs = {
        "revision": DEFAULT_MODEL_REVISION,
        "local_files_only": True,
        "trust_remote_code": False,
        "device": identity["device"],
    }
    for field, expected in {
        "model": DEFAULT_MODEL_NAME,
        "embedding_dims": DEFAULT_MODEL_DIM,
        "huggingface_base_url": None,
        "model_kwargs": expected_kwargs,
    }.items():
        _must_equal(
            getattr(config, field, None),
            expected,
            f"bound Mem0 embedder {field}",
        )
    model = getattr(embedder, "model", None)
    dimension_reader = getattr(model, "get_sentence_embedding_dimension", None)
    if not callable(dimension_reader):
        raise ProductionBindingError("bound BGE-M3 omitted its dimension reader")
    _must_equal(
        int(dimension_reader()),
        DEFAULT_MODEL_DIM,
        "bound BGE-M3 dimension",
    )
    observed_device = str(getattr(model, "device", "")).casefold().split(":", 1)[0]
    _must_equal(observed_device, identity["device"], "bound BGE-M3 device")
    body = {
        "format": "memory-condense-bound-mem0-bge-m3-v1",
        "concrete_type": (
            f"{embedder_type.__module__}.{embedder_type.__name__}"
        ),
        "model": DEFAULT_MODEL_NAME,
        "revision": DEFAULT_MODEL_REVISION,
        "authorized_checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
        "checkpoint_bytes_rehashed_per_factory": False,
        "dimension": DEFAULT_MODEL_DIM,
        "device": identity["device"],
        "local_files_only": True,
        "trust_remote_code": False,
        "network_calls_authorized": 0,
    }
    return {**body, "receipt_sha256": canonical_json_sha256(body)}


def _bind_memory_transport(
    memory: Any,
    transport: LiteLLMTerraExtractionTransport,
) -> None:
    original_llm = getattr(memory, "llm", None)
    if original_llm is None:
        raise ProductionBindingError("Mem0 Memory omitted its default llm")
    original_type = type(original_llm)
    if (
        original_type.__module__ != "mem0.llms.openai"
        or original_type.__name__ != "OpenAILLM"
    ):
        raise ProductionBindingError("Mem0 default LLM type changed before binding")
    original_close = getattr(memory, "close", None)
    memory.llm = transport
    if getattr(memory, "llm", None) is not transport:
        raise ProductionBindingError("could not bind exact extraction transport")

    old_client = getattr(original_llm, "client", None)
    close_old_client = getattr(old_client, "close", None)
    if callable(close_old_client):
        close_old_client()

    closed = False

    def close_bound_memory() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        errors: list[BaseException] = []
        if callable(original_close):
            try:
                original_close()
            except BaseException as exc:
                errors.append(exc)
        try:
            transport.assert_call_budget_closed()
        except BaseException as exc:
            errors.append(exc)
        try:
            transport.close()
        except BaseException as exc:
            errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("bound Mem0 transport cleanup failed", errors)

    try:
        memory.close = close_bound_memory
    except BaseException:
        memory.llm = original_llm
        transport.close()
        raise


def _materialize_exact_qdrant_stores(memory: Any) -> tuple[Any, Any]:
    vector_store = getattr(memory, "vector_store", None)
    try:
        entity_store = memory.entity_store
    except BaseException as exc:
        raise ProductionBindingError(
            "Mem0 entity Qdrant store could not be materialized locally"
        ) from exc
    stores = (vector_store, entity_store)
    if stores[0] is None or stores[1] is None or stores[0] is stores[1]:
        raise ProductionBindingError("Mem0 dense/entity Qdrant stores are invalid")
    for index, store in enumerate(stores):
        store_type = type(store)
        if (
            store_type.__module__ != "mem0.vector_stores.qdrant"
            or store_type.__name__ != "Qdrant"
        ):
            raise ProductionBindingError(
                f"Mem0 Qdrant store {index} type changed"
            )
    return stores


def _harden_owned_qdrant_cleanup(owned: Any) -> Mapping[str, Any]:
    """Close local collection SQLite handles before Qdrant drops their names.

    Qdrant Client 1.15.1's local ``delete_collection`` removes the collection
    object from its registry without calling ``LocalCollection.close`` first.
    On Windows that leaves ``storage.sqlite`` locked when the owned-state
    wrapper tries to erase its directory.  This exact-version shim pre-closes
    those handles, then delegates to the existing owned cleanup unchanged.
    """

    memory = getattr(owned, "backend", None)

    def live_locals_and_handles() -> tuple[list[Any], list[Any]]:
        stores = [getattr(memory, "vector_store", None)]
        entity_store = getattr(memory, "_entity_store", None)
        if entity_store is not None:
            stores.append(entity_store)
        locals_: list[Any] = []
        handles: list[Any] = []
        for index, store in enumerate(stores):
            store_type = type(store)
            if (
                store_type.__module__ != "mem0.vector_stores.qdrant"
                or store_type.__name__ != "Qdrant"
            ):
                raise ProductionBindingError(
                    f"owned Mem0 Qdrant store {index} type changed"
                )
            client = getattr(store, "client", None)
            client_type = type(client)
            if (
                client_type.__module__ != "qdrant_client.qdrant_client"
                or client_type.__name__ != "QdrantClient"
            ):
                raise ProductionBindingError(
                    f"owned Mem0 Qdrant client {index} type changed"
                )
            local = getattr(client, "_client", None)
            local_type = type(local)
            if (
                local_type.__module__ != "qdrant_client.local.qdrant_local"
                or local_type.__name__ != "QdrantLocal"
            ):
                raise ProductionBindingError(
                    f"owned Mem0 Qdrant backend {index} is not local"
                )
            if all(local is not existing for existing in locals_):
                locals_.append(local)
        for local in locals_:
            collections = getattr(local, "collections", None)
            if not isinstance(collections, Mapping) or not collections:
                raise ProductionBindingError(
                    "owned local Qdrant has no live collection"
                )
            for collection in list(collections.values()):
                collection_type = type(collection)
                if (
                    collection_type.__module__
                    != "qdrant_client.local.local_collection"
                    or collection_type.__name__ != "LocalCollection"
                ):
                    raise ProductionBindingError(
                        "owned Qdrant collection type changed"
                    )
                close = getattr(collection, "close", None)
                if not callable(close):
                    raise ProductionBindingError(
                        "owned Qdrant collection cannot close"
                    )
                if all(collection is not existing for existing in handles):
                    handles.append(collection)
        return locals_, handles

    initial_locals, initial_handles = live_locals_and_handles()

    original_close = getattr(owned, "close", None)
    if not callable(original_close):
        raise ProductionBindingError("owned Mem0 backend omitted cleanup")
    preclosed = False

    def close_owned_qdrant_first() -> None:
        nonlocal preclosed
        errors: list[BaseException] = []
        if not preclosed:
            preclosed = True
            try:
                _locals, handles = live_locals_and_handles()
            except BaseException as exc:
                errors.append(exc)
                handles = []
            else:
                for collection in handles:
                    try:
                        collection.close()
                    except BaseException as exc:
                        errors.append(exc)
        try:
            original_close()
        except BaseException as exc:
            errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup(
                "owned Qdrant pre-close and Mem0 cleanup failed", errors
            )

    owned.close = close_owned_qdrant_first
    if getattr(owned, "close", None) is not close_owned_qdrant_first:
        raise ProductionBindingError("could not bind owned Qdrant cleanup order")
    return MappingProxyType(
        {
            "format": "memory-condense-owned-qdrant-cleanup-v1",
            "qdrant_client_version": "1.15.1",
            "initial_local_clients_bound": len(initial_locals),
            "initial_collection_handles_bound": len(initial_handles),
            "dynamic_store_and_collection_registries_bound": True,
            "collection_handles_preclosed_before_delete": True,
        }
    )


def _bind_exact_bm25_encoders(memory: Any) -> dict[str, Any]:
    """Bind two verified local-only encoders before Mem0 can lazy-download."""

    stores = _materialize_exact_qdrant_stores(memory)
    encoders: list[Any] = []
    cache_receipts: list[dict[str, Any]] = []
    try:
        for index, store in enumerate(stores):
            store_type = type(store)
            if (
                store_type.__module__ != "mem0.vector_stores.qdrant"
                or store_type.__name__ != "Qdrant"
            ):
                raise ProductionBindingError(
                    f"Mem0 Qdrant store {index} type changed"
                )
            if getattr(store, "_has_bm25_slot", None) is not True:
                raise ProductionBindingError(
                    f"Mem0 Qdrant store {index} omitted its BM25 slot"
                )
            if getattr(store, "_bm25_encoder", None) is not None:
                raise ProductionBindingError(
                    f"Mem0 Qdrant store {index} initialized BM25 before binding"
                )
            encoder, cache_receipt = _new_exact_bm25_encoder()
            store._bm25_encoder = encoder
            if getattr(store, "_bm25_encoder", None) is not encoder:
                raise ProductionBindingError(
                    f"Mem0 Qdrant store {index} rejected exact BM25 encoder"
                )
            encoders.append(encoder)
            cache_receipts.append(cache_receipt)
    except BaseException:
        for encoder in encoders:
            close = getattr(encoder, "close", None)
            if callable(close):
                close()
        raise
    if encoders[0] is encoders[1]:
        raise ProductionBindingError("Mem0 Qdrant stores shared one BM25 instance")
    if cache_receipts[0] != cache_receipts[1]:
        raise ProductionBindingError("Mem0 BM25 cache identity changed between stores")
    body = {
        "format": "memory-condense-bound-mem0-bm25-v1",
        **cache_receipts[0],
        "bound_store_roles": ["memory", "entity"],
        "encoder_instances": 2,
        "distinct_encoder_instances": True,
        "internal_lazy_download_path_reachable": False,
    }
    return {**body, "receipt_sha256": canonical_json_sha256(body)}


class ExactMem0AdapterFactory:
    """Single-use, non-injectable factory for the locked Mem0 retrieval arm."""

    production_eligible = True

    def __init_subclass__(cls, **_kwargs: Any) -> None:
        raise TypeError("ExactMem0AdapterFactory cannot be subclassed")

    def __init__(self, authorization: RetrievalStageAuthorization) -> None:
        _require_exact_authorization(authorization, stage="retrieval")
        config_receipt = validate_production_mem0_config(authorization)
        embedder_contract = validate_local_bge_m3_contract(authorization)
        authorized = authorization.authorized_extraction_calls
        if isinstance(authorized, bool) or not isinstance(authorized, int) or authorized < 1:
            raise ProductionBindingError("authorized extraction calls are invalid")
        self._authorization = authorization
        self._config_receipt = config_receipt
        self._embedder_contract = embedder_contract
        self._transport: LiteLLMTerraExtractionTransport | None = None
        self._bound_cleanup: Mapping[str, Any] | None = None
        self._bound_embedder: Mapping[str, Any] | None = None
        self._bound_bm25: Mapping[str, Any] | None = None
        self._called = False

    def __call__(self, owned_state_dir: Path) -> Any:
        if self._called:
            raise ProductionBindingError("exact Mem0 adapter factory is single-use")
        self._called = True
        if validate_production_mem0_config(self._authorization) != self._config_receipt:
            raise ProductionBindingError("Mem0 config changed after factory creation")
        if validate_local_bge_m3_contract(self._authorization) != self._embedder_contract:
            raise ProductionBindingError("BGE-M3 contract changed after factory creation")

        state_root = Path(owned_state_dir).resolve(strict=False)
        if state_root == state_root.parent:
            raise ProductionBindingError("owned Mem0 state cannot be a filesystem root")
        config = _materialize_mem0_config(self._authorization, state_root)
        transport = LiteLLMTerraExtractionTransport(
            authorized=self._authorization.authorized_extraction_calls
        )
        self._transport = transport
        owned: Any | None = None
        try:
            from memory_condense.eval import mem0_adapter

            backend_factory = mem0_adapter.Mem0OSSBackendFactory(
                config=config,
                llm_model_id=MEM0_EXTRACTION_MODEL,
                embedder_model_id=DEFAULT_MODEL_NAME,
                owned_state_dir=state_root,
                _stack_preflight=_exact_mem0_stack_preflight,
            )
            with _mem0_default_llm_constructor_environment():
                owned = backend_factory()
            memory = getattr(owned, "backend", None)
            if memory is None:
                raise ProductionBindingError("owned Mem0 backend omitted Memory")
            try:
                _materialize_exact_qdrant_stores(memory)
            except BaseException:
                # Even a partial entity-store construction must inherit the
                # Windows-safe main-store cleanup order before unwinding.
                _harden_owned_qdrant_cleanup(owned)
                raise
            bound_cleanup = _harden_owned_qdrant_cleanup(owned)
            bound_bm25 = _bind_exact_bm25_encoders(memory)
            bound_embedder = _bound_embedder_receipt(memory, self._authorization)
            _bind_memory_transport(memory, transport)
            adapter = mem0_adapter.Mem0LongMemEvalAdapter(
                backend=owned,
                token_counter=count_tokens,
            )
            adapter._production_extraction_transport = transport
            adapter._bound_cleanup_receipt = MappingProxyType(
                dict(bound_cleanup)
            )
            adapter._bound_embedder_receipt = MappingProxyType(bound_embedder)
            adapter._bound_bm25_receipt = MappingProxyType(bound_bm25)
            self._bound_cleanup = MappingProxyType(dict(bound_cleanup))
            self._bound_embedder = MappingProxyType(bound_embedder)
            self._bound_bm25 = MappingProxyType(bound_bm25)
            return adapter
        except BaseException:
            if owned is not None:
                close_owned = getattr(owned, "close", None)
                if callable(close_owned):
                    try:
                        close_owned()
                    except BaseException:
                        pass
            transport.close()
            raise

    def binding_receipt(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "kind": "exact_mem0_adapter_factory_v1",
                "config": dict(self._config_receipt),
                "embedder_contract": dict(self._embedder_contract),
                "bound_cleanup": (
                    dict(self._bound_cleanup)
                    if self._bound_cleanup is not None
                    else None
                ),
                "bound_embedder": (
                    dict(self._bound_embedder)
                    if self._bound_embedder is not None
                    else None
                ),
                "bound_bm25": (
                    dict(self._bound_bm25)
                    if self._bound_bm25 is not None
                    else None
                ),
                "transport": (
                    self._transport.transport_receipt()
                    if self._transport is not None
                    else None
                ),
            }
        )


def run_terra_extraction_canary() -> Mapping[str, Any]:
    """Spend exactly one real extraction call and return a text-free receipt."""

    transport = LiteLLMTerraExtractionTransport(authorized=1)
    try:
        response = transport.generate_response(
            [
                {
                    "role": "system",
                    "content": (
                        "Extract durable facts. Return one JSON object with a "
                        "memory array; every item must contain a text field."
                    ),
                },
                {
                    "role": "user",
                    "content": "Alice's favorite color is blue.",
                },
            ],
            response_format={"type": "json_object"},
        )
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ProductionBindingError(
                "Terra extraction canary did not return strict JSON"
            ) from exc
        memories = parsed.get("memory") if isinstance(parsed, dict) else None
        if (
            not isinstance(memories, list)
            or not memories
            or not all(
                isinstance(row, Mapping)
                and isinstance(row.get("text"), str)
                and bool(row["text"].strip())
                for row in memories
            )
        ):
            raise ProductionBindingError(
                "Terra extraction canary omitted its memory/text payload"
            )
        transport.assert_call_budget_closed()
        body = {
            "format": "memory-condense-mem0-terra-extraction-canary-v1",
            "route_identity_sha256": MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
            "memory_count": len(memories),
            "response_sha256": _sha256_bytes(response.encode("utf-8")),
            "transport": transport.transport_receipt(),
        }
        return MappingProxyType(
            {**body, "receipt_sha256": canonical_json_sha256(body)}
        )
    finally:
        transport.close()


def _factory_canary_authorization(
    record: Mapping[str, Any],
) -> RetrievalStageAuthorization:
    """Build explicit non-campaign authority for one diagnostic add/search."""

    from .run_shard import RetrievalStageAuthorization

    extraction_body = {
        "provider": MEM0_EXTRACTION_PROVIDER,
        "model": MEM0_EXTRACTION_MODEL,
        "revision": MEM0_EXTRACTION_REVISION,
        "provider_retries": 0,
        "logical_call_boundary": "Memory.llm.generate_response",
        "logical_calls_per_add": 1,
        "http_attempts_certified": False,
    }
    extraction = {
        **extraction_body,
        "model_identity_sha256": canonical_json_sha256(extraction_body),
    }
    embedder_body = {
        "provider": "huggingface",
        "model": DEFAULT_MODEL_NAME,
        "revision": DEFAULT_MODEL_REVISION,
        "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
        "dimension": DEFAULT_MODEL_DIM,
        "device": "cuda",
        "dtype": "float32",
        "execution": "local_offline",
        "network_calls_authorized": 0,
        "runtime_probe_required": True,
    }
    embedder = {
        **embedder_body,
        "model_identity_sha256": canonical_json_sha256(embedder_body),
    }
    stable = {
        "protocol": "mem0-oss-2.0.18-certified-local-v1",
        "config": {
            "version": "v1.1",
            "custom_instructions": None,
            "reranker": None,
            "llm": {
                "provider": MEM0_EXTRACTION_PROVIDER,
                "config": {"model": MEM0_EXTRACTION_MODEL},
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": DEFAULT_MODEL_NAME,
                    "embedding_dims": DEFAULT_MODEL_DIM,
                    "huggingface_base_url": None,
                    "model_kwargs": {
                        "revision": DEFAULT_MODEL_REVISION,
                        "local_files_only": True,
                        "trust_remote_code": False,
                        "device": "cuda",
                    },
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "embedding_model_dims": DEFAULT_MODEL_DIM,
                    "collection_name": "longmemeval",
                    "path": "<owned_state>/qdrant",
                    "on_disk": True,
                },
            },
            "history_db_path": "<owned_state>/history.sqlite",
        },
        "stack": {
            "dependency_versions": dict(_MEM0_STACK_DEPENDENCY_VERSIONS),
            "bm25_model": "Qdrant/bm25",
            "spacy_model": "en_core_web_sm",
            "bm25_operational": True,
            "entity_extraction_operational": True,
        },
    }
    record_sha = canonical_json_sha256(record)
    lock_path = Path(__file__).with_name("pixi.lock")
    lock_sha, _lock_bytes = _sha256_file(lock_path)
    policy_sha = canonical_json_sha256(
        {
            "format": "memory-condense-mem0-factory-canary-authority-v1",
            "campaign_authority": False,
            "record_sha256": record_sha,
        }
    )
    return RetrievalStageAuthorization(
        sample_offset=0,
        sample_sha256=record_sha,
        raw_history_bundle_sha256=record_sha,
        question_ids=(str(record["question_id"]),),
        authorized_add_operations=1,
        authorized_extraction_calls=1,
        authorized_search_operations=1,
        source_validation_policy_sha256=policy_sha,
        source_implementation_sha256=policy_sha,
        source_environment_lock_sha256=policy_sha,
        mem0_policy_sha256=policy_sha,
        mem0_tool_implementation_sha256=tool_implementation_sha256(
            Path(__file__).parent
        ),
        mem0_environment_lock_sha256=lock_sha,
        mem0_stable_config_sha256=canonical_json_sha256(stable),
        source_evaluation_identity={
            "format": "synthetic_factory_canary_not_campaign_input_v1"
        },
        mem0_stable_payload=stable,
        extraction_model_identity=extraction,
        extraction_model_identity_sha256=canonical_json_sha256(extraction),
        embedder_model_identity=embedder,
        embedder_model_identity_sha256=canonical_json_sha256(embedder),
        mem0_provider_retries=0,
    )


def _factory_canary_record() -> dict[str, Any]:
    return {
        "question_id": "mem0-factory-canary-v2",
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "Maya's preferred tea is oolong.",
                },
                {"role": "assistant", "content": "I will record that."},
            ]
        ],
        "haystack_session_ids": ["factory-canary-session"],
        "haystack_dates": ["2026-08-29 00:00"],
    }


def _factory_canary_state(value: str | os.PathLike[str]) -> Path:
    state = Path(value).resolve(strict=False)
    if state.exists():
        raise ProductionBindingError("factory canary owned_state_dir already exists")
    if not state.parent.is_dir():
        raise ProductionBindingError(
            "factory canary owned_state_dir parent must already exist"
        )
    return state


def _exception_texts(value: BaseException) -> tuple[str, ...]:
    if isinstance(value, BaseExceptionGroup):
        return tuple(
            text
            for child in value.exceptions
            for text in _exception_texts(child)
        )
    return (f"{type(value).__name__}: {value}",)


def _capture_exact_factory_cleanup_topology(
    adapter: Any,
) -> tuple[Any, Any, tuple[Any, ...], Mapping[str, Any]]:
    """Capture only the owned handles needed to prove complete local cleanup."""

    owned = getattr(adapter, "_backend", None)
    memory = getattr(owned, "backend", None)
    if memory is None:
        raise ProductionBindingError(
            "exact Mem0 adapter omitted its owned Memory backend"
        )
    stores = _materialize_exact_qdrant_stores(memory)
    local_clients: list[Any] = []
    for index, store in enumerate(stores):
        client = getattr(store, "client", None)
        client_type = type(client)
        if (
            client_type.__module__ != "qdrant_client.qdrant_client"
            or client_type.__name__ != "QdrantClient"
        ):
            raise ProductionBindingError(
                f"exact Mem0 Qdrant client {index} type changed"
            )
        local = getattr(client, "_client", None)
        local_type = type(local)
        if (
            local_type.__module__ != "qdrant_client.local.qdrant_local"
            or local_type.__name__ != "QdrantLocal"
        ):
            raise ProductionBindingError(
                f"exact Mem0 Qdrant backend {index} is not local"
            )
        if all(local is not existing for existing in local_clients):
            local_clients.append(local)

    history = getattr(memory, "db", None)
    history_type = type(history)
    if (
        history_type.__module__ != "mem0.memory.storage"
        or history_type.__name__ != "SQLiteManager"
        or getattr(history, "connection", None) is None
    ):
        raise ProductionBindingError(
            "exact Mem0 history SQLite connection was not live before cleanup"
        )
    graph_fields = ("graph", "graph_store", "_graph", "_graph_store")
    if any(getattr(memory, field, None) is not None for field in graph_fields):
        raise ProductionBindingError(
            "exact Mem0 factory unexpectedly initialized a graph store"
        )
    if getattr(memory, "_telemetry_vector_store", None) is not None:
        raise ProductionBindingError(
            "exact Mem0 factory unexpectedly initialized telemetry state"
        )
    body = {
        "format": "memory-condense-mem0-cleanup-topology-v1",
        "qdrant_store_count": 2,
        "distinct_local_qdrant_client_count": len(local_clients),
        "entity_store_materialized": getattr(memory, "_entity_store", None)
        is stores[1],
        "history_connection_live_before_cleanup": True,
        "graph_store_absent": True,
        "telemetry_store_absent": True,
    }
    if body["entity_store_materialized"] is not True:
        raise ProductionBindingError(
            "exact Mem0 entity store was not retained after materialization"
        )
    return memory, history, tuple(local_clients), MappingProxyType(body)


def run_mem0_factory_presend_cleanup_canary(
    *,
    owned_state_dir: str | os.PathLike[str],
) -> Mapping[str, Any]:
    """Construct and erase the real factory with sockets denied and no send."""

    state = _factory_canary_state(owned_state_dir)
    record = _factory_canary_record()
    authorization = _factory_canary_authorization(record)
    factory = ExactMem0AdapterFactory(authorization)
    adapter: Any | None = None
    memory: Any | None = None
    history: Any | None = None
    local_clients: tuple[Any, ...] = ()
    topology: Mapping[str, Any] | None = None
    cleanup_error: BaseException | None = None
    with _blocked_network_probe() as network_attempts:
        adapter = factory(state)
        memory, history, local_clients, topology = (
            _capture_exact_factory_cleanup_topology(adapter)
        )
        try:
            adapter.cleanup()
        except BaseException as exc:
            cleanup_error = exc
    if cleanup_error is None:
        raise ProductionBindingError(
            "provider-free factory cleanup failed to enforce its unused call budget"
        )
    error_texts = _exception_texts(cleanup_error)
    if not any("attempt accounting did not close" in text for text in error_texts):
        raise cleanup_error
    if state.exists():
        raise ProductionBindingError(
            "provider-free factory cleanup left owned state behind"
        )
    assert memory is not None and history is not None and topology is not None
    if getattr(memory, "db", None) is not None:
        raise ProductionBindingError(
            "provider-free factory cleanup retained its history manager"
        )
    if getattr(history, "connection", None) is not None:
        raise ProductionBindingError(
            "provider-free factory cleanup retained its history connection"
        )
    if not local_clients or any(
        getattr(local, "closed", None) is not True for local in local_clients
    ):
        raise ProductionBindingError(
            "provider-free factory cleanup retained a local Qdrant client"
        )
    _must_equal(len(network_attempts), 0, "provider-free factory network attempts")
    factory_receipt = dict(factory.binding_receipt())
    transport = _mapping(factory_receipt.get("transport"), "canary transport")
    for field, expected in {
        "authorized": 1,
        "attempted": 0,
        "completed": 0,
        "failed": 0,
        "rejected": 0,
    }.items():
        _must_equal(transport.get(field), expected, f"canary transport {field}")
    body = {
        "format": "memory-condense-mem0-factory-presend-cleanup-canary-v1",
        "campaign_authority": False,
        "provider_calls_authorized": 0,
        "network_attempts": 0,
        "unused_transport_budget_rejected": True,
        "owned_state_removed": True,
        "history_connection_closed": True,
        "all_local_qdrant_clients_closed": True,
        "topology": dict(topology),
        "environment_lock_sha256": authorization.mem0_environment_lock_sha256,
        "tool_implementation_sha256": (
            authorization.mem0_tool_implementation_sha256
        ),
        "factory": factory_receipt,
    }
    return MappingProxyType(
        {**body, "receipt_sha256": canonical_json_sha256(body)}
    )


def run_mem0_factory_canary(
    *,
    owned_state_dir: str | os.PathLike[str],
) -> Mapping[str, Any]:
    """Run one real inferred add and one local search through the exact factory.

    This diagnostic has its own explicit synthetic authority and never counts
    toward, publishes into, or reuses state from the locked 100-question arm.
    Its receipt hashes all generated/retrieved text rather than returning it.
    """

    state = _factory_canary_state(owned_state_dir)
    record = _factory_canary_record()
    authorization = _factory_canary_authorization(record)
    factory = ExactMem0AdapterFactory(authorization)
    adapter: Any | None = None
    ingest: Any | None = None
    search: Any | None = None
    operation_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        adapter = factory(state)
        ingest = adapter.ingest_longmemeval_record(record)
        if not tuple(getattr(ingest, "returned_memory_ids", ())):
            raise ProductionBindingError(
                "factory canary extraction returned no durable memory"
            )
        search = adapter.search(
            "Which tea does Maya prefer?",
            max_prompt_tokens=1_024,
            prompt_renderer=lambda question, context: (
                f"Evidence:\n{context}\nQuestion: {question}"
            ),
        )
        if not tuple(getattr(search, "raw_pool", ())):
            raise ProductionBindingError(
                "factory canary local search returned no memory"
            )
    except BaseException as exc:
        operation_error = exc
    finally:
        if adapter is not None:
            try:
                adapter.cleanup()
            except BaseException as exc:
                cleanup_error = exc

    if operation_error is not None:
        if cleanup_error is not None:
            raise BaseExceptionGroup(
                "factory canary operation and exact cleanup both failed",
                [operation_error, cleanup_error],
            )
        raise operation_error
    if cleanup_error is not None:
        raise cleanup_error

    if state.exists():
        raise ProductionBindingError("factory canary owned state was not removed")
    assert ingest is not None and search is not None
    factory_receipt = dict(factory.binding_receipt())
    transport = _mapping(factory_receipt.get("transport"), "canary transport")
    for field, expected in {
        "authorized": 1,
        "attempted": 1,
        "completed": 1,
        "failed": 0,
        "rejected": 0,
    }.items():
        _must_equal(transport.get(field), expected, f"canary transport {field}")
    body = {
        "format": "memory-condense-mem0-exact-factory-canary-v1",
        "campaign_authority": False,
        "record_sha256": canonical_json_sha256(record),
        "environment_lock_sha256": authorization.mem0_environment_lock_sha256,
        "tool_implementation_sha256": (
            authorization.mem0_tool_implementation_sha256
        ),
        "returned_memory_count": len(ingest.returned_memory_ids),
        "ingest_comparison_certified": bool(ingest.comparison_certified),
        "raw_search_pool_count": len(search.raw_pool),
        "packed_search_count": len(search.packed),
        "search_context_sha256": _sha256_bytes(search.context.encode("utf-8")),
        "owned_state_removed": True,
        "factory": factory_receipt,
    }
    return MappingProxyType(
        {**body, "receipt_sha256": canonical_json_sha256(body)}
    )


_EXACT_EXTRACTION_TRANSPORT_TYPE = LiteLLMTerraExtractionTransport
_EXACT_MEM0_ADAPTER_FACTORY_TYPE = ExactMem0AdapterFactory


class _InjectedHardCappedTransport:
    __slots__ = ("_cap", "_delegate")
    role = ""
    production_eligible = False

    def __init__(self, delegate: Callable[..., Any], *, authorized: int) -> None:
        if not callable(delegate):
            raise TypeError("injected transport delegate must be callable")
        self._delegate = delegate
        self._cap = HardTransportAttemptCap(role=self.role, authorized=authorized)

    def _call(self, *args: Any, **kwargs: Any) -> Any:
        return self._cap.call(self._delegate, *args, **kwargs)

    def transport_receipt(self) -> dict[str, Any]:
        return {
            **self._cap.receipt(),
            "production_eligible": False,
            "external_http_attempts_certified": False,
            "external_provider_persistence_certified": False,
        }


class InjectedHardCappedExtractionTransport(_InjectedHardCappedTransport):
    """Development-only logical extraction wrapper; never production trust."""

    role = "extraction"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call(*args, **kwargs)


class _InjectedHardCappedScoringTransport(_InjectedHardCappedTransport):
    expected_model: str

    def __init__(
        self,
        delegate: Callable[..., ProviderCallResult],
        *,
        authorized: int,
        expected_model: str,
    ) -> None:
        if not isinstance(expected_model, str) or not expected_model.strip():
            raise ValueError("scoring transport expected_model must be non-empty")
        self.expected_model = expected_model.strip()
        super().__init__(delegate, authorized=authorized)

    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        max_output_tokens: int,
    ) -> ProviderCallResult:
        if model != self.expected_model:
            raise ProductionBindingError(f"{self.role} transport model mismatch")
        result = self._call(
            messages,
            model=model,
            max_output_tokens=max_output_tokens,
        )
        from .run_shard import ProviderCallResult

        if not isinstance(result, ProviderCallResult):
            raise TypeError("scoring transport must return ProviderCallResult")
        return result

    def request_token_state_receipt(self) -> Mapping[str, Any]:
        reader = getattr(self._delegate, "request_token_state_receipt", None)
        if not callable(reader):
            raise ProductionBindingError(
                f"injected {self.role} delegate omitted request-state receipt"
            )
        return reader()


class InjectedHardCappedResponderTransport(_InjectedHardCappedScoringTransport):
    """Development-only responder cap; rejected by the production issuer."""

    role = "responder"


class InjectedHardCappedJudgeTransport(_InjectedHardCappedScoringTransport):
    """Development-only judge cap; rejected by the production issuer."""

    role = "judge"


def production_binding_readiness() -> Mapping[str, Any]:
    """Return the static, provider-free reason production issuance is closed."""

    blockers: list[dict[str, str]] = []
    if _EXACT_EXTRACTION_TRANSPORT_TYPE is None:
        blockers.append(
            {
                "code": "extraction_send_transport_unresolved",
                "detail": (
                    "implement the frozen Terra observable route with a zero-retry "
                    "hard cap at its concrete HTTP send boundary"
                ),
            }
        )
    if _EXACT_MEM0_ADAPTER_FACTORY_TYPE is None:
        blockers.append(
            {
                "code": "production_mem0_adapter_factory_unresolved",
                "detail": (
                    "bind the selected extraction transport and exact local BGE-M3 "
                    "runtime to one non-injectable Mem0 adapter factory"
                ),
            }
        )
    if _EXACT_RESPONDER_TRANSPORT_TYPE is None:
        blockers.append(
            {
                "code": "responder_send_transport_unresolved",
                "detail": "implement the frozen Terra single-attempt send boundary",
            }
        )
    if _EXACT_JUDGE_TRANSPORT_TYPE is None:
        blockers.append(
            {
                "code": "judge_send_transport_unresolved",
                "detail": "implement the frozen Sol single-attempt send boundary",
            }
        )
    blockers.extend(
        [
            {
                "code": "post_run_transport_closure_unimplemented",
                "detail": (
                    "close extraction/responder/judge send-boundary receipts after "
                    "execution with attempted=completed=authorized and zero "
                    "failed/rejected attempts before publication"
                ),
            },
            {
                "code": "full_source_artifact_attestation_unimplemented",
                "detail": (
                    "persist and independently revalidate dataset, split, selection, "
                    "source-policy, source-implementation, and environment evidence"
                ),
            },
            {
                "code": "positive_report_compare_schema_unimplemented",
                "detail": (
                    "report and comparator intentionally accept only the current "
                    "injected_nonproduction receipt schema"
                ),
            },
        ]
    )
    payload = {
        "format": PROVIDER_FREE_STATUS_FORMAT,
        "status": "ready" if not blockers else "blocked",
        "production_binding_issuance_permitted": not blockers,
        "blockers": blockers,
        "extraction_route": {
            **dict(MEM0_EXTRACTION_ROUTE_IDENTITY),
            "route_identity_sha256": MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
            "revision": MEM0_EXTRACTION_REVISION,
        },
        "local_bge_contract": {
            "model": DEFAULT_MODEL_NAME,
            "revision": DEFAULT_MODEL_REVISION,
            "checkpoint_sha256": BGE_M3_CHECKPOINT_SHA256,
            "dimension": DEFAULT_MODEL_DIM,
            "dtype": "float32",
            "runtime_probe_required": True,
            "network_calls_authorized": 0,
        },
        "external_provider_persistence_certified": False,
    }
    return MappingProxyType(payload)


def _validate_public_production_receipt(
    value: Mapping[str, Any],
    *,
    stage: str,
    authorization_sha256: str,
) -> dict[str, Any]:
    """Validate the full sanitized evidence carried by a positive receipt."""

    receipt = _mapping(value, "production binding receipt")
    _exact_keys(
        receipt,
        {
            "kind",
            "trusted_runtime_binding_receipt_sha256",
            "comparison_certified",
            "external_http_attempts_certified",
            "external_provider_persistence_certified",
            "attestation",
        },
        "production binding receipt",
    )
    _must_equal(receipt.get("kind"), PRODUCTION_BINDING_KIND, "binding kind")
    _must_equal(
        receipt.get("comparison_certified"), True, "binding comparison certification"
    )
    _must_equal(
        receipt.get("external_http_attempts_certified"),
        True,
        "binding HTTP-attempt certification",
    )
    _must_equal(
        receipt.get("external_provider_persistence_certified"),
        False,
        "binding external-provider persistence certification",
    )

    attestation = _mapping(receipt.get("attestation"), "production attestation")
    _exact_keys(
        attestation,
        {
            "format",
            "stage",
            "authorization_sha256",
            "artifact_binding",
            "mem0_config_binding",
            "local_bge_runtime_probe",
            "transport_bindings",
            "external_provider_persistence_certified",
        },
        "production attestation",
    )
    _must_equal(attestation.get("format"), PRODUCTION_BINDING_FORMAT, "attestation format")
    _must_equal(attestation.get("stage"), stage, "attestation stage")
    _must_equal(
        attestation.get("authorization_sha256"),
        authorization_sha256,
        "attestation authorization SHA-256",
    )
    _must_equal(
        attestation.get("external_provider_persistence_certified"),
        False,
        "attestation external-provider persistence certification",
    )
    artifact = _mapping(attestation.get("artifact_binding"), "artifact binding")
    required_artifact = {
        "policy_sha256",
        "mem0_environment_lock_sha256",
        "tool_implementation_sha256",
        "source_environment_lock_sha256",
    }
    _exact_keys(artifact, required_artifact, "artifact binding")
    for field in required_artifact - {"source_environment_lock_sha256"}:
        observed = artifact.get(field)
        if not isinstance(observed, str) or len(observed) != 64:
            raise ProductionBindingError(f"artifact binding {field} is not SHA-256")
    source_lock_sha = artifact.get("source_environment_lock_sha256")
    if stage == "retrieval":
        _must_equal(source_lock_sha, None, "retrieval source-lock binding")
    elif not isinstance(source_lock_sha, str) or len(source_lock_sha) != 64:
        raise ProductionBindingError("scoring source-lock binding is not SHA-256")

    config = _mapping(
        attestation.get("mem0_config_binding"), "Mem0 config binding"
    )
    _exact_keys(
        config,
        {
            "format",
            "stable_config_sha256",
            "extraction_model_identity_sha256",
            "llm_config_fields",
            "provider_retries",
            "qdrant_remote_fields_disabled",
            "owned_state_paths_normalized",
        },
        "Mem0 config binding",
    )
    for field, expected in {
        "format": "memory-condense-mem0-production-config-v1",
        "llm_config_fields": ["model"],
        "provider_retries": 0,
        "qdrant_remote_fields_disabled": True,
        "owned_state_paths_normalized": True,
    }.items():
        _must_equal(config.get(field), expected, f"Mem0 config binding {field}")

    runtime_probe = attestation.get("local_bge_runtime_probe")
    if stage == "retrieval":
        probe = _mapping(runtime_probe, "local BGE runtime probe")
        required_probe = {
            "format",
            "model",
            "revision",
            "checkpoint_sha256_before",
            "checkpoint_sha256_after",
            "checkpoint_unchanged",
            "dimension",
            "device",
            "output_dtype",
            "probe_vectors_sha256",
            "local_files_only",
            "trust_remote_code",
            "network_attempts",
            "network_calls_authorized",
            "receipt_sha256",
        }
        _exact_keys(probe, required_probe, "local BGE runtime probe")
        probe_body = dict(probe)
        probe_sha = probe_body.pop("receipt_sha256")
        _must_equal(
            probe_sha,
            canonical_json_sha256(probe_body),
            "local BGE runtime-probe SHA-256",
        )
        for field, expected in {
            "format": LOCAL_BGE_PROBE_FORMAT,
            "model": DEFAULT_MODEL_NAME,
            "revision": DEFAULT_MODEL_REVISION,
            "checkpoint_sha256_before": BGE_M3_CHECKPOINT_SHA256,
            "checkpoint_sha256_after": BGE_M3_CHECKPOINT_SHA256,
            "checkpoint_unchanged": True,
            "dimension": DEFAULT_MODEL_DIM,
            "output_dtype": "float32",
            "local_files_only": True,
            "trust_remote_code": False,
            "network_attempts": 0,
            "network_calls_authorized": 0,
        }.items():
            _must_equal(probe.get(field), expected, f"local BGE probe {field}")
        if probe.get("device") not in {"cpu", "cuda"}:
            raise ProductionBindingError("local BGE probe device is unresolved")
    else:
        _must_equal(runtime_probe, None, "scoring local BGE runtime probe")

    transports = _mapping(
        attestation.get("transport_bindings"), "transport bindings"
    )
    expected_roles = {"extraction"} if stage == "retrieval" else {"responder", "judge"}
    _exact_keys(transports, expected_roles, "transport bindings")
    for role in expected_roles:
        transport = _mapping(transports[role], f"{role} transport binding")
        _exact_keys(
            transport,
            {
                "role",
                "model",
                "revision",
                "model_identity_sha256",
                "concrete_transport_type",
                "transport_implementation_sha256",
                "cap_boundary",
                "authorized_http_attempts",
                "attempted_http_attempts_before_run",
                "retries_authorized",
                "external_http_attempts_certified",
                "external_provider_persistence_certified",
            },
            f"{role} transport binding",
        )
        for field, expected in {
            "role": role,
            "cap_boundary": "concrete_http_send",
            "attempted_http_attempts_before_run": 0,
            "retries_authorized": 0,
            "external_http_attempts_certified": True,
            "external_provider_persistence_certified": False,
        }.items():
            _must_equal(transport.get(field), expected, f"{role} transport {field}")
        authorized = transport.get("authorized_http_attempts")
        if isinstance(authorized, bool) or not isinstance(authorized, int) or authorized < 1:
            raise ProductionBindingError(
                f"{role} transport authorized_http_attempts is invalid"
            )
        for field in ("model", "revision", "concrete_transport_type"):
            if not isinstance(transport.get(field), str) or not transport[field].strip():
                raise ProductionBindingError(f"{role} transport {field} is empty")
        for field in ("model_identity_sha256", "transport_implementation_sha256"):
            observed = transport.get(field)
            if not isinstance(observed, str) or len(observed) != 64:
                raise ProductionBindingError(f"{role} transport {field} is not SHA-256")

    _must_equal(
        receipt.get("trusted_runtime_binding_receipt_sha256"),
        canonical_json_sha256(attestation),
        "trusted runtime binding receipt SHA-256",
    )
    return receipt


class FrozenMem0RetrievalLauncher:
    """Exact Stage-A launcher type; closed until issuance/report integration."""

    __slots__ = ()

    def __init_subclass__(cls, **_kwargs: Any) -> None:
        raise TypeError("FrozenMem0RetrievalLauncher cannot be subclassed")

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise ProductionBindingBlocked(
            "Stage-A production binding is closed: the exact extraction route "
            "and adapter exist, but trusted issuance and post-run closure are "
            "not integrated with the report boundary"
        )


class FrozenMem0ScoringLauncher:
    """Exact Stage-B launcher type; construction is closed until transports exist."""

    __slots__ = ()

    def __init_subclass__(cls, **_kwargs: Any) -> None:
        raise TypeError("FrozenMem0ScoringLauncher cannot be subclassed")

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise ProductionBindingBlocked(
            "Stage-B production binding is closed: exact hard-capped Terra and "
            "Sol send-boundary transports have not been implemented"
        )


def _consume_trusted_runtime_claim(
    launcher: Any,
    *,
    stage: str,
    authorization_sha256: str,
    bound_callables: Sequence[Any],
) -> Mapping[str, Any]:
    """Runner-only issuer hook; currently no launcher can own a valid claim."""

    expected = (
        FrozenMem0RetrievalLauncher
        if stage == "retrieval"
        else FrozenMem0ScoringLauncher
    )
    if type(launcher) is not expected:
        raise ProductionBindingError(
            f"{stage} binding issuer must be exact {expected.__name__}"
        )
    # ``object.__new__`` can allocate a slotless shell, but it cannot create a
    # verified launcher because both exact constructors fail while the concrete
    # transport slots are unresolved.  Keep this terminal guard even after the
    # constructors are implemented; it must then be replaced by consumption of
    # a one-use, internally rechecked claim.
    del authorization_sha256, bound_callables
    raise ProductionBindingBlocked(
        f"{stage} production binding issuance is closed by unresolved transports"
    )


def _recheck_trusted_runtime_claim(
    launcher: Any,
    *,
    stage: str,
    authorization_sha256: str,
    bound_callables: Sequence[Any],
    receipt: Mapping[str, Any],
) -> None:
    """Runner-only live recheck; unreachable until a concrete claim exists."""

    del authorization_sha256, bound_callables, receipt
    expected = (
        FrozenMem0RetrievalLauncher
        if stage == "retrieval"
        else FrozenMem0ScoringLauncher
    )
    if type(launcher) is not expected:
        raise ProductionBindingError("trusted runtime launcher type changed")
    raise ProductionBindingBlocked(
        f"{stage} production binding recheck is closed by unresolved transports"
    )


__all__ = [
    "ExactMem0AdapterFactory",
    "FrozenArtifactBindingReceipt",
    "FrozenMem0RetrievalLauncher",
    "FrozenMem0ScoringLauncher",
    "HardTransportAttemptCap",
    "InjectedHardCappedExtractionTransport",
    "InjectedHardCappedJudgeTransport",
    "InjectedHardCappedResponderTransport",
    "LOCAL_BGE_PROBE_FORMAT",
    "LiteLLMTerraExtractionTransport",
    "PRODUCTION_BINDING_FORMAT",
    "ProductionBindingBlocked",
    "ProductionBindingError",
    "TransportAttemptLimitExceeded",
    "probe_local_bge_m3_runtime",
    "production_binding_readiness",
    "run_mem0_factory_canary",
    "run_mem0_factory_presend_cleanup_canary",
    "run_terra_extraction_canary",
    "validate_local_bge_m3_contract",
    "validate_production_mem0_config",
    "verify_frozen_artifact_binding",
]
