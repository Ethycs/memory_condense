"""Concrete, fail-closed production-binding seam for the Mem0 campaign.

This module is intentionally provider-free: importing it installs no package,
loads no model, opens no socket, and makes no provider call.  It does define
the exact checks a future production launcher must pass before the injectable
runner can receive a :class:`~tools.mem0_eval.run_shard.TrustedRuntimeBinding`.

The extraction provider/model and its concrete send-boundary transport are not
yet frozen.  Consequently no positive binding can currently be issued.  This
is a deliberate safety property, not a stub that falls back to an injected
callable.  The responder and judge model names are frozen, but their concrete
single-attempt transports are likewise not implemented in this repository.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import re
import socket
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Sequence

from .source_compat import (
    BGE_M3_CHECKPOINT_SHA256,
    DEFAULT_MODEL_DIM,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    verify_bge_m3_checkpoint,
)

from .preflight import tool_implementation_sha256
from .run_shard import (
    ProviderCallResult,
    RetrievalStageAuthorization,
    ScoringStageAuthorization,
    TrustedRuntimeBinding,
    canonical_json_sha256,
)


PRODUCTION_BINDING_FORMAT = "memory-condense-mem0-production-binding-v1"
PRODUCTION_BINDING_KIND = "frozen_concrete_production_launcher_v1"
LOCAL_BGE_PROBE_FORMAT = "memory-condense-local-bge-m3-runtime-probe-v1"
PROVIDER_FREE_STATUS_FORMAT = "memory-condense-mem0-production-readiness-v1"

# These are code-owned slots, not configuration inputs.  A later commit must
# implement and freeze exact concrete send-boundary types before replacing
# ``None``.  Policy JSON or an injected object can never populate these slots.
_EXACT_EXTRACTION_TRANSPORT_TYPE: type[Any] | None = None
_EXACT_MEM0_ADAPTER_FACTORY_TYPE: type[Any] | None = None
_EXACT_RESPONDER_TRANSPORT_TYPE: type[Any] | None = None
_EXACT_JUDGE_TRANSPORT_TYPE: type[Any] | None = None

_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "LITELLM_LOCAL_MODEL_COST_MAP": "true",
    "MEM0_TELEMETRY": "false",
}
_NETWORK_PROBE_LOCK = threading.Lock()
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

    The extraction provider is unresolved, so the only currently admissible
    redacted LLM config field is its exact model identifier.  A future provider
    integration must change this code-owned allowlist deliberately; arbitrary
    policy keys such as retries, temperature, base URLs, or proxy settings are
    never inherited into production trust.
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
                "local BGE-M3 probe attempted forbidden network access"
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
        self._lock = threading.Lock()

    def call(self, send_once: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if not callable(send_once):
            raise TypeError("transport send boundary must be callable")
        with self._lock:
            if self._attempted >= self._authorized:
                self._rejected += 1
                raise TransportAttemptLimitExceeded(
                    f"{self._role} transport attempt authorization exhausted"
                )
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
                "code": "extraction_provider_model_and_transport_unresolved",
                "detail": (
                    "freeze one exact extraction provider/model/revision and a "
                    "zero-retry hard cap at its concrete HTTP send boundary"
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
                "code": "actual_mem0_embedder_instance_probe_unimplemented",
                "detail": (
                    "bind the verified checkpoint/config to the exact embedder "
                    "instance owned by Memory.from_config before its first add"
                ),
            },
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
    """Exact Stage-A launcher type; construction is closed until selection."""

    __slots__ = ()

    def __init_subclass__(cls, **_kwargs: Any) -> None:
        raise TypeError("FrozenMem0RetrievalLauncher cannot be subclassed")

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise ProductionBindingBlocked(
            "Stage-A production binding is closed: the exact extraction "
            "provider/model, concrete send transport, and bound adapter factory "
            "have not been frozen"
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
    "FrozenArtifactBindingReceipt",
    "FrozenMem0RetrievalLauncher",
    "FrozenMem0ScoringLauncher",
    "HardTransportAttemptCap",
    "InjectedHardCappedExtractionTransport",
    "InjectedHardCappedJudgeTransport",
    "InjectedHardCappedResponderTransport",
    "LOCAL_BGE_PROBE_FORMAT",
    "PRODUCTION_BINDING_FORMAT",
    "ProductionBindingBlocked",
    "ProductionBindingError",
    "TransportAttemptLimitExceeded",
    "probe_local_bge_m3_runtime",
    "production_binding_readiness",
    "validate_local_bge_m3_contract",
    "validate_production_mem0_config",
    "verify_frozen_artifact_binding",
]
