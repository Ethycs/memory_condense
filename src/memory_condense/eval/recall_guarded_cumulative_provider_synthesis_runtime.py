"""Provider completion plus pinned local scoring for cumulative synthesis.

Only completion crosses the controlled OpenAI-compatible gateway.  Episodic
candidate scoring remains on the exact local Qwen checkpoint owned by one
``RecallGuardedCumulativeSynthesisRuntime``.  API keys are constructor-only
inputs and never enter adapter state, identities, reports, or log messages.
Optional content-addressed request/response journals make paid completions
replayable and reserve authorization before a physical gateway call.
"""

from __future__ import annotations

import json
import math
import os
import re
import ssl
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.domain.schemas import RetrievalResult
from memory_condense.eval._artifact_json import (
    canonical_json_bytes as _canonical_json_bytes,
)
from memory_condense.eval.recall_guarded_cumulative_synthesis_runtime import (
    RecallGuardedCumulativeSynthesisRuntime,
)
from memory_condense.search.selectors.causal_choice_scorer import (
    CausalChoiceEvidence,
    CausalChoiceScoreReport,
)


PROVIDER_RUNTIME_FORMAT = (
    "memory-condense-recall-guarded-provider-synthesis-runtime-v1"
)
CENTRAL_DEV_GATEWAY_URL = "https://central-dev.zt:4000/v1"
DEFAULT_CALLER_MODEL = "openai/codex_sdk/gpt-5.6-terra"
CALL_REQUEST_JOURNAL_FORMAT = (
    "memory-condense-provider-synthesis-call-request-v1"
)
CALL_RESPONSE_JOURNAL_FORMAT = (
    "memory-condense-provider-synthesis-call-response-v1"
)
_JOURNAL_NAME = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SECRET_FIELD_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "password",
        "refresh_token",
        "secret",
        "token",
    }
)
_SECRET_FIELD_SUFFIXES = (
    "_api_key",
    "_credential",
    "_credentials",
    "_password",
    "_secret",
    "_token",
)


def _plain_campaign_value(value: Any, *, api_key: str, path: str) -> Any:
    """Copy one finite JSON value while refusing credential-shaped content."""

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return value
    if isinstance(value, str):
        if value == api_key or (len(api_key) >= 8 and api_key in value):
            raise ValueError("campaign_binding must not contain the API key")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            normalized_key = key.strip().casefold().replace("-", "_")
            if normalized_key in _SECRET_FIELD_NAMES or normalized_key.endswith(
                _SECRET_FIELD_SUFFIXES
            ):
                raise ValueError(
                    "campaign_binding must not contain credential fields"
                )
            copied[key] = _plain_campaign_value(
                child,
                api_key=api_key,
                path=f"{path}.{key}",
            )
        return copied
    if isinstance(value, (list, tuple)):
        return [
            _plain_campaign_value(
                child,
                api_key=api_key,
                path=f"{path}[{index}]",
            )
            for index, child in enumerate(value)
        ]
    raise TypeError(f"{path} must contain only plain JSON values")


def _campaign_binding(
    value: Mapping[str, Any] | None,
    *,
    api_key: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("campaign_binding must be a mapping")
    copied = _plain_campaign_value(
        value,
        api_key=api_key,
        path="campaign_binding",
    )
    assert isinstance(copied, dict)
    return copied


def _sealed_journal(body: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(body)
    payload["journal_sha256"] = identity_sha256(dict(body))
    return payload


def _atomic_publish_journal(path: Path, body: Mapping[str, Any]) -> str:
    """Atomically publish one canonical, self-sealed journal without overwrite."""

    payload = _sealed_journal(body)
    encoded = _canonical_json_bytes(payload)
    if path.exists():
        raise FileExistsError(f"refusing to replace call journal: {path}")
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(payload["journal_sha256"])


def _read_journal(path: Path) -> tuple[dict[str, Any], str]:
    """Read one canonical self-sealed journal and reject byte/body tampering."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"call journal must be a regular file: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"call journal is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"call journal must be a JSON object: {path}")
    try:
        canonical = _canonical_json_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"call journal is not canonical JSON: {path}") from exc
    if raw != canonical:
        raise ValueError(f"call journal is not canonical JSON: {path}")
    declared = payload.get("journal_sha256")
    body = dict(payload)
    body.pop("journal_sha256", None)
    if not isinstance(declared, str) or not _DIGEST.fullmatch(declared):
        raise ValueError(f"call journal has no valid receipt: {path}")
    if identity_sha256(body) != declared:
        raise ValueError(f"call journal receipt does not verify: {path}")
    return payload, declared


@contextmanager
def _checkpoint_lock(root: Path):
    """Serialize check/reserve/call/publish across conforming processes."""

    lock_path = root / ".provider-synthesis-journal.lock"
    with lock_path.open("a+b") as handle:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _dump(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): child for key, child in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return {str(key): child for key, child in dumped.items()}
    raise TypeError("runtime identity must expose a mapping model_dump")


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _reported_count(usage: Any, name: str) -> tuple[int, bool]:
    if usage is None:
        return 0, False
    value = _field(usage, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0, False
    count = int(value)
    # Some gateway routes synthesize an all-zero usage object when upstream
    # token accounting is unavailable.  Preserve zero as the numeric value,
    # but never misrepresent it as an observed count.
    if count <= 0:
        return 0, False
    return count, True


def _gateway_model(caller_model: str) -> str:
    """Strip exactly one leading LiteLLM provider namespace."""

    return (
        caller_model[len("openai/") :]
        if caller_model.startswith("openai/")
        else caller_model
    )


def _new_gateway_client(api_key: str) -> Any:
    """Construct the zero-retry internal-CA OpenAI client."""

    import httpx
    import truststore
    from openai import OpenAI

    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return OpenAI(
        api_key=api_key,
        base_url=CENTRAL_DEV_GATEWAY_URL,
        http_client=httpx.Client(verify=ssl_context),
        max_retries=0,
    )


def _local_scorer_identity(runtime: Any) -> dict[str, Any]:
    """Snapshot the exact local score-provider identity and bounds."""

    runtime_identity = _dump(runtime.identity)
    scorer = getattr(runtime, "_scorer", None)
    if scorer is None:
        raise TypeError("local synthesis runtime does not expose its scorer")
    choice_ids = tuple(
        tuple(int(token) for token in row)
        for row in getattr(scorer, "_choice_ids", ())
    )
    choices = tuple(str(value) for value in getattr(scorer, "_choices", ()))
    if len(choice_ids) != 2 or len(choices) != 2:
        raise ValueError("local scorer has an unexpected choice contract")
    scorer_payload = {
        "model_id": str(scorer.model_id),
        "model_revision": str(scorer.model_revision),
        "checkpoint_sha256": str(scorer.checkpoint_sha256),
        "runtime": (
            f"{type(scorer.model).__module__}.{type(scorer.model).__name__}"
        ),
        "device": str(scorer.device),
        "dtype": str(scorer.dtype_name),
        "max_candidates": int(scorer.max_candidates),
        "requested_batch_size": int(scorer.requested_batch_size),
        "effective_batch_size": int(scorer.batch_size),
        "query_tokens": int(scorer.query_tokens),
        "candidate_tokens": int(scorer.candidate_tokens),
        "max_prompt_tokens": int(scorer.max_prompt_tokens),
        "max_workspace_tokens": int(scorer.max_workspace_tokens),
        "choices": list(choices),
        "choice_token_ids": [list(row) for row in choice_ids],
        "single_token_labels": all(len(row) == 1 for row in choice_ids),
        "strict": bool(scorer.strict),
        "generation": False,
        "kv_cache": False,
    }
    return {
        "runtime_identity": runtime_identity,
        "score_provider": scorer_payload,
        "identity_sha256": identity_sha256(
            {
                "runtime_identity": runtime_identity,
                "score_provider": scorer_payload,
            }
        ),
    }


@dataclass(frozen=True, slots=True)
class ProviderSynthesisRuntimeIdentity:
    """Gateway identity plus the exact nested local scorer configuration."""

    format: str
    gateway_url: str
    caller_model: str
    gateway_model: str
    default_max_new_tokens: int
    retries: int
    temperature: float | None
    local_scorer: Mapping[str, Any]
    campaign_binding: Mapping[str, Any]
    campaign_binding_sha256: str
    call_request_journal_format: str
    call_response_journal_format: str

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProviderSynthesisCompletionReport:
    """Text-free provenance, provider metadata, and usage for one call.

    ``physical_call`` describes this runtime session's invocation.  A cache
    hit can therefore restore one historically journaled unique response
    while correctly reporting ``physical_call=False`` for the current session.
    """

    gateway_url: str
    caller_model: str
    gateway_model: str
    call_key_sha256: str
    runtime_identity_sha256: str
    campaign_binding_sha256: str
    request_journal_sha256: str
    messages_sha256: str
    completion_sha256: str
    response_id: str
    response_model: str
    finish_reason: str
    max_new_tokens: int
    reported_usage_available: bool
    reported_input_tokens: int
    reported_output_tokens: int
    reported_total_tokens: int
    reported_input_tokens_available: bool
    reported_output_tokens_available: bool
    reported_total_tokens_available: bool
    input_token_proxy: int
    output_token_proxy: int
    elapsed_s: float
    retries: int
    cache_hit: bool
    physical_call: bool
    cumulative_completion_calls: int
    cumulative_logical_completion_calls: int
    cumulative_unique_completion_calls: int
    cumulative_physical_completion_calls: int
    cumulative_checkpoint_hits: int

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _JournalRecord:
    call_key_sha256: str
    request_journal_sha256: str
    completion: str
    report: ProviderSynthesisCompletionReport


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


def _require_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_request_journal(
    path: Path,
    payload: Mapping[str, Any],
    receipt: str,
) -> dict[str, Any]:
    _require_exact_keys(
        payload,
        {
            "format",
            "call_key_sha256",
            "call_key_payload",
            "runtime_identity",
            "campaign_binding",
            "journal_sha256",
        },
        label="call request journal",
    )
    if payload["format"] != CALL_REQUEST_JOURNAL_FORMAT:
        raise ValueError(f"unexpected call request journal format: {path}")
    call_key = _require_digest(
        payload["call_key_sha256"],
        label="call key",
    )
    match = _JOURNAL_NAME.fullmatch(path.name)
    if match is None or match.group("kind") != "request":
        raise ValueError(f"invalid call request journal name: {path}")
    if match.group("key") != call_key:
        raise ValueError(f"call request journal filename changed: {path}")
    key_payload = payload["call_key_payload"]
    runtime_identity = payload["runtime_identity"]
    campaign_binding = payload["campaign_binding"]
    if not isinstance(key_payload, Mapping):
        raise ValueError(f"call key payload must be an object: {path}")
    if not isinstance(runtime_identity, Mapping):
        raise ValueError(f"runtime identity must be an object: {path}")
    if not isinstance(campaign_binding, Mapping):
        raise ValueError(f"campaign binding must be an object: {path}")
    _require_exact_keys(
        key_payload,
        {
            "messages_sha256",
            "runtime_identity_sha256",
            "max_new_tokens",
            "campaign_binding_sha256",
        },
        label="call key payload",
    )
    _require_digest(key_payload["messages_sha256"], label="messages SHA-256")
    runtime_sha = _require_digest(
        key_payload["runtime_identity_sha256"],
        label="runtime identity SHA-256",
    )
    campaign_sha = _require_digest(
        key_payload["campaign_binding_sha256"],
        label="campaign binding SHA-256",
    )
    max_new_tokens = key_payload["max_new_tokens"]
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 1
    ):
        raise ValueError(f"call max_new_tokens is invalid: {path}")
    if identity_sha256(dict(runtime_identity)) != runtime_sha:
        raise ValueError(f"call runtime identity does not verify: {path}")
    if identity_sha256(dict(campaign_binding)) != campaign_sha:
        raise ValueError(f"call campaign binding does not verify: {path}")
    if identity_sha256(dict(key_payload)) != call_key:
        raise ValueError(f"call key does not verify: {path}")
    if payload["journal_sha256"] != receipt:
        raise ValueError(f"call request journal receipt changed: {path}")
    return dict(payload)


def _validate_stored_report(
    report_value: Any,
    *,
    request: Mapping[str, Any],
    request_receipt: str,
    completion: str,
    path: Path,
) -> ProviderSynthesisCompletionReport:
    if not isinstance(report_value, Mapping):
        raise ValueError(f"stored completion report must be an object: {path}")
    expected_fields = {
        field.name for field in fields(ProviderSynthesisCompletionReport)
    }
    _require_exact_keys(
        report_value,
        expected_fields,
        label="stored completion report",
    )
    try:
        report = ProviderSynthesisCompletionReport(**dict(report_value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stored completion report is invalid: {path}") from exc
    key_payload = request["call_key_payload"]
    runtime_identity = request["runtime_identity"]
    boolean_fields = (
        report.reported_usage_available,
        report.reported_input_tokens_available,
        report.reported_output_tokens_available,
        report.reported_total_tokens_available,
        report.cache_hit,
        report.physical_call,
    )
    if any(type(value) is not bool for value in boolean_fields):
        raise ValueError(f"stored completion report booleans changed: {path}")
    counts = (
        report.reported_input_tokens,
        report.reported_output_tokens,
        report.reported_total_tokens,
        report.input_token_proxy,
        report.output_token_proxy,
        report.cumulative_completion_calls,
        report.cumulative_logical_completion_calls,
        report.cumulative_unique_completion_calls,
        report.cumulative_physical_completion_calls,
        report.cumulative_checkpoint_hits,
    )
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError(f"stored completion report counters changed: {path}")
    availability = (
        report.reported_input_tokens_available,
        report.reported_output_tokens_available,
        report.reported_total_tokens_available,
    )
    reported = (
        report.reported_input_tokens,
        report.reported_output_tokens,
        report.reported_total_tokens,
    )
    if tuple(value > 0 for value in reported) != availability:
        raise ValueError(f"stored provider usage availability changed: {path}")
    if report.reported_usage_available != any(availability):
        raise ValueError(f"stored provider usage summary changed: {path}")
    if (
        report.call_key_sha256 != request["call_key_sha256"]
        or report.runtime_identity_sha256
        != key_payload["runtime_identity_sha256"]
        or report.campaign_binding_sha256
        != key_payload["campaign_binding_sha256"]
        or report.request_journal_sha256 != request_receipt
        or report.messages_sha256 != key_payload["messages_sha256"]
        or report.max_new_tokens != key_payload["max_new_tokens"]
        or report.gateway_url != runtime_identity.get("gateway_url")
        or report.caller_model != runtime_identity.get("caller_model")
        or report.gateway_model != runtime_identity.get("gateway_model")
        or report.completion_sha256 != quote_sha256(completion)
        or report.output_token_proxy != count_tokens(completion)
        or report.retries != 0
        or report.cache_hit is not False
        or report.physical_call is not True
    ):
        raise ValueError(f"stored completion report binding changed: {path}")
    if (
        isinstance(report.elapsed_s, bool)
        or not isinstance(report.elapsed_s, (int, float))
        or not math.isfinite(float(report.elapsed_s))
        or report.elapsed_s < 0
    ):
        raise ValueError(f"stored completion elapsed time changed: {path}")
    if (
        report.cumulative_completion_calls < 1
        or report.cumulative_logical_completion_calls < 1
        or report.cumulative_unique_completion_calls < 1
        or report.cumulative_physical_completion_calls < 1
        or report.cumulative_completion_calls
        != report.cumulative_logical_completion_calls
    ):
        raise ValueError(f"stored completion cumulative counts changed: {path}")
    return report


def _validate_response_journal(
    path: Path,
    payload: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    request_receipt: str,
) -> _JournalRecord:
    _require_exact_keys(
        payload,
        {
            "format",
            "call_key_sha256",
            "request_journal_sha256",
            "completion",
            "report",
            "journal_sha256",
        },
        label="call response journal",
    )
    if payload["format"] != CALL_RESPONSE_JOURNAL_FORMAT:
        raise ValueError(f"unexpected call response journal format: {path}")
    call_key = _require_digest(payload["call_key_sha256"], label="call key")
    match = _JOURNAL_NAME.fullmatch(path.name)
    if match is None or match.group("kind") != "response":
        raise ValueError(f"invalid call response journal name: {path}")
    if match.group("key") != call_key or call_key != request["call_key_sha256"]:
        raise ValueError(f"call response journal filename changed: {path}")
    if payload["request_journal_sha256"] != request_receipt:
        raise ValueError(f"call response request binding changed: {path}")
    completion = payload["completion"]
    if (
        not isinstance(completion, str)
        or not completion
        or completion.strip() != completion
    ):
        raise ValueError(f"stored completion text is invalid: {path}")
    report = _validate_stored_report(
        payload["report"],
        request=request,
        request_receipt=request_receipt,
        completion=completion,
        path=path,
    )
    return _JournalRecord(
        call_key_sha256=call_key,
        request_journal_sha256=request_receipt,
        completion=completion,
        report=report,
    )


def _load_checkpoint_journal(root: Path) -> dict[str, _JournalRecord]:
    """Verify the complete published request/response inventory."""

    request_paths: dict[str, Path] = {}
    response_paths: dict[str, Path] = {}
    for path in root.glob("*.json"):
        match = _JOURNAL_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected JSON file in call journal: {path}")
        target = request_paths if match.group("kind") == "request" else response_paths
        target[match.group("key")] = path
    orphan_responses = set(response_paths) - set(request_paths)
    incomplete_requests = set(request_paths) - set(response_paths)
    if orphan_responses:
        key = min(orphan_responses)
        raise ValueError(f"call response has no request reservation: {key}")
    if incomplete_requests:
        key = min(incomplete_requests)
        raise RuntimeError(
            "call request has no response; refusing an unsafe retry: " + key
        )
    records: dict[str, _JournalRecord] = {}
    for key in sorted(request_paths):
        request_payload, request_receipt = _read_journal(request_paths[key])
        request = _validate_request_journal(
            request_paths[key],
            request_payload,
            request_receipt,
        )
        response_payload, _response_receipt = _read_journal(response_paths[key])
        records[key] = _validate_response_journal(
            response_paths[key],
            response_payload,
            request=request,
            request_receipt=request_receipt,
        )
    return records


class RecallGuardedCumulativeProviderSynthesisRuntime:
    """Use central-dev for completion and one pinned local runtime for scores."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        api_key: str,
        caller_model: str = DEFAULT_CALLER_MODEL,
        max_new_tokens: int = 2048,
        client: Any | None = None,
        local_runtime: RecallGuardedCumulativeSynthesisRuntime | None = None,
        checkpoint_dir: str | Path | None = None,
        campaign_binding: Mapping[str, Any] | None = None,
        authorized_completion_calls: int | None = None,
        gpu_memory: str = "6GiB",
        cpu_memory: str = "24GiB",
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be supplied explicitly")
        secret = api_key.strip()
        selected_model = str(caller_model).strip()
        if not selected_model:
            raise ValueError("caller_model must be non-empty")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be positive")
        if (
            authorized_completion_calls is not None
            and (
                isinstance(authorized_completion_calls, bool)
                or not isinstance(authorized_completion_calls, int)
                or authorized_completion_calls < 0
            )
        ):
            raise ValueError(
                "authorized_completion_calls must be a nonnegative integer"
            )
        if local_runtime is None and model_dir is None:
            raise ValueError("model_dir is required when local_runtime is omitted")
        if local_runtime is not None and model_dir is not None:
            raise ValueError("supply model_dir or local_runtime, not both")
        binding = _campaign_binding(campaign_binding, api_key=secret)
        binding_sha256 = identity_sha256(binding)
        checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if checkpoint_root is not None:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
                raise ValueError("checkpoint_dir must be a regular directory")

        scorer_runtime = local_runtime
        try:
            if scorer_runtime is None:
                scorer_runtime = RecallGuardedCumulativeSynthesisRuntime(
                    Path(model_dir),
                    max_new_tokens=max_new_tokens,
                    gpu_memory=gpu_memory,
                    cpu_memory=cpu_memory,
                )
            local_identity = _local_scorer_identity(scorer_runtime)
        except BaseException:
            if scorer_runtime is not None:
                scorer_runtime.close()
            raise

        sent_model = _gateway_model(selected_model)
        temperature = None if "codex_sdk/" in sent_model else 0.0
        identity = ProviderSynthesisRuntimeIdentity(
            format=PROVIDER_RUNTIME_FORMAT,
            gateway_url=CENTRAL_DEV_GATEWAY_URL,
            caller_model=selected_model,
            gateway_model=sent_model,
            default_max_new_tokens=int(max_new_tokens),
            retries=0,
            temperature=temperature,
            local_scorer=local_identity,
            campaign_binding=binding,
            campaign_binding_sha256=binding_sha256,
            call_request_journal_format=CALL_REQUEST_JOURNAL_FORMAT,
            call_response_journal_format=CALL_RESPONSE_JOURNAL_FORMAT,
        )
        runtime_identity_sha256 = identity_sha256(identity.model_dump())
        try:
            if checkpoint_root is not None:
                with _checkpoint_lock(checkpoint_root):
                    existing = _load_checkpoint_journal(checkpoint_root)
                if (
                    authorized_completion_calls is not None
                    and len(existing) > authorized_completion_calls
                ):
                    raise RuntimeError(
                        "verified call journal exceeds the authorized "
                        "completion-call budget"
                    )
            provider_client = (
                client if client is not None else _new_gateway_client(secret)
            )
        except BaseException:
            scorer_runtime.close()
            raise

        self._local_runtime = scorer_runtime
        self._client = provider_client
        self._lock = threading.Lock()
        self._closed = False
        self._default_max_new_tokens = int(max_new_tokens)
        self._checkpoint_dir = checkpoint_root
        self._authorized_completion_calls = authorized_completion_calls
        self._runtime_identity_sha256 = runtime_identity_sha256
        self._memory_journal: dict[str, _JournalRecord] = {}
        self._memory_pending: set[str] = set()
        self._accounted_call_keys: set[str] = set()
        self._completion_logical_calls = 0
        self._completion_unique_calls = 0
        self._completion_physical_calls = 0
        self._completion_checkpoint_hits = 0
        self._completion_reported_input_tokens = 0
        self._completion_reported_output_tokens = 0
        self._completion_reported_total_tokens = 0
        self._completion_input_token_proxy = 0
        self._completion_output_token_proxy = 0
        self._completion_elapsed_s = 0.0
        self.last_completion_report: ProviderSynthesisCompletionReport | None = None
        self.identity = identity

    @staticmethod
    def _validated_messages(
        messages: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        if not messages:
            raise ValueError("completion messages must not be empty")
        normalized: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            role = str(message.get("role", "")).strip()
            content = message.get("content")
            if not role or not isinstance(content, str):
                raise ValueError(
                    f"completion message {index} requires role and string content"
                )
            normalized.append({"role": role, "content": content})
        return normalized

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("provider synthesis runtime is closed")

    def _call_key_payload(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        return {
            "messages_sha256": identity_sha256(list(messages)),
            "runtime_identity_sha256": self._runtime_identity_sha256,
            "max_new_tokens": int(max_new_tokens),
            "campaign_binding_sha256": self.identity.campaign_binding_sha256,
        }

    def _request_journal_body(
        self,
        *,
        call_key: str,
        key_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "format": CALL_REQUEST_JOURNAL_FORMAT,
            "call_key_sha256": call_key,
            "call_key_payload": dict(key_payload),
            "runtime_identity": self.identity.model_dump(),
            "campaign_binding": dict(self.identity.campaign_binding),
        }

    def _invoke_provider(
        self,
        normalized: list[dict[str, str]],
        *,
        requested: int,
        call_key: str,
        request_journal_sha256: str,
    ) -> _JournalRecord:
        request: dict[str, Any] = {
            "model": self.identity.gateway_model,
            "messages": normalized,
            "max_tokens": requested,
        }
        if self.identity.temperature is not None:
            request["temperature"] = self.identity.temperature
        started = time.perf_counter()
        self._completion_physical_calls += 1
        response = self._client.chat.completions.create(**request)
        elapsed = time.perf_counter() - started

        choices = _field(response, "choices", ())
        if not choices:
            raise RuntimeError("provider returned no completion choices")
        choice = choices[0]
        message = _field(choice, "message")
        completion = str(_field(message, "content", "") or "").strip()
        if not completion:
            raise RuntimeError("provider returned no answer text")

        usage = _field(response, "usage")
        input_tokens, input_available = _reported_count(usage, "prompt_tokens")
        output_tokens, output_available = _reported_count(
            usage,
            "completion_tokens",
        )
        total_tokens, total_available = _reported_count(usage, "total_tokens")
        input_token_proxy = count_chat_prompt_token_proxy(normalized)
        output_token_proxy = count_tokens(completion)
        next_unique = self._completion_unique_calls + int(
            call_key not in self._accounted_call_keys
        )
        report = ProviderSynthesisCompletionReport(
            gateway_url=self.identity.gateway_url,
            caller_model=self.identity.caller_model,
            gateway_model=self.identity.gateway_model,
            call_key_sha256=call_key,
            runtime_identity_sha256=self._runtime_identity_sha256,
            campaign_binding_sha256=self.identity.campaign_binding_sha256,
            request_journal_sha256=request_journal_sha256,
            messages_sha256=identity_sha256(normalized),
            completion_sha256=quote_sha256(completion),
            response_id=str(_field(response, "id", "") or ""),
            response_model=str(_field(response, "model", "") or ""),
            finish_reason=str(_field(choice, "finish_reason", "") or ""),
            max_new_tokens=requested,
            reported_usage_available=bool(
                input_available or output_available or total_available
            ),
            reported_input_tokens=input_tokens,
            reported_output_tokens=output_tokens,
            reported_total_tokens=total_tokens,
            reported_input_tokens_available=input_available,
            reported_output_tokens_available=output_available,
            reported_total_tokens_available=total_available,
            input_token_proxy=input_token_proxy,
            output_token_proxy=output_token_proxy,
            elapsed_s=elapsed,
            retries=0,
            cache_hit=False,
            physical_call=True,
            cumulative_completion_calls=self._completion_logical_calls + 1,
            cumulative_logical_completion_calls=(
                self._completion_logical_calls + 1
            ),
            cumulative_unique_completion_calls=next_unique,
            cumulative_physical_completion_calls=self._completion_physical_calls,
            cumulative_checkpoint_hits=self._completion_checkpoint_hits,
        )
        return _JournalRecord(
            call_key_sha256=call_key,
            request_journal_sha256=request_journal_sha256,
            completion=completion,
            report=report,
        )

    @staticmethod
    def _response_journal_body(record: _JournalRecord) -> dict[str, Any]:
        return {
            "format": CALL_RESPONSE_JOURNAL_FORMAT,
            "call_key_sha256": record.call_key_sha256,
            "request_journal_sha256": record.request_journal_sha256,
            "completion": record.completion,
            "report": record.report.model_dump(),
        }

    def _accept_completion(
        self,
        record: _JournalRecord,
        *,
        cache_hit: bool,
    ) -> str:
        self._completion_logical_calls += 1
        if cache_hit:
            self._completion_checkpoint_hits += 1
        if record.call_key_sha256 not in self._accounted_call_keys:
            self._accounted_call_keys.add(record.call_key_sha256)
            self._completion_unique_calls += 1
            report = record.report
            self._completion_reported_input_tokens += (
                report.reported_input_tokens
            )
            self._completion_reported_output_tokens += (
                report.reported_output_tokens
            )
            self._completion_reported_total_tokens += (
                report.reported_total_tokens
            )
            self._completion_input_token_proxy += report.input_token_proxy
            self._completion_output_token_proxy += report.output_token_proxy
            self._completion_elapsed_s += report.elapsed_s
        self.last_completion_report = replace(
            record.report,
            cache_hit=cache_hit,
            physical_call=not cache_hit,
            cumulative_completion_calls=self._completion_logical_calls,
            cumulative_logical_completion_calls=self._completion_logical_calls,
            cumulative_unique_completion_calls=self._completion_unique_calls,
            cumulative_physical_completion_calls=(
                self._completion_physical_calls
            ),
            cumulative_checkpoint_hits=self._completion_checkpoint_hits,
        )
        return record.completion

    def _complete_in_memory(
        self,
        normalized: list[dict[str, str]],
        *,
        requested: int,
        call_key: str,
        key_payload: Mapping[str, Any],
    ) -> str:
        cached = self._memory_journal.get(call_key)
        if cached is not None:
            return self._accept_completion(cached, cache_hit=True)
        pending = self._memory_pending
        if call_key in pending:
            raise RuntimeError(
                "call request has no response; refusing an unsafe retry: "
                + call_key
            )
        occupied = len(self._memory_journal) + len(pending)
        if (
            self._authorized_completion_calls is not None
            and occupied >= self._authorized_completion_calls
        ):
            raise RuntimeError("authorized unique completion-call budget exhausted")
        request_body = self._request_journal_body(
            call_key=call_key,
            key_payload=key_payload,
        )
        request_receipt = identity_sha256(request_body)
        pending.add(call_key)
        record = self._invoke_provider(
            normalized,
            requested=requested,
            call_key=call_key,
            request_journal_sha256=request_receipt,
        )
        self._memory_journal[call_key] = record
        pending.remove(call_key)
        return self._accept_completion(record, cache_hit=False)

    def _complete_checkpointed(
        self,
        normalized: list[dict[str, str]],
        *,
        requested: int,
        call_key: str,
        key_payload: Mapping[str, Any],
    ) -> str:
        root = self._checkpoint_dir
        assert root is not None
        with _checkpoint_lock(root):
            records = _load_checkpoint_journal(root)
            cached = records.get(call_key)
            if cached is not None:
                return self._accept_completion(cached, cache_hit=True)
            if (
                self._authorized_completion_calls is not None
                and len(records) >= self._authorized_completion_calls
            ):
                raise RuntimeError(
                    "authorized unique completion-call budget exhausted"
                )
            request_body = self._request_journal_body(
                call_key=call_key,
                key_payload=key_payload,
            )
            request_path = root / f"{call_key}.request.json"
            request_receipt = _atomic_publish_journal(
                request_path,
                request_body,
            )
            record = self._invoke_provider(
                normalized,
                requested=requested,
                call_key=call_key,
                request_journal_sha256=request_receipt,
            )
            response_path = root / f"{call_key}.response.json"
            _atomic_publish_journal(
                response_path,
                self._response_journal_body(record),
            )
            # Read the just-published pair through the same verifier used on
            # resume before exposing output to the synthesis parser.
            verified = _load_checkpoint_journal(root)[call_key]
            return self._accept_completion(verified, cache_hit=False)

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        """Return one zero-retry gateway completion and retain its report."""

        normalized = self._validated_messages(messages)
        requested = (
            self._default_max_new_tokens
            if max_new_tokens is None
            else int(max_new_tokens)
        )
        if requested < 1:
            raise ValueError("max_new_tokens must be positive")

        with self._lock:
            self._require_open()
            key_payload = self._call_key_payload(
                normalized,
                max_new_tokens=requested,
            )
            call_key = identity_sha256(key_payload)
            if self._checkpoint_dir is None:
                return self._complete_in_memory(
                    normalized,
                    requested=requested,
                    call_key=call_key,
                    key_payload=key_payload,
                )
            return self._complete_checkpointed(
                normalized,
                requested=requested,
                call_key=call_key,
                key_payload=key_payload,
            )

    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        return self.complete(messages, max_new_tokens=max_new_tokens)

    def score_candidates(
        self,
        query: str,
        candidates: Sequence[RetrievalResult] | Mapping[str, str],
        *,
        source_timestamps: Mapping[str, str] | None = None,
    ) -> Mapping[str, CausalChoiceEvidence]:
        """Delegate scoring to the one wrapped pinned local runtime."""

        with self._lock:
            self._require_open()
            return self._local_runtime.score_candidates(
                query,
                candidates,
                source_timestamps=source_timestamps,
            )

    @property
    def last_score_report(self) -> CausalChoiceScoreReport | None:
        return self._local_runtime.last_score_report

    @property
    def usage(self) -> dict[str, int | float]:
        """Return gateway completion totals beside delegated score totals.

        ``completion_unique_calls`` is the authoritative count of distinct
        response records accounted by this session, including a verified
        journal response restored after restart. ``completion_physical_calls``
        counts only gateway attempts made by this runtime instance.  Thus a
        fresh-process cache hit has logical=1, unique=1, physical=0, hits=1.
        Reported tokens and elapsed time are charged once per unique response,
        never once per cache-hit invocation.
        """

        local_usage = self._local_runtime.usage
        return {
            # ``completion_calls`` remains the synthesis core's compatibility
            # counter and is intentionally logical: a verified replay still
            # satisfies one requested completion.
            "completion_calls": self._completion_logical_calls,
            "completion_logical_calls": self._completion_logical_calls,
            "completion_unique_calls": self._completion_unique_calls,
            "completion_physical_calls": self._completion_physical_calls,
            "completion_checkpoint_hits": self._completion_checkpoint_hits,
            "completion_reported_input_tokens": (
                self._completion_reported_input_tokens
            ),
            "completion_reported_output_tokens": (
                self._completion_reported_output_tokens
            ),
            "completion_reported_total_tokens": (
                self._completion_reported_total_tokens
            ),
            "completion_input_token_proxy": self._completion_input_token_proxy,
            "completion_output_token_proxy": self._completion_output_token_proxy,
            "completion_elapsed_s": self._completion_elapsed_s,
            "score_calls": int(_field(local_usage, "score_calls", 0)),
            "score_forward_passes": int(
                _field(local_usage, "score_forward_passes", 0)
            ),
            "score_elapsed_s": float(
                _field(local_usage, "score_elapsed_s", 0.0)
            ),
        }

    @property
    def completion_calls(self) -> int:
        return self._completion_logical_calls

    def close(self) -> None:
        """Close both the provider client and wrapped local runtime once."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            close_client = getattr(self._client, "close", None)
            try:
                if callable(close_client):
                    close_client()
            finally:
                self._local_runtime.close()

    def __enter__(
        self,
    ) -> "RecallGuardedCumulativeProviderSynthesisRuntime":
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "CALL_REQUEST_JOURNAL_FORMAT",
    "CALL_RESPONSE_JOURNAL_FORMAT",
    "CENTRAL_DEV_GATEWAY_URL",
    "DEFAULT_CALLER_MODEL",
    "PROVIDER_RUNTIME_FORMAT",
    "ProviderSynthesisCompletionReport",
    "ProviderSynthesisRuntimeIdentity",
    "RecallGuardedCumulativeProviderSynthesisRuntime",
]
