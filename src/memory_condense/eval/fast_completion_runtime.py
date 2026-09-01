"""Fast, checkpointed provider completions for matched benchmark arms.

The runtime preflights the complete logical prompt population before it
creates a checkpoint directory or calls a provider.  Identical prompts share
one physical call.  Unique cache misses run concurrently, while short
filesystem critical sections atomically reserve a request and publish its
response.  No lock is held during provider I/O.

A request journal without its response is deliberately terminal: the
provider may have accepted the call before the process failed, so retrying it
would make call counts unknowable.  Journals contain prompt hashes, response
text, and scalar usage/provenance only.  They never contain prompt token IDs,
hidden states, residuals, or K/V state.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
    tokenizer_proxy_identity,
)
from memory_condense.domain.discourse import (
    canonical_json,
    identity_sha256,
    quote_sha256,
)


FAST_COMPLETION_POPULATION_FORMAT = "memory-condense-fast-prompt-population-v1"
FAST_COMPLETION_RUNTIME_FORMAT = "memory-condense-fast-completion-runtime-v1"
FAST_COMPLETION_REQUEST_FORMAT = "memory-condense-fast-completion-request-v1"
FAST_COMPLETION_RESPONSE_FORMAT = "memory-condense-fast-completion-response-v1"
FAST_COMPLETION_TOKEN_STATE_CONTRACT = (
    "stateless-fast-provider-completion-token-state-v1"
)

_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_JOURNAL_NAME = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)
_RESERVED_REQUEST_KEYS = frozenset(
    {"model", "messages", "max_tokens", "num_retries", "retries", "stream"}
)
_TRANSFORMER_STATE_KEY_SUFFIXES = (
    "attention_mask",
    "cache_position",
    "hidden_state",
    "hidden_states",
    "input_ids",
    "key_cache",
    "kv_cache",
    "output_ids",
    "past_key_values",
    "position_ids",
    "residual_stream",
    "token_ids",
    "value_cache",
)


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _strict_json_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    try:
        detached = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must contain strict JSON values") from exc
    if not isinstance(detached, dict):  # pragma: no cover - guarded above
        raise TypeError(f"{label} must be a mapping")
    return detached


def _freeze_json(value: Any) -> Any:
    """Recursively freeze already-validated JSON without sharing containers."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child) for child in value)
    return value


def _reject_transformer_state(value: Any, *, label: str, path: str = "") -> None:
    """Reject recognized token/residual/KV payloads from persisted metadata."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            child_path = f"{path}.{key}" if path else str(key)
            if normalized.endswith(_TRANSFORMER_STATE_KEY_SUFFIXES):
                raise ValueError(
                    f"{label} must not persist transformer state at {child_path}"
                )
            _reject_transformer_state(child, label=label, path=child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_transformer_state(
                child,
                label=label,
                path=f"{path}[{index}]",
            )


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(child) for child in value]
    return value


def _normalize_messages(
    messages: Sequence[Mapping[str, str]],
    *,
    label: str,
) -> tuple[Mapping[str, str], ...]:
    if isinstance(messages, (str, bytes, bytearray)) or not messages:
        raise ValueError(f"{label} must be a non-empty message sequence")
    normalized: list[Mapping[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"{label} message {index} must be a mapping")
        if set(message) != {"role", "content"}:
            raise ValueError(
                f"{label} message {index} must contain exactly role and content"
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or role not in _ALLOWED_ROLES:
            raise ValueError(f"{label} message {index} has an unsupported role")
        if not isinstance(content, str):
            raise ValueError(f"{label} message {index} content must be a string")
        normalized.append(MappingProxyType({"role": role, "content": content}))
    return tuple(normalized)


def _plain_messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [dict(message) for message in messages]


@dataclass(frozen=True, slots=True)
class FastPromptRow:
    ordinal: int
    messages_sha256: str
    prompt_token_proxy: int

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FastPromptPopulation:
    """Provider-free result of validating and counting every prompt."""

    format: str
    logical_prompt_count: int
    unique_prompt_count: int
    ordered_rows: tuple[FastPromptRow, ...]
    prompt_population_sha256: str
    max_prompt_token_proxy: int
    prompt_token_proxy_identity: Mapping[str, Any]
    normalized_prompts: tuple[tuple[Mapping[str, str], ...], ...]

    def model_dump(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "logical_prompt_count": self.logical_prompt_count,
            "unique_prompt_count": self.unique_prompt_count,
            "ordered_rows": [row.model_dump() for row in self.ordered_rows],
            "prompt_population_sha256": self.prompt_population_sha256,
            "max_prompt_token_proxy": self.max_prompt_token_proxy,
            "prompt_token_proxy_identity": dict(self.prompt_token_proxy_identity),
        }


def preflight_fast_completion_prompts(
    prompt_population: Sequence[Sequence[Mapping[str, str]]],
    *,
    max_prompt_tokens: int,
) -> FastPromptPopulation:
    """Validate, hash, and budget the full population without side effects."""

    cap = _positive_int(max_prompt_tokens, label="max_prompt_tokens")
    if isinstance(prompt_population, (str, bytes, bytearray)) or not prompt_population:
        raise ValueError("prompt_population must be a non-empty sequence")

    normalized: list[tuple[Mapping[str, str], ...]] = []
    rows: list[FastPromptRow] = []
    unique: dict[str, tuple[Mapping[str, str], ...]] = {}
    for ordinal, candidate in enumerate(prompt_population):
        messages = _normalize_messages(candidate, label=f"prompt {ordinal}")
        plain = _plain_messages(messages)
        messages_sha = identity_sha256(plain)
        prompt_tokens = count_chat_prompt_token_proxy(messages)
        if type(prompt_tokens) is not int or prompt_tokens < 1:
            raise ValueError(
                "prompt token counter must return a positive integer at "
                f"ordinal {ordinal}"
            )
        if prompt_tokens > cap:
            raise ValueError(
                "prompt population exceeds the hard token cap at ordinal "
                f"{ordinal}: {prompt_tokens} > {cap}"
            )
        previous = unique.setdefault(messages_sha, messages)
        if previous != messages:
            raise RuntimeError("prompt SHA-256 collision")
        normalized.append(messages)
        rows.append(
            FastPromptRow(
                ordinal=ordinal,
                messages_sha256=messages_sha,
                prompt_token_proxy=prompt_tokens,
            )
        )

    token_identity = tokenizer_proxy_identity()
    identity_body = {
        "format": FAST_COMPLETION_POPULATION_FORMAT,
        "logical_prompt_count": len(rows),
        "unique_prompt_count": len(unique),
        "ordered_rows": [row.model_dump() for row in rows],
        "max_prompt_token_proxy": cap,
        "prompt_token_proxy_identity": token_identity,
    }
    return FastPromptPopulation(
        format=FAST_COMPLETION_POPULATION_FORMAT,
        logical_prompt_count=len(rows),
        unique_prompt_count=len(unique),
        ordered_rows=tuple(rows),
        prompt_population_sha256=identity_sha256(identity_body),
        max_prompt_token_proxy=cap,
        prompt_token_proxy_identity=MappingProxyType(token_identity),
        normalized_prompts=tuple(normalized),
    )


@dataclass(frozen=True, slots=True)
class FastCompletionProvenance:
    format: str
    model: str
    max_new_tokens: int
    max_prompt_token_proxy: int
    max_concurrency: int
    retries: int
    request_options: Mapping[str, Any]
    prompt_population_sha256: str
    prompt_token_proxy_identity: Mapping[str, Any]
    benchmark_provenance: Mapping[str, Any]
    persisted_transformer_token_state: bool
    retained_transformer_token_state_bytes: int
    external_provider_persistence_certified: bool

    def model_dump(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "model": self.model,
            "max_new_tokens": self.max_new_tokens,
            "max_prompt_token_proxy": self.max_prompt_token_proxy,
            "max_concurrency": self.max_concurrency,
            "retries": self.retries,
            "request_options": _plain_json(self.request_options),
            "prompt_population_sha256": self.prompt_population_sha256,
            "prompt_token_proxy_identity": dict(
                self.prompt_token_proxy_identity
            ),
            "benchmark_provenance": _plain_json(self.benchmark_provenance),
            "persisted_transformer_token_state": (
                self.persisted_transformer_token_state
            ),
            "retained_transformer_token_state_bytes": (
                self.retained_transformer_token_state_bytes
            ),
            "external_provider_persistence_certified": (
                self.external_provider_persistence_certified
            ),
        }


@dataclass(frozen=True, slots=True)
class FastCompletionRecord:
    """One verified unique response, plus this run's cache disposition."""

    call_key_sha256: str
    request_journal_sha256: str
    response_journal_sha256: str
    messages_sha256: str
    completion: str
    completion_sha256: str
    requested_model: str
    response_id: str
    response_model: str
    finish_reason: str
    prompt_token_proxy: int
    completion_token_proxy: int
    reported_prompt_tokens: int | None
    reported_completion_tokens: int | None
    reported_total_tokens: int | None
    provider_elapsed_s: float
    checkpoint_hit: bool
    physical_call: bool

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FastCompletionUsage:
    logical_calls: int
    unique_calls: int
    deduplicated_logical_calls: int
    physical_calls: int
    checkpoint_hits: int
    prompt_token_proxy: int
    completion_token_proxy: int
    recorded_reported_prompt_tokens: int
    recorded_reported_completion_tokens: int
    recorded_reported_total_tokens: int
    reported_prompt_tokens_complete: bool
    reported_completion_tokens_complete: bool
    reported_total_tokens_complete: bool
    recorded_provider_elapsed_s: float

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FastCompletionBatch:
    """Logical outputs and unique-call evidence for one completed batch."""

    logical_completions: tuple[str, ...]
    unique_records: tuple[FastCompletionRecord, ...]
    usage: FastCompletionUsage
    provenance: FastCompletionProvenance
    runtime_identity_sha256: str
    prompt_population: FastPromptPopulation

    def model_dump(self) -> dict[str, Any]:
        return {
            "logical_completions": list(self.logical_completions),
            "unique_records": [record.model_dump() for record in self.unique_records],
            "usage": self.usage.model_dump(),
            "provenance": self.provenance.model_dump(),
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "prompt_population": self.prompt_population.model_dump(),
        }


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical_json(dict(value)) + "\n").encode("utf-8")


def _sealed(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["journal_sha256"] = identity_sha256(dict(body))
    return result


def _atomic_publish(path: Path, body: Mapping[str, Any]) -> str:
    """Publish one complete journal atomically; caller holds the journal lock."""

    if path.exists():
        raise FileExistsError(f"refusing to replace completion journal: {path}")
    payload = _sealed(body)
    encoded = _canonical_bytes(payload)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"completion journal must be a regular file: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"completion journal is not valid JSON: {path}") from exc
    if not isinstance(payload, dict) or raw != _canonical_bytes(payload):
        raise ValueError(f"completion journal is not canonical JSON: {path}")
    receipt = payload.get("journal_sha256")
    if not isinstance(receipt, str) or _DIGEST.fullmatch(receipt) is None:
        raise ValueError(f"completion journal has no valid receipt: {path}")
    body = dict(payload)
    body.pop("journal_sha256")
    if identity_sha256(body) != receipt:
        raise ValueError(f"completion journal receipt does not verify: {path}")
    return payload, receipt


@contextmanager
def _checkpoint_lock(root: Path) -> Iterator[None]:
    lock_path = root / ".fast-completion-journal.lock"
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


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(
        value, name, default
    )


def _reported_count(usage: Any, name: str) -> int | None:
    if usage is None:
        return None
    value = _field(usage, name)
    # The controlled central-dev Codex routes use zero as an explicit
    # "usage unavailable" sentinel.  A non-empty completion cannot genuinely
    # consume zero prompt/total tokens, so normalize the sentinel before it is
    # persisted instead of misreporting complete provider accounting.
    if type(value) is not int or value <= 0:
        return None
    return value


class FastCompletionRuntime:
    """Complete or replay one fully preflighted prompt population."""

    def __init__(
        self,
        *,
        checkpoint_dir: str | Path,
        prompt_population: Sequence[Sequence[Mapping[str, str]]],
        model: str,
        client: Any | None,
        max_prompt_tokens: int,
        max_new_tokens: int,
        max_concurrency: int = 4,
        retries: int = 0,
        request_options: Mapping[str, Any] | None = None,
        benchmark_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        # This intentionally precedes every path mutation and provider access.
        population = preflight_fast_completion_prompts(
            prompt_population,
            max_prompt_tokens=max_prompt_tokens,
        )
        requested_model = str(model)
        if not requested_model or requested_model.strip() != requested_model:
            raise ValueError("model must be a non-empty exact string")
        output_cap = _positive_int(max_new_tokens, label="max_new_tokens")
        concurrency = _positive_int(max_concurrency, label="max_concurrency")
        if type(retries) is not int or retries != 0:
            raise ValueError("retries must be zero; uncertain calls are not retried")
        if client is not None:
            client_retries = getattr(client, "max_retries", None)
            if type(client_retries) is not int or client_retries != 0:
                raise ValueError(
                    "the injected provider client must expose max_retries=0"
                )
        options = _strict_json_mapping(
            request_options or {}, label="request_options"
        )
        conflict = set(options) & _RESERVED_REQUEST_KEYS
        if conflict:
            raise ValueError(
                "request_options contains a runtime-owned key: " + min(conflict)
            )
        provenance = _strict_json_mapping(
            benchmark_provenance or {}, label="benchmark_provenance"
        )
        _reject_transformer_state(options, label="request_options")
        _reject_transformer_state(provenance, label="benchmark_provenance")

        identity = FastCompletionProvenance(
            format=FAST_COMPLETION_RUNTIME_FORMAT,
            model=requested_model,
            max_new_tokens=output_cap,
            max_prompt_token_proxy=population.max_prompt_token_proxy,
            max_concurrency=concurrency,
            retries=0,
            request_options=_freeze_json(options),
            prompt_population_sha256=population.prompt_population_sha256,
            prompt_token_proxy_identity=population.prompt_token_proxy_identity,
            benchmark_provenance=_freeze_json(provenance),
            persisted_transformer_token_state=False,
            retained_transformer_token_state_bytes=0,
            external_provider_persistence_certified=False,
        )
        runtime_sha = identity_sha256(identity.model_dump())

        prompts_by_hash: dict[str, tuple[tuple[Mapping[str, str], ...], int]] = {}
        unique_order: list[str] = []
        for messages, row in zip(
            population.normalized_prompts, population.ordered_rows, strict=True
        ):
            if row.messages_sha256 not in prompts_by_hash:
                prompts_by_hash[row.messages_sha256] = (
                    messages,
                    row.prompt_token_proxy,
                )
                unique_order.append(row.messages_sha256)

        self.population = population
        self.provenance = identity
        self.runtime_identity_sha256 = runtime_sha
        self._model = requested_model
        self._client = client
        self._max_new_tokens = output_cap
        self._max_concurrency = concurrency
        self._request_options = options
        self._prompts_by_hash = prompts_by_hash
        self._unique_order = tuple(unique_order)
        self._call_keys = {
            messages_sha: identity_sha256(
                {
                    "format": FAST_COMPLETION_REQUEST_FORMAT,
                    "runtime_identity_sha256": runtime_sha,
                    "prompt_population_sha256": population.prompt_population_sha256,
                    "messages_sha256": messages_sha,
                    "prompt_token_proxy": prompt_tokens,
                    "max_new_tokens": output_cap,
                }
            )
            for messages_sha, (_messages, prompt_tokens) in prompts_by_hash.items()
        }
        self._thread_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._running = False
        self._closed = False

        root = Path(checkpoint_dir)
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("checkpoint_dir must be a regular directory")
        self._checkpoint_dir = root
        with self._journal_guard():
            self._load_all_records()

    @contextmanager
    def _journal_guard(self) -> Iterator[None]:
        with self._thread_lock:
            with _checkpoint_lock(self._checkpoint_dir):
                yield

    def _request_body(self, messages_sha: str) -> dict[str, Any]:
        if identity_sha256(self.provenance.model_dump()) != (
            self.runtime_identity_sha256
        ):
            raise RuntimeError("fast completion runtime identity changed")
        _messages, prompt_tokens = self._prompts_by_hash[messages_sha]
        call_key = self._call_keys[messages_sha]
        return {
            "format": FAST_COMPLETION_REQUEST_FORMAT,
            "call_key_sha256": call_key,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "runtime_identity": self.provenance.model_dump(),
            "prompt_population_sha256": self.population.prompt_population_sha256,
            "messages_sha256": messages_sha,
            "prompt_token_proxy": prompt_tokens,
            "max_new_tokens": self._max_new_tokens,
        }

    def _paths(self, call_key: str) -> tuple[Path, Path]:
        return (
            self._checkpoint_dir / f"{call_key}.request.json",
            self._checkpoint_dir / f"{call_key}.response.json",
        )

    def _load_record(self, messages_sha: str) -> FastCompletionRecord | None:
        call_key = self._call_keys[messages_sha]
        request_path, response_path = self._paths(call_key)
        if response_path.exists() and not request_path.exists():
            raise ValueError(
                "completion response has no request reservation: " + call_key
            )
        if request_path.exists() and not response_path.exists():
            raise RuntimeError(
                "completion request has no response; refusing an unsafe retry: "
                + call_key
            )
        if not request_path.exists():
            return None

        request, request_receipt = _read_journal(request_path)
        expected_request = self._request_body(messages_sha)
        request_body = dict(request)
        request_body.pop("journal_sha256")
        if request_body != expected_request:
            raise ValueError(f"completion request provenance changed: {request_path}")

        response, response_receipt = _read_journal(response_path)
        expected_fields = {
            "format",
            "call_key_sha256",
            "request_journal_sha256",
            "messages_sha256",
            "completion",
            "completion_sha256",
            "requested_model",
            "response_id",
            "response_model",
            "finish_reason",
            "prompt_token_proxy",
            "completion_token_proxy",
            "reported_prompt_tokens",
            "reported_completion_tokens",
            "reported_total_tokens",
            "provider_elapsed_s",
            "journal_sha256",
        }
        if set(response) != expected_fields:
            raise ValueError(f"completion response fields changed: {response_path}")
        if (
            response.get("format") != FAST_COMPLETION_RESPONSE_FORMAT
            or response.get("call_key_sha256") != call_key
            or response.get("request_journal_sha256") != request_receipt
            or response.get("messages_sha256") != messages_sha
            or response.get("requested_model") != self._model
        ):
            raise ValueError(f"completion response provenance changed: {response_path}")
        completion = response.get("completion")
        if (
            not isinstance(completion, str)
            or not completion
            or response.get("completion_sha256") != quote_sha256(completion)
            or response.get("completion_token_proxy") != count_tokens(completion)
        ):
            raise ValueError(f"completion response text does not verify: {response_path}")
        completion_token_proxy = int(response["completion_token_proxy"])
        if completion_token_proxy > self._max_new_tokens:
            raise ValueError(
                "completion token proxy violates the configured max_new_tokens: "
                f"{completion_token_proxy} > {self._max_new_tokens}: {response_path}"
            )
        _messages, prompt_tokens = self._prompts_by_hash[messages_sha]
        if response.get("prompt_token_proxy") != prompt_tokens:
            raise ValueError(f"completion prompt usage changed: {response_path}")
        reported_prompt_tokens = response.get("reported_prompt_tokens")
        if reported_prompt_tokens is not None and (
            type(reported_prompt_tokens) is not int
            or reported_prompt_tokens < 1
            or reported_prompt_tokens > self.population.max_prompt_token_proxy
        ):
            raise ValueError(
                "completion reported prompt usage violates the hard token cap: "
                f"{response_path}"
            )
        for name in ("reported_completion_tokens", "reported_total_tokens"):
            value = response.get(name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"completion reported usage is invalid: {response_path}")
        elapsed = response.get("provider_elapsed_s")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise ValueError(f"completion elapsed time is invalid: {response_path}")
        for name in ("response_id", "response_model", "finish_reason"):
            if not isinstance(response.get(name), str):
                raise ValueError(f"completion provider metadata is invalid: {response_path}")

        return FastCompletionRecord(
            call_key_sha256=call_key,
            request_journal_sha256=request_receipt,
            response_journal_sha256=response_receipt,
            messages_sha256=messages_sha,
            completion=completion,
            completion_sha256=str(response["completion_sha256"]),
            requested_model=self._model,
            response_id=str(response["response_id"]),
            response_model=str(response["response_model"]),
            finish_reason=str(response["finish_reason"]),
            prompt_token_proxy=prompt_tokens,
            completion_token_proxy=completion_token_proxy,
            reported_prompt_tokens=response["reported_prompt_tokens"],
            reported_completion_tokens=response["reported_completion_tokens"],
            reported_total_tokens=response["reported_total_tokens"],
            provider_elapsed_s=float(elapsed),
            checkpoint_hit=True,
            physical_call=False,
        )

    def _load_all_records(self) -> dict[str, FastCompletionRecord]:
        requests: set[str] = set()
        responses: set[str] = set()
        for path in self._checkpoint_dir.glob("*.json"):
            match = _JOURNAL_NAME.fullmatch(path.name)
            if match is None:
                raise ValueError(f"unexpected JSON completion journal: {path}")
            (requests if match.group("kind") == "request" else responses).add(
                match.group("key")
            )
        allowed = set(self._call_keys.values())
        unexpected = (requests | responses) - allowed
        if unexpected:
            raise ValueError(
                "completion journal is outside the preflighted population: "
                + min(unexpected)
            )
        orphan = responses - requests
        if orphan:
            raise ValueError(
                "completion response has no request reservation: " + min(orphan)
            )
        incomplete = requests - responses
        if incomplete:
            raise RuntimeError(
                "completion request has no response; refusing an unsafe retry: "
                + min(incomplete)
            )
        records: dict[str, FastCompletionRecord] = {}
        for messages_sha in self._unique_order:
            record = self._load_record(messages_sha)
            if record is not None:
                records[messages_sha] = record
        return records

    def _reserve(self, messages_sha: str) -> FastCompletionRecord | str:
        """Return a cache record or the new immutable request receipt."""

        with self._journal_guard():
            cached = self._load_record(messages_sha)
            if cached is not None:
                return cached
            call_key = self._call_keys[messages_sha]
            request_path, _response_path = self._paths(call_key)
            return _atomic_publish(request_path, self._request_body(messages_sha))

    def _provider_call(self, messages_sha: str) -> FastCompletionRecord:
        messages, prompt_tokens = self._prompts_by_hash[messages_sha]
        observed_prompt_tokens = count_chat_prompt_token_proxy(messages)
        # Detect in-memory corruption after preflight, before a reservation.
        if (
            identity_sha256(_plain_messages(messages)) != messages_sha
            or type(observed_prompt_tokens) is not int
            or observed_prompt_tokens < 1
            or observed_prompt_tokens != prompt_tokens
            or prompt_tokens > self.population.max_prompt_token_proxy
        ):
            raise RuntimeError("preflighted prompt changed before completion")

        reservation = self._reserve(messages_sha)
        if isinstance(reservation, FastCompletionRecord):
            return reservation
        if self._client is None:
            # Reservation is intentionally not made when replay has no result.
            # `_reserve` only reaches here after creating one, so guard this in
            # `run` before workers start.
            raise RuntimeError("provider client is required for uncached prompts")

        request = dict(self._request_options)
        request.update(
            {
                "model": self._model,
                "messages": _plain_messages(messages),
                "max_tokens": self._max_new_tokens,
            }
        )
        started = time.perf_counter()
        # Deliberately outside `_journal_guard`: calls may overlap.
        response = self._client.chat.completions.create(**request)
        elapsed = time.perf_counter() - started

        choices = _field(response, "choices", ())
        if not choices:
            raise RuntimeError("provider returned no completion choices")
        choice = choices[0]
        message = _field(choice, "message")
        completion = str(_field(message, "content", "") or "").strip()
        if not completion:
            raise RuntimeError("provider returned no completion text")
        completion_token_proxy = count_tokens(completion)
        if completion_token_proxy > self._max_new_tokens:
            raise RuntimeError(
                "provider completion token proxy violates the configured "
                "max_new_tokens: "
                f"{completion_token_proxy} > {self._max_new_tokens}"
            )
        finish_reason = _field(choice, "finish_reason")
        if finish_reason != "stop":
            raise RuntimeError(
                "provider completion was not terminally complete: "
                f"finish_reason={finish_reason!r}"
            )
        usage = _field(response, "usage")
        reported_prompt_tokens = (
            None if usage is None else _field(usage, "prompt_tokens")
        )
        if type(reported_prompt_tokens) is int and reported_prompt_tokens == 0:
            reported_prompt_tokens = None
        if reported_prompt_tokens is not None and (
            type(reported_prompt_tokens) is not int
            or reported_prompt_tokens < 1
            or reported_prompt_tokens > self.population.max_prompt_token_proxy
        ):
            raise RuntimeError(
                "provider-reported prompt tokens violate the hard token cap: "
                f"{reported_prompt_tokens} not in "
                f"[1, {self.population.max_prompt_token_proxy}]"
            )
        call_key = self._call_keys[messages_sha]
        response_body = {
            "format": FAST_COMPLETION_RESPONSE_FORMAT,
            "call_key_sha256": call_key,
            "request_journal_sha256": reservation,
            "messages_sha256": messages_sha,
            "completion": completion,
            "completion_sha256": quote_sha256(completion),
            "requested_model": self._model,
            "response_id": str(_field(response, "id", "") or ""),
            "response_model": str(_field(response, "model", "") or ""),
            "finish_reason": finish_reason,
            "prompt_token_proxy": prompt_tokens,
            "completion_token_proxy": completion_token_proxy,
            "reported_prompt_tokens": reported_prompt_tokens,
            "reported_completion_tokens": _reported_count(
                usage, "completion_tokens"
            ),
            "reported_total_tokens": _reported_count(usage, "total_tokens"),
            "provider_elapsed_s": elapsed,
        }
        with self._journal_guard():
            # Verify the reservation is still byte-identical before publishing.
            request_path, response_path = self._paths(call_key)
            _request, observed_reservation = _read_journal(request_path)
            if observed_reservation != reservation:
                raise ValueError("completion request changed during provider call")
            _atomic_publish(response_path, response_body)
            verified = self._load_record(messages_sha)
        assert verified is not None
        return replace(verified, checkpoint_hit=False, physical_call=True)

    def _run_batch(self) -> FastCompletionBatch:
        with self._journal_guard():
            cached = self._load_all_records()
        pending = [key for key in self._unique_order if key not in cached]
        if pending and self._client is None:
            raise RuntimeError("provider client is required for uncached prompts")

        completed: dict[str, FastCompletionRecord] = dict(cached)
        if pending:
            workers = min(self._max_concurrency, len(pending))
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="fast-completion"
            ) as executor:
                pending_iterator = iter(pending)
                futures: dict[Any, str] = {}

                def submit_one() -> bool:
                    try:
                        messages_sha = next(pending_iterator)
                    except StopIteration:
                        return False
                    future = executor.submit(self._provider_call, messages_sha)
                    futures[future] = messages_sha
                    return True

                while len(futures) < workers and submit_one():
                    pass
                while futures:
                    done, _not_done = wait(
                        tuple(futures),
                        return_when=FIRST_COMPLETED,
                    )
                    failure: Exception | None = None
                    for future in done:
                        messages_sha = futures.pop(future)
                        try:
                            completed[messages_sha] = future.result()
                        except Exception as exc:  # preserve the first provider error
                            if failure is None:
                                failure = exc
                    if failure is not None:
                        # At most `workers` calls were in flight.  No further
                        # authorized work is submitted after the first known
                        # failure; already-running calls retain their journals.
                        for future in futures:
                            future.cancel()
                        raise failure
                    while len(futures) < workers and submit_one():
                        pass

        records = tuple(completed[key] for key in self._unique_order)
        logical = tuple(
            completed[row.messages_sha256].completion
            for row in self.population.ordered_rows
        )

        def known_sum(name: str) -> int:
            return sum(
                int(value)
                for record in records
                if (value := getattr(record, name)) is not None
            )

        usage = FastCompletionUsage(
            logical_calls=self.population.logical_prompt_count,
            unique_calls=self.population.unique_prompt_count,
            deduplicated_logical_calls=(
                self.population.logical_prompt_count
                - self.population.unique_prompt_count
            ),
            physical_calls=sum(record.physical_call for record in records),
            checkpoint_hits=sum(record.checkpoint_hit for record in records),
            prompt_token_proxy=sum(record.prompt_token_proxy for record in records),
            completion_token_proxy=sum(
                record.completion_token_proxy for record in records
            ),
            recorded_reported_prompt_tokens=known_sum("reported_prompt_tokens"),
            recorded_reported_completion_tokens=known_sum(
                "reported_completion_tokens"
            ),
            recorded_reported_total_tokens=known_sum("reported_total_tokens"),
            reported_prompt_tokens_complete=all(
                record.reported_prompt_tokens is not None for record in records
            ),
            reported_completion_tokens_complete=all(
                record.reported_completion_tokens is not None
                for record in records
            ),
            reported_total_tokens_complete=all(
                record.reported_total_tokens is not None for record in records
            ),
            recorded_provider_elapsed_s=sum(
                record.provider_elapsed_s for record in records
            ),
        )
        return FastCompletionBatch(
            logical_completions=logical,
            unique_records=records,
            usage=usage,
            provenance=self.provenance,
            runtime_identity_sha256=self.runtime_identity_sha256,
            prompt_population=self.population,
        )

    def run(self) -> FastCompletionBatch:
        """Complete all unique misses concurrently and expand logical outputs."""

        with self._state_lock:
            if self._closed:
                raise RuntimeError("fast completion runtime is closed")
            if self._running:
                raise RuntimeError("fast completion runtime is already running")
            self._running = True
        try:
            return self._run_batch()
        finally:
            with self._state_lock:
                self._running = False

    def request_token_state_receipt(self) -> dict[str, Any]:
        return {
            "contract": FAST_COMPLETION_TOKEN_STATE_CONTRACT,
            "persisted_transformer_token_state": False,
            "retained_transformer_token_state_bytes": 0,
            "journal_payload_kinds": (
                "prompt_hashes_response_text_scalar_usage_and_provenance"
            ),
            "external_provider_persistence_certified": False,
        }

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            if self._running:
                raise RuntimeError("cannot close fast completion runtime while running")
            self._closed = True
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "FastCompletionRuntime":
        with self._state_lock:
            if self._closed:
                raise RuntimeError("fast completion runtime is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "FAST_COMPLETION_POPULATION_FORMAT",
    "FAST_COMPLETION_REQUEST_FORMAT",
    "FAST_COMPLETION_RESPONSE_FORMAT",
    "FAST_COMPLETION_RUNTIME_FORMAT",
    "FAST_COMPLETION_TOKEN_STATE_CONTRACT",
    "FastCompletionBatch",
    "FastCompletionProvenance",
    "FastCompletionRecord",
    "FastCompletionRuntime",
    "FastCompletionUsage",
    "FastPromptPopulation",
    "FastPromptRow",
    "preflight_fast_completion_prompts",
]
