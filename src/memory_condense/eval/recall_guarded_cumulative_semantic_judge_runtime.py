"""Durable, zero-retry provider runtime for independent binary QA judging.

The runtime is deliberately separate from the cumulative synthesis runtime:
it owns no transformer scorer and cannot alter a measured synthesis artifact.
Every physical call is reserved by an immutable request journal before the
gateway is contacted, then paired with an immutable response journal.  A
request without a response is a terminal uncertainty and is never retried.
"""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.domain._tokenizer import (
    count_chat_prompt_token_proxy,
    count_tokens,
)
from memory_condense.domain.discourse import identity_sha256, quote_sha256
from memory_condense.eval._binary_judge_protocol import (
    parse_binary_judge_verdict,
)
from memory_condense.eval.recall_guarded_cumulative_provider_synthesis_runtime import (
    CENTRAL_DEV_GATEWAY_URL,
    _atomic_publish_journal,
    _campaign_binding,
    _checkpoint_lock,
    _field,
    _gateway_model,
    _new_gateway_client,
    _read_journal,
    _reported_count,
)


SEMANTIC_JUDGE_RUNTIME_FORMAT = (
    "memory-condense-recall-guarded-semantic-judge-runtime-v1"
)
JUDGE_REQUEST_JOURNAL_FORMAT = (
    "memory-condense-semantic-judge-call-request-v1"
)
JUDGE_RESPONSE_JOURNAL_FORMAT = (
    "memory-condense-semantic-judge-call-response-v1"
)
DEFAULT_JUDGE_MODEL = "openai/codex_sdk/gpt-5.6-sol"
DEFAULT_JUDGE_MAX_NEW_TOKENS = 1024

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_JOURNAL_NAME = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)


def _require_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


@dataclass(frozen=True, slots=True)
class SemanticJudgeRuntimeIdentity:
    """The exact provider route, request policy, and campaign binding."""

    format: str
    gateway_url: str
    caller_model: str
    gateway_model: str
    default_max_new_tokens: int
    retries: int
    temperature: float | None
    authorized_unique_calls: int
    campaign_binding: Mapping[str, Any]
    campaign_binding_sha256: str
    request_journal_format: str
    response_journal_format: str

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SemanticJudgeCompletionReport:
    """Text-free immutable metadata for one journaled provider response."""

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
    cumulative_logical_calls: int
    cumulative_unique_calls: int
    cumulative_physical_calls: int
    cumulative_checkpoint_hits: int

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _JournalRecord:
    call_key_sha256: str
    request_journal_sha256: str
    response_journal_sha256: str
    completion: str
    report: SemanticJudgeCompletionReport


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
        label="judge request journal",
    )
    if payload["format"] != JUDGE_REQUEST_JOURNAL_FORMAT:
        raise ValueError(f"unexpected judge request journal format: {path}")
    call_key = _require_digest(payload["call_key_sha256"], label="call key")
    match = _JOURNAL_NAME.fullmatch(path.name)
    if match is None or match.group("kind") != "request":
        raise ValueError(f"invalid judge request journal name: {path}")
    if match.group("key") != call_key:
        raise ValueError(f"judge request journal filename changed: {path}")
    key_payload = payload["call_key_payload"]
    runtime_identity = payload["runtime_identity"]
    campaign_binding = payload["campaign_binding"]
    if not isinstance(key_payload, Mapping):
        raise ValueError(f"judge call key payload must be an object: {path}")
    if not isinstance(runtime_identity, Mapping):
        raise ValueError(f"judge runtime identity must be an object: {path}")
    if not isinstance(campaign_binding, Mapping):
        raise ValueError(f"judge campaign binding must be an object: {path}")
    _require_exact_keys(
        key_payload,
        {
            "messages_sha256",
            "runtime_identity_sha256",
            "max_new_tokens",
            "campaign_binding_sha256",
        },
        label="judge call key payload",
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
    maximum = key_payload["max_new_tokens"]
    if type(maximum) is not int or maximum < 1:
        raise ValueError(f"judge call max_new_tokens is invalid: {path}")
    if identity_sha256(dict(runtime_identity)) != runtime_sha:
        raise ValueError(f"judge runtime identity does not verify: {path}")
    if identity_sha256(dict(campaign_binding)) != campaign_sha:
        raise ValueError(f"judge campaign binding does not verify: {path}")
    if identity_sha256(dict(key_payload)) != call_key:
        raise ValueError(f"judge call key does not verify: {path}")
    if payload["journal_sha256"] != receipt:
        raise ValueError(f"judge request journal receipt changed: {path}")
    return dict(payload)


def _validated_report(
    value: Any,
    *,
    request: Mapping[str, Any],
    request_receipt: str,
    completion: str,
    path: Path,
) -> SemanticJudgeCompletionReport:
    if not isinstance(value, Mapping):
        raise ValueError(f"stored judge report must be an object: {path}")
    _require_exact_keys(
        value,
        {field.name for field in fields(SemanticJudgeCompletionReport)},
        label="stored judge report",
    )
    try:
        report = SemanticJudgeCompletionReport(**dict(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stored judge report is invalid: {path}") from exc
    booleans = (
        report.reported_usage_available,
        report.reported_input_tokens_available,
        report.reported_output_tokens_available,
        report.reported_total_tokens_available,
        report.cache_hit,
        report.physical_call,
    )
    if any(type(item) is not bool for item in booleans):
        raise ValueError(f"stored judge report booleans changed: {path}")
    counters = (
        report.reported_input_tokens,
        report.reported_output_tokens,
        report.reported_total_tokens,
        report.input_token_proxy,
        report.output_token_proxy,
        report.cumulative_logical_calls,
        report.cumulative_unique_calls,
        report.cumulative_physical_calls,
        report.cumulative_checkpoint_hits,
    )
    if any(type(item) is not int or item < 0 for item in counters):
        raise ValueError(f"stored judge report counters changed: {path}")
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
    if tuple(item > 0 for item in reported) != availability:
        raise ValueError(f"stored judge usage availability changed: {path}")
    if report.reported_usage_available != any(availability):
        raise ValueError(f"stored judge usage summary changed: {path}")
    key_payload = request["call_key_payload"]
    runtime_identity = request["runtime_identity"]
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
        raise ValueError(f"stored judge report binding changed: {path}")
    if (
        isinstance(report.elapsed_s, bool)
        or not isinstance(report.elapsed_s, (int, float))
        or not math.isfinite(float(report.elapsed_s))
        or report.elapsed_s < 0
    ):
        raise ValueError(f"stored judge elapsed time changed: {path}")
    if (
        report.cumulative_logical_calls < 1
        or report.cumulative_unique_calls < 1
        or report.cumulative_physical_calls < 1
    ):
        raise ValueError(f"stored judge cumulative counts changed: {path}")
    return report


def _validate_response_journal(
    path: Path,
    payload: Mapping[str, Any],
    *,
    receipt: str,
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
        label="judge response journal",
    )
    if payload["format"] != JUDGE_RESPONSE_JOURNAL_FORMAT:
        raise ValueError(f"unexpected judge response journal format: {path}")
    call_key = _require_digest(payload["call_key_sha256"], label="call key")
    match = _JOURNAL_NAME.fullmatch(path.name)
    if match is None or match.group("kind") != "response":
        raise ValueError(f"invalid judge response journal name: {path}")
    if match.group("key") != call_key or call_key != request["call_key_sha256"]:
        raise ValueError(f"judge response journal filename changed: {path}")
    if payload["request_journal_sha256"] != request_receipt:
        raise ValueError(f"judge response request binding changed: {path}")
    completion = payload["completion"]
    if (
        not isinstance(completion, str)
        or not completion
        or completion.strip() != completion
    ):
        raise ValueError(f"stored judge completion text is invalid: {path}")
    report = _validated_report(
        payload["report"],
        request=request,
        request_receipt=request_receipt,
        completion=completion,
        path=path,
    )
    return _JournalRecord(
        call_key_sha256=call_key,
        request_journal_sha256=request_receipt,
        response_journal_sha256=receipt,
        completion=completion,
        report=report,
    )


def _load_checkpoint_journal(root: Path) -> dict[str, _JournalRecord]:
    """Verify the complete journal inventory, failing on uncertain calls."""

    requests: dict[str, Path] = {}
    responses: dict[str, Path] = {}
    for path in root.glob("*.json"):
        match = _JOURNAL_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected JSON file in judge journal: {path}")
        target = requests if match.group("kind") == "request" else responses
        target[match.group("key")] = path
    orphan = set(responses) - set(requests)
    incomplete = set(requests) - set(responses)
    if orphan:
        raise ValueError(
            "judge response has no request reservation: " + min(orphan)
        )
    if incomplete:
        raise RuntimeError(
            "judge request has no response; refusing an unsafe retry: "
            + min(incomplete)
        )
    records: dict[str, _JournalRecord] = {}
    for key in sorted(requests):
        request_payload, request_receipt = _read_journal(requests[key])
        request = _validate_request_journal(
            requests[key], request_payload, request_receipt
        )
        response_payload, response_receipt = _read_journal(responses[key])
        records[key] = _validate_response_journal(
            responses[key],
            response_payload,
            receipt=response_receipt,
            request=request,
            request_receipt=request_receipt,
        )
    return records


class RecallGuardedCumulativeSemanticJudgeRuntime:
    """Call or replay an independent binary judge under a unique-call cap."""

    def __init__(
        self,
        *,
        checkpoint_dir: str | Path,
        campaign_binding: Mapping[str, Any],
        authorized_unique_calls: int,
        api_key: str | None = None,
        caller_model: str = DEFAULT_JUDGE_MODEL,
        max_new_tokens: int = DEFAULT_JUDGE_MAX_NEW_TOKENS,
        replay_only: bool = False,
        client: Any | None = None,
    ) -> None:
        if type(authorized_unique_calls) is not int or authorized_unique_calls < 1:
            raise ValueError("authorized_unique_calls must be a positive integer")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        selected_model = str(caller_model).strip()
        if not selected_model:
            raise ValueError("caller_model must be non-empty")
        secret = "" if api_key is None else str(api_key).strip()
        if not replay_only and not secret:
            raise ValueError("a provider API key is required outside replay-only mode")
        if replay_only and client is not None:
            raise ValueError("replay-only mode must not receive a provider client")
        safe_binding = _campaign_binding(
            campaign_binding,
            api_key=secret or "\0memory-condense-replay-only\0",
        )
        binding_sha = identity_sha256(safe_binding)
        gateway_model = _gateway_model(selected_model)
        temperature = None if "codex_sdk/" in gateway_model else 0.0
        self.identity = SemanticJudgeRuntimeIdentity(
            format=SEMANTIC_JUDGE_RUNTIME_FORMAT,
            gateway_url=CENTRAL_DEV_GATEWAY_URL,
            caller_model=selected_model,
            gateway_model=gateway_model,
            default_max_new_tokens=max_new_tokens,
            retries=0,
            temperature=temperature,
            authorized_unique_calls=authorized_unique_calls,
            campaign_binding=safe_binding,
            campaign_binding_sha256=binding_sha,
            request_journal_format=JUDGE_REQUEST_JOURNAL_FORMAT,
            response_journal_format=JUDGE_RESPONSE_JOURNAL_FORMAT,
        )
        self._runtime_identity_sha256 = identity_sha256(
            self.identity.model_dump()
        )
        root = Path(checkpoint_dir)
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("checkpoint_dir must be a regular directory")
        with _checkpoint_lock(root):
            existing = _load_checkpoint_journal(root)
        if len(existing) > authorized_unique_calls:
            raise RuntimeError(
                "verified judge journal exceeds the authorized unique-call budget"
            )
        self._client = (
            None
            if replay_only
            else client if client is not None else _new_gateway_client(secret)
        )
        self._checkpoint_dir = root
        self._authorized_unique_calls = authorized_unique_calls
        self._replay_only = bool(replay_only)
        self._lock = threading.Lock()
        self._closed = False
        self._accounted_keys: set[str] = set()
        self._logical_calls = 0
        self._unique_calls = 0
        self._physical_calls = 0
        self._checkpoint_hits = 0
        self._reported_input_tokens = 0
        self._reported_output_tokens = 0
        self._reported_total_tokens = 0
        self._input_token_proxy = 0
        self._output_token_proxy = 0
        self._elapsed_s = 0.0
        self.last_completion_report: SemanticJudgeCompletionReport | None = None
        self.last_journal_record: Mapping[str, Any] | None = None

    @staticmethod
    def _validated_messages(
        messages: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        if not messages:
            raise ValueError("judge messages must not be empty")
        normalized: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            role = str(message.get("role", "")).strip()
            content = message.get("content")
            if not role or not isinstance(content, str):
                raise ValueError(
                    f"judge message {index} requires role and string content"
                )
            normalized.append({"role": role, "content": content})
        return normalized

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("semantic judge runtime is closed")

    def _key_payload(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        return {
            "messages_sha256": identity_sha256(list(messages)),
            "runtime_identity_sha256": self._runtime_identity_sha256,
            "max_new_tokens": max_new_tokens,
            "campaign_binding_sha256": self.identity.campaign_binding_sha256,
        }

    def _request_body(
        self,
        *,
        call_key: str,
        key_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "format": JUDGE_REQUEST_JOURNAL_FORMAT,
            "call_key_sha256": call_key,
            "call_key_payload": dict(key_payload),
            "runtime_identity": self.identity.model_dump(),
            "campaign_binding": dict(self.identity.campaign_binding),
        }

    def _invoke(
        self,
        messages: list[dict[str, str]],
        *,
        requested: int,
        call_key: str,
        request_receipt: str,
    ) -> _JournalRecord:
        request: dict[str, Any] = {
            "model": self.identity.gateway_model,
            "messages": messages,
            "max_tokens": requested,
        }
        if self.identity.temperature is not None:
            request["temperature"] = self.identity.temperature
        self._physical_calls += 1
        started = time.perf_counter()
        assert self._client is not None
        response = self._client.chat.completions.create(**request)
        elapsed = time.perf_counter() - started
        choices = _field(response, "choices", ())
        if not choices:
            raise RuntimeError("judge provider returned no completion choices")
        choice = choices[0]
        message = _field(choice, "message")
        completion = str(_field(message, "content", "") or "").strip()
        if not completion:
            raise RuntimeError("judge provider returned no verdict text")
        usage = _field(response, "usage")
        input_tokens, input_available = _reported_count(usage, "prompt_tokens")
        output_tokens, output_available = _reported_count(
            usage, "completion_tokens"
        )
        total_tokens, total_available = _reported_count(usage, "total_tokens")
        report = SemanticJudgeCompletionReport(
            gateway_url=self.identity.gateway_url,
            caller_model=self.identity.caller_model,
            gateway_model=self.identity.gateway_model,
            call_key_sha256=call_key,
            runtime_identity_sha256=self._runtime_identity_sha256,
            campaign_binding_sha256=self.identity.campaign_binding_sha256,
            request_journal_sha256=request_receipt,
            messages_sha256=identity_sha256(messages),
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
            input_token_proxy=count_chat_prompt_token_proxy(messages),
            output_token_proxy=count_tokens(completion),
            elapsed_s=elapsed,
            retries=0,
            cache_hit=False,
            physical_call=True,
            cumulative_logical_calls=self._logical_calls + 1,
            cumulative_unique_calls=self._unique_calls + 1,
            cumulative_physical_calls=self._physical_calls,
            cumulative_checkpoint_hits=self._checkpoint_hits,
        )
        return _JournalRecord(
            call_key_sha256=call_key,
            request_journal_sha256=request_receipt,
            response_journal_sha256="",
            completion=completion,
            report=report,
        )

    @staticmethod
    def _response_body(record: _JournalRecord) -> dict[str, Any]:
        return {
            "format": JUDGE_RESPONSE_JOURNAL_FORMAT,
            "call_key_sha256": record.call_key_sha256,
            "request_journal_sha256": record.request_journal_sha256,
            "completion": record.completion,
            "report": record.report.model_dump(),
        }

    def _accept(self, record: _JournalRecord, *, cache_hit: bool) -> str:
        self._logical_calls += 1
        if cache_hit:
            self._checkpoint_hits += 1
        if record.call_key_sha256 not in self._accounted_keys:
            self._accounted_keys.add(record.call_key_sha256)
            self._unique_calls += 1
            report = record.report
            self._reported_input_tokens += report.reported_input_tokens
            self._reported_output_tokens += report.reported_output_tokens
            self._reported_total_tokens += report.reported_total_tokens
            self._input_token_proxy += report.input_token_proxy
            self._output_token_proxy += report.output_token_proxy
            self._elapsed_s += report.elapsed_s
        self.last_completion_report = replace(
            record.report,
            cache_hit=cache_hit,
            physical_call=not cache_hit,
            cumulative_logical_calls=self._logical_calls,
            cumulative_unique_calls=self._unique_calls,
            cumulative_physical_calls=self._physical_calls,
            cumulative_checkpoint_hits=self._checkpoint_hits,
        )
        self.last_journal_record = {
            "call_key_sha256": record.call_key_sha256,
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
            "completion_sha256": record.report.completion_sha256,
            # Preserve the immutable physical-call report so replayed result
            # artifacts are byte-identical to the first successful run.
            "completion_report": record.report.model_dump(),
        }
        return record.completion

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        normalized = self._validated_messages(messages)
        requested = (
            self.identity.default_max_new_tokens
            if max_new_tokens is None
            else max_new_tokens
        )
        if type(requested) is not int or requested < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        with self._lock:
            self._require_open()
            key_payload = self._key_payload(
                normalized, max_new_tokens=requested
            )
            call_key = identity_sha256(key_payload)
            with _checkpoint_lock(self._checkpoint_dir):
                records = _load_checkpoint_journal(self._checkpoint_dir)
                cached = records.get(call_key)
                if cached is not None:
                    return self._accept(cached, cache_hit=True)
                if self._replay_only:
                    raise RuntimeError(
                        "replay-only judge has no verified response for call: "
                        + call_key
                    )
                if len(records) >= self._authorized_unique_calls:
                    raise RuntimeError(
                        "authorized unique judge-call budget exhausted"
                    )
                request_receipt = _atomic_publish_journal(
                    self._checkpoint_dir / f"{call_key}.request.json",
                    self._request_body(
                        call_key=call_key,
                        key_payload=key_payload,
                    ),
                )
                record = self._invoke(
                    normalized,
                    requested=requested,
                    call_key=call_key,
                    request_receipt=request_receipt,
                )
                _atomic_publish_journal(
                    self._checkpoint_dir / f"{call_key}.response.json",
                    self._response_body(record),
                )
                verified = _load_checkpoint_journal(self._checkpoint_dir)[
                    call_key
                ]
                return self._accept(verified, cache_hit=False)

    def judge(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> tuple[bool, str]:
        """Return the strict official binary verdict and its exact text."""

        completion = self.complete(messages)
        return parse_binary_judge_verdict(completion), completion

    @property
    def usage(self) -> dict[str, int | float]:
        return {
            "logical_calls": self._logical_calls,
            "unique_calls": self._unique_calls,
            "physical_calls": self._physical_calls,
            "checkpoint_hits": self._checkpoint_hits,
            "reported_input_tokens": self._reported_input_tokens,
            "reported_output_tokens": self._reported_output_tokens,
            "reported_total_tokens": self._reported_total_tokens,
            "input_token_proxy": self._input_token_proxy,
            "output_token_proxy": self._output_token_proxy,
            "elapsed_s": self._elapsed_s,
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

    def __enter__(
        self,
    ) -> "RecallGuardedCumulativeSemanticJudgeRuntime":
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "DEFAULT_JUDGE_MAX_NEW_TOKENS",
    "DEFAULT_JUDGE_MODEL",
    "JUDGE_REQUEST_JOURNAL_FORMAT",
    "JUDGE_RESPONSE_JOURNAL_FORMAT",
    "SEMANTIC_JUDGE_RUNTIME_FORMAT",
    "RecallGuardedCumulativeSemanticJudgeRuntime",
    "SemanticJudgeCompletionReport",
    "SemanticJudgeRuntimeIdentity",
]
