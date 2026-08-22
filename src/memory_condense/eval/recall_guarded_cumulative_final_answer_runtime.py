"""Durable, replayable runtime for the locked final Terra answer call.

This module owns only the provider-call boundary.  Retrieval and artifact
orchestration remain separate so a caller must authorize the complete ordered
prompt population before constructing a live runtime.  Every prompt is
validated and counted before a checkpoint directory or provider client is
created.  A physical call is then reserved by an immutable request journal
before network I/O and paired with an immutable response journal afterward.

A request without a response is terminal uncertainty.  It is never retried.
The runtime stores answer text in the response journal, but persists no
request-derived transformer token IDs, K/V tensors, or residual sequences.
External provider persistence is deliberately not certified here.
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
    tokenizer_proxy_identity,
)
from memory_condense.domain.discourse import identity_sha256, quote_sha256
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


FINAL_ANSWER_RUNTIME_FORMAT = (
    "memory-condense-recall-guarded-fixed-stage-final-answer-runtime-v1"
)
FINAL_ANSWER_REQUEST_JOURNAL_FORMAT = (
    "memory-condense-final-answer-call-request-v1"
)
FINAL_ANSWER_RESPONSE_JOURNAL_FORMAT = (
    "memory-condense-final-answer-call-response-v1"
)
FINAL_ANSWER_PROMPT_POPULATION_FORMAT = (
    "memory-condense-final-answer-prompt-population-v1"
)
FINAL_ANSWER_REQUEST_TOKEN_STATE_CONTRACT = (
    "stateless-final-answer-request-token-state-v1"
)

LOCKED_FINAL_ANSWER_MODEL = "openai/codex_sdk/gpt-5.6-terra"
LOCKED_FINAL_ANSWER_GATEWAY_MODEL = "codex_sdk/gpt-5.6-terra"
LOCKED_FINAL_ANSWER_MAX_NEW_TOKENS = 256
LOCKED_FINAL_ANSWER_MAX_PROMPT_TOKENS = 8_000

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_JOURNAL_NAME = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)
_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})


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


def _exact_positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validated_messages(
    messages: Sequence[Mapping[str, str]],
    *,
    label: str,
) -> tuple[dict[str, str], ...]:
    if isinstance(messages, (str, bytes, bytearray)) or not messages:
        raise ValueError(f"{label} must not be empty")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"{label} message {index} must be an object")
        _require_exact_keys(
            message,
            {"role", "content"},
            label=f"{label} message {index}",
        )
        role = message.get("role")
        content = message.get("content")
        if (
            not isinstance(role, str)
            or role not in _ALLOWED_ROLES
            or role.strip() != role
            or not isinstance(content, str)
        ):
            raise ValueError(
                f"{label} message {index} requires an exact supported role "
                "and string content"
            )
        normalized.append({"role": role, "content": content})
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class FinalAnswerPromptPopulation:
    """Provider-free authorization for a complete ordered prompt population."""

    format: str
    logical_prompt_count: int
    unique_prompt_count: int
    ordered_prompt_rows: tuple[Mapping[str, Any], ...]
    prompt_population_sha256: str
    max_prompt_token_proxy: int
    prompt_token_proxy_identity: Mapping[str, Any]
    normalized_prompts: tuple[tuple[Mapping[str, str], ...], ...]

    def identity_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "logical_prompt_count": self.logical_prompt_count,
            "unique_prompt_count": self.unique_prompt_count,
            "ordered_prompt_rows": [dict(row) for row in self.ordered_prompt_rows],
            "prompt_population_sha256": self.prompt_population_sha256,
            "max_prompt_token_proxy": self.max_prompt_token_proxy,
            "prompt_token_proxy_identity": dict(
                self.prompt_token_proxy_identity
            ),
        }


def preflight_final_answer_prompt_population(
    prompt_population: Sequence[Sequence[Mapping[str, str]]],
    *,
    authorized_unique_calls: int,
    max_prompt_tokens: int = LOCKED_FINAL_ANSWER_MAX_PROMPT_TOKENS,
) -> FinalAnswerPromptPopulation:
    """Validate and count every prompt before any journal or network access."""

    authorized = _exact_positive_int(
        authorized_unique_calls,
        label="authorized_unique_calls",
    )
    if (
        type(max_prompt_tokens) is not int
        or max_prompt_tokens != LOCKED_FINAL_ANSWER_MAX_PROMPT_TOKENS
    ):
        raise ValueError(
            "max_prompt_tokens must exactly equal the locked 8000-token cap"
        )
    if isinstance(prompt_population, (str, bytes, bytearray)) or not (
        prompt_population
    ):
        raise ValueError("final-answer prompt population must not be empty")

    normalized: list[tuple[Mapping[str, str], ...]] = []
    ordered_rows: list[dict[str, Any]] = []
    unique: dict[str, tuple[Mapping[str, str], ...]] = {}
    for ordinal, messages in enumerate(prompt_population):
        prompt = _validated_messages(
            messages,
            label=f"final-answer prompt {ordinal}",
        )
        messages_sha = identity_sha256(list(prompt))
        tokens = count_chat_prompt_token_proxy(prompt)
        if tokens > max_prompt_tokens:
            raise ValueError(
                "final-answer prompt population exceeds the locked "
                f"8000-token cap at ordinal {ordinal}: {tokens}"
            )
        previous = unique.setdefault(messages_sha, prompt)
        if previous != prompt:
            raise RuntimeError("final-answer prompt SHA-256 collision")
        normalized.append(prompt)
        ordered_rows.append(
            {
                "ordinal": ordinal,
                "messages_sha256": messages_sha,
                "prompt_token_proxy": tokens,
            }
        )

    if authorized != len(unique):
        raise ValueError(
            "authorized unique final-answer-call cap must exactly equal the "
            f"precomputed requirement ({authorized} != {len(unique)})"
        )
    population_body = {
        "format": FINAL_ANSWER_PROMPT_POPULATION_FORMAT,
        "logical_prompt_count": len(normalized),
        "unique_prompt_count": len(unique),
        "ordered_prompt_rows": ordered_rows,
        "max_prompt_token_proxy": max_prompt_tokens,
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
    }
    population_sha = identity_sha256(population_body)
    return FinalAnswerPromptPopulation(
        format=FINAL_ANSWER_PROMPT_POPULATION_FORMAT,
        logical_prompt_count=len(normalized),
        unique_prompt_count=len(unique),
        ordered_prompt_rows=tuple(ordered_rows),
        prompt_population_sha256=population_sha,
        max_prompt_token_proxy=max_prompt_tokens,
        prompt_token_proxy_identity=tokenizer_proxy_identity(),
        normalized_prompts=tuple(normalized),
    )


@dataclass(frozen=True, slots=True)
class FinalAnswerRuntimeIdentity:
    """The exact Terra route, prompt population, and request policy."""

    format: str
    gateway_url: str
    caller_model: str
    gateway_model: str
    default_max_new_tokens: int
    max_prompt_token_proxy: int
    retries: int
    temperature: None
    authorized_unique_calls: int
    logical_prompt_count: int
    unique_prompt_count: int
    prompt_population_sha256: str
    prompt_token_proxy_identity: Mapping[str, Any]
    campaign_binding: Mapping[str, Any]
    campaign_binding_sha256: str
    request_journal_format: str
    response_journal_format: str
    persisted_request_token_state: bool
    retained_request_token_state_bytes: int
    external_provider_persistence_certified: bool

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FinalAnswerCompletionReport:
    """Immutable text-free metadata for one physical Terra completion."""

    gateway_url: str
    caller_model: str
    gateway_model: str
    call_key_sha256: str
    runtime_identity_sha256: str
    campaign_binding_sha256: str
    prompt_population_sha256: str
    request_journal_sha256: str
    messages_sha256: str
    completion_sha256: str
    response_id: str
    response_model: str
    finish_reason: str
    max_new_tokens: int
    max_prompt_token_proxy: int
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
    messages_sha256: str
    input_token_proxy: int
    runtime_identity_sha256: str
    campaign_binding_sha256: str
    prompt_population_sha256: str
    completion: str
    report: FinalAnswerCompletionReport


def _validate_runtime_identity(
    value: Any,
    *,
    path: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"final-answer runtime identity must be an object: {path}")
    _require_exact_keys(
        value,
        {field.name for field in fields(FinalAnswerRuntimeIdentity)},
        label="final-answer runtime identity",
    )
    expected = {
        "format": FINAL_ANSWER_RUNTIME_FORMAT,
        "gateway_url": CENTRAL_DEV_GATEWAY_URL,
        "caller_model": LOCKED_FINAL_ANSWER_MODEL,
        "gateway_model": LOCKED_FINAL_ANSWER_GATEWAY_MODEL,
        "default_max_new_tokens": LOCKED_FINAL_ANSWER_MAX_NEW_TOKENS,
        "max_prompt_token_proxy": LOCKED_FINAL_ANSWER_MAX_PROMPT_TOKENS,
        "retries": 0,
        "temperature": None,
        "request_journal_format": FINAL_ANSWER_REQUEST_JOURNAL_FORMAT,
        "response_journal_format": FINAL_ANSWER_RESPONSE_JOURNAL_FORMAT,
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "external_provider_persistence_certified": False,
    }
    if any(
        value.get(name) != expected_value
        for name, expected_value in expected.items()
    ):
        raise ValueError(f"stored final-answer runtime policy changed: {path}")
    for name in (
        "authorized_unique_calls",
        "logical_prompt_count",
        "unique_prompt_count",
    ):
        _exact_positive_int(value.get(name), label=f"runtime identity {name}")
    if value["authorized_unique_calls"] != value["unique_prompt_count"]:
        raise ValueError(f"stored final-answer authorization changed: {path}")
    _require_digest(
        value.get("prompt_population_sha256"),
        label="prompt population SHA-256",
    )
    campaign = value.get("campaign_binding")
    if not isinstance(campaign, Mapping):
        raise ValueError(f"stored campaign binding must be an object: {path}")
    campaign_sha = _require_digest(
        value.get("campaign_binding_sha256"),
        label="campaign binding SHA-256",
    )
    if identity_sha256(dict(campaign)) != campaign_sha:
        raise ValueError(f"stored campaign binding does not verify: {path}")
    proxy = value.get("prompt_token_proxy_identity")
    if not isinstance(proxy, Mapping) or dict(proxy) != tokenizer_proxy_identity():
        raise ValueError(f"stored prompt tokenizer identity changed: {path}")
    return dict(value)


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
        label="final-answer request journal",
    )
    if payload["format"] != FINAL_ANSWER_REQUEST_JOURNAL_FORMAT:
        raise ValueError(f"unexpected final-answer request format: {path}")
    call_key = _require_digest(payload["call_key_sha256"], label="call key")
    match = _JOURNAL_NAME.fullmatch(path.name)
    if match is None or match.group("kind") != "request":
        raise ValueError(f"invalid final-answer request journal name: {path}")
    if match.group("key") != call_key:
        raise ValueError(f"final-answer request filename changed: {path}")
    key_payload = payload["call_key_payload"]
    if not isinstance(key_payload, Mapping):
        raise ValueError(f"final-answer call key payload must be an object: {path}")
    _require_exact_keys(
        key_payload,
        {
            "messages_sha256",
            "input_token_proxy",
            "runtime_identity_sha256",
            "max_new_tokens",
            "campaign_binding_sha256",
            "prompt_population_sha256",
        },
        label="final-answer call key payload",
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
    population_sha = _require_digest(
        key_payload["prompt_population_sha256"],
        label="prompt population SHA-256",
    )
    input_tokens = key_payload["input_token_proxy"]
    if (
        type(input_tokens) is not int
        or input_tokens < 0
        or input_tokens > LOCKED_FINAL_ANSWER_MAX_PROMPT_TOKENS
    ):
        raise ValueError(f"final-answer input token proxy is invalid: {path}")
    if key_payload["max_new_tokens"] != LOCKED_FINAL_ANSWER_MAX_NEW_TOKENS:
        raise ValueError(f"final-answer max_new_tokens changed: {path}")

    runtime_identity = _validate_runtime_identity(
        payload["runtime_identity"],
        path=path,
    )
    campaign = payload["campaign_binding"]
    if not isinstance(campaign, Mapping):
        raise ValueError(f"final-answer campaign binding must be an object: {path}")
    if (
        identity_sha256(runtime_identity) != runtime_sha
        or identity_sha256(dict(campaign)) != campaign_sha
        or runtime_identity["campaign_binding"] != dict(campaign)
        or runtime_identity["campaign_binding_sha256"] != campaign_sha
        or runtime_identity["prompt_population_sha256"] != population_sha
    ):
        raise ValueError(f"final-answer request binding does not verify: {path}")
    if identity_sha256(dict(key_payload)) != call_key:
        raise ValueError(f"final-answer call key does not verify: {path}")
    if payload["journal_sha256"] != receipt:
        raise ValueError(f"final-answer request journal receipt changed: {path}")
    return dict(payload)


def _validated_report(
    value: Any,
    *,
    request: Mapping[str, Any],
    request_receipt: str,
    completion: str,
    path: Path,
) -> FinalAnswerCompletionReport:
    if not isinstance(value, Mapping):
        raise ValueError(f"stored final-answer report must be an object: {path}")
    _require_exact_keys(
        value,
        {field.name for field in fields(FinalAnswerCompletionReport)},
        label="stored final-answer report",
    )
    try:
        report = FinalAnswerCompletionReport(**dict(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stored final-answer report is invalid: {path}") from exc
    booleans = (
        report.reported_usage_available,
        report.reported_input_tokens_available,
        report.reported_output_tokens_available,
        report.reported_total_tokens_available,
        report.cache_hit,
        report.physical_call,
    )
    if any(type(item) is not bool for item in booleans):
        raise ValueError(f"stored final-answer report booleans changed: {path}")
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
        raise ValueError(f"stored final-answer report counters changed: {path}")
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
        raise ValueError(f"stored final-answer usage availability changed: {path}")
    if report.reported_usage_available != any(availability):
        raise ValueError(f"stored final-answer usage summary changed: {path}")

    key_payload = request["call_key_payload"]
    runtime_identity = request["runtime_identity"]
    if (
        report.call_key_sha256 != request["call_key_sha256"]
        or report.runtime_identity_sha256
        != key_payload["runtime_identity_sha256"]
        or report.campaign_binding_sha256
        != key_payload["campaign_binding_sha256"]
        or report.prompt_population_sha256
        != key_payload["prompt_population_sha256"]
        or report.request_journal_sha256 != request_receipt
        or report.messages_sha256 != key_payload["messages_sha256"]
        or report.input_token_proxy != key_payload["input_token_proxy"]
        or report.max_new_tokens != key_payload["max_new_tokens"]
        or report.max_prompt_token_proxy
        != runtime_identity["max_prompt_token_proxy"]
        or report.gateway_url != runtime_identity["gateway_url"]
        or report.caller_model != runtime_identity["caller_model"]
        or report.gateway_model != runtime_identity["gateway_model"]
        or report.completion_sha256 != quote_sha256(completion)
        or report.output_token_proxy != count_tokens(completion)
        or report.retries != 0
        or report.cache_hit is not False
        or report.physical_call is not True
    ):
        raise ValueError(f"stored final-answer report binding changed: {path}")
    if report.input_token_proxy > report.max_prompt_token_proxy:
        raise ValueError(f"stored final-answer prompt exceeds its cap: {path}")
    if (
        isinstance(report.elapsed_s, bool)
        or not isinstance(report.elapsed_s, (int, float))
        or not math.isfinite(float(report.elapsed_s))
        or report.elapsed_s < 0
    ):
        raise ValueError(f"stored final-answer elapsed time changed: {path}")
    if (
        report.cumulative_logical_calls < 1
        or report.cumulative_unique_calls < 1
        or report.cumulative_physical_calls < 1
    ):
        raise ValueError(f"stored final-answer cumulative counts changed: {path}")
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
        label="final-answer response journal",
    )
    if payload["format"] != FINAL_ANSWER_RESPONSE_JOURNAL_FORMAT:
        raise ValueError(f"unexpected final-answer response format: {path}")
    call_key = _require_digest(payload["call_key_sha256"], label="call key")
    match = _JOURNAL_NAME.fullmatch(path.name)
    if match is None or match.group("kind") != "response":
        raise ValueError(f"invalid final-answer response journal name: {path}")
    if match.group("key") != call_key or call_key != request["call_key_sha256"]:
        raise ValueError(f"final-answer response filename changed: {path}")
    if payload["request_journal_sha256"] != request_receipt:
        raise ValueError(f"final-answer response request binding changed: {path}")
    completion = payload["completion"]
    if (
        not isinstance(completion, str)
        or not completion
        or completion.strip() != completion
    ):
        raise ValueError(f"stored final-answer completion is invalid: {path}")
    report = _validated_report(
        payload["report"],
        request=request,
        request_receipt=request_receipt,
        completion=completion,
        path=path,
    )
    key_payload = request["call_key_payload"]
    return _JournalRecord(
        call_key_sha256=call_key,
        request_journal_sha256=request_receipt,
        response_journal_sha256=receipt,
        messages_sha256=str(key_payload["messages_sha256"]),
        input_token_proxy=int(key_payload["input_token_proxy"]),
        runtime_identity_sha256=str(key_payload["runtime_identity_sha256"]),
        campaign_binding_sha256=str(key_payload["campaign_binding_sha256"]),
        prompt_population_sha256=str(
            key_payload["prompt_population_sha256"]
        ),
        completion=completion,
        report=report,
    )


def _load_checkpoint_journal(root: Path) -> dict[str, _JournalRecord]:
    """Verify the entire journal inventory and reject uncertain calls."""

    requests: dict[str, Path] = {}
    responses: dict[str, Path] = {}
    for path in root.glob("*.json"):
        match = _JOURNAL_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected JSON file in final-answer journal: {path}")
        target = requests if match.group("kind") == "request" else responses
        key = match.group("key")
        if key in target:
            raise ValueError(f"duplicate final-answer journal key: {path}")
        target[key] = path
    orphan = set(responses) - set(requests)
    incomplete = set(requests) - set(responses)
    if orphan:
        raise ValueError(
            "final-answer response has no request reservation: " + min(orphan)
        )
    if incomplete:
        raise RuntimeError(
            "final-answer request has no response; refusing an unsafe retry: "
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


class RecallGuardedCumulativeFinalAnswerRuntime:
    """Call or replay locked Terra over one preflighted prompt population."""

    def __init__(
        self,
        *,
        checkpoint_dir: str | Path,
        campaign_binding: Mapping[str, Any],
        prompt_population: Sequence[Sequence[Mapping[str, str]]],
        authorized_unique_calls: int,
        api_key: str | None = None,
        caller_model: str = LOCKED_FINAL_ANSWER_MODEL,
        max_new_tokens: int = LOCKED_FINAL_ANSWER_MAX_NEW_TOKENS,
        max_prompt_tokens: int = LOCKED_FINAL_ANSWER_MAX_PROMPT_TOKENS,
        replay_only: bool = False,
        client: Any | None = None,
    ) -> None:
        # Complete-population prompt validation is intentionally first.  A bad
        # late prompt therefore cannot leave an earlier paid call behind.
        population = preflight_final_answer_prompt_population(
            prompt_population,
            authorized_unique_calls=authorized_unique_calls,
            max_prompt_tokens=max_prompt_tokens,
        )
        if (
            type(max_new_tokens) is not int
            or max_new_tokens != LOCKED_FINAL_ANSWER_MAX_NEW_TOKENS
        ):
            raise ValueError(
                "max_new_tokens must exactly equal the locked 256-token allowance"
            )
        if caller_model != LOCKED_FINAL_ANSWER_MODEL:
            raise ValueError("caller_model must be the locked Terra responder")
        if _gateway_model(caller_model) != LOCKED_FINAL_ANSWER_GATEWAY_MODEL:
            raise ValueError("locked Terra caller and gateway models conflict")
        if type(replay_only) is not bool:
            raise TypeError("replay_only must be boolean")
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
        identity = FinalAnswerRuntimeIdentity(
            format=FINAL_ANSWER_RUNTIME_FORMAT,
            gateway_url=CENTRAL_DEV_GATEWAY_URL,
            caller_model=LOCKED_FINAL_ANSWER_MODEL,
            gateway_model=LOCKED_FINAL_ANSWER_GATEWAY_MODEL,
            default_max_new_tokens=LOCKED_FINAL_ANSWER_MAX_NEW_TOKENS,
            max_prompt_token_proxy=LOCKED_FINAL_ANSWER_MAX_PROMPT_TOKENS,
            retries=0,
            temperature=None,
            authorized_unique_calls=authorized_unique_calls,
            logical_prompt_count=population.logical_prompt_count,
            unique_prompt_count=population.unique_prompt_count,
            prompt_population_sha256=population.prompt_population_sha256,
            prompt_token_proxy_identity=dict(
                population.prompt_token_proxy_identity
            ),
            campaign_binding=safe_binding,
            campaign_binding_sha256=binding_sha,
            request_journal_format=FINAL_ANSWER_REQUEST_JOURNAL_FORMAT,
            response_journal_format=FINAL_ANSWER_RESPONSE_JOURNAL_FORMAT,
            persisted_request_token_state=False,
            retained_request_token_state_bytes=0,
            external_provider_persistence_certified=False,
        )
        runtime_sha = identity_sha256(identity.model_dump())
        allowed: dict[str, tuple[tuple[Mapping[str, str], ...], int]] = {}
        for messages, row in zip(
            population.normalized_prompts,
            population.ordered_prompt_rows,
            strict=True,
        ):
            messages_sha = str(row["messages_sha256"])
            previous = allowed.setdefault(
                messages_sha,
                (messages, int(row["prompt_token_proxy"])),
            )
            if previous[0] != messages:
                raise RuntimeError("final-answer prompt SHA-256 collision")

        # No path mutation happens until the entire population, route, and
        # campaign identity have passed validation.
        root = Path(checkpoint_dir)
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("checkpoint_dir must be a regular directory")
        with _checkpoint_lock(root):
            existing = _load_checkpoint_journal(root)
        allowed_keys = {
            identity_sha256(
                {
                    "messages_sha256": messages_sha,
                    "input_token_proxy": input_tokens,
                    "runtime_identity_sha256": runtime_sha,
                    "max_new_tokens": LOCKED_FINAL_ANSWER_MAX_NEW_TOKENS,
                    "campaign_binding_sha256": binding_sha,
                    "prompt_population_sha256": (
                        population.prompt_population_sha256
                    ),
                }
            )
            for messages_sha, (_messages, input_tokens) in allowed.items()
        }
        unexpected = set(existing) - allowed_keys
        if unexpected:
            raise ValueError(
                "final-answer journal contains a call outside the authorized "
                "prompt population: "
                + min(unexpected)
            )
        if len(existing) > authorized_unique_calls:
            raise RuntimeError(
                "verified final-answer journal exceeds the authorized "
                "unique-call budget"
            )

        self.identity = identity
        self.prompt_population = population
        self._runtime_identity_sha256 = runtime_sha
        self._client = (
            None
            if replay_only
            else client if client is not None else _new_gateway_client(secret)
        )
        self._checkpoint_dir = root
        self._authorized_unique_calls = authorized_unique_calls
        self._replay_only = replay_only
        self._allowed_prompts = allowed
        self._allowed_call_keys = allowed_keys
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
        self.last_completion_report: FinalAnswerCompletionReport | None = None
        self.last_journal_record: Mapping[str, Any] | None = None

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("final-answer runtime is closed")

    def _key_payload(
        self,
        *,
        messages_sha256: str,
        input_token_proxy: int,
    ) -> dict[str, Any]:
        return {
            "messages_sha256": messages_sha256,
            "input_token_proxy": input_token_proxy,
            "runtime_identity_sha256": self._runtime_identity_sha256,
            "max_new_tokens": LOCKED_FINAL_ANSWER_MAX_NEW_TOKENS,
            "campaign_binding_sha256": self.identity.campaign_binding_sha256,
            "prompt_population_sha256": (
                self.identity.prompt_population_sha256
            ),
        }

    def _request_body(
        self,
        *,
        call_key: str,
        key_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "format": FINAL_ANSWER_REQUEST_JOURNAL_FORMAT,
            "call_key_sha256": call_key,
            "call_key_payload": dict(key_payload),
            "runtime_identity": self.identity.model_dump(),
            "campaign_binding": dict(self.identity.campaign_binding),
        }

    def _invoke(
        self,
        messages: list[dict[str, str]],
        *,
        call_key: str,
        request_receipt: str,
        input_token_proxy: int,
    ) -> _JournalRecord:
        request = {
            "model": LOCKED_FINAL_ANSWER_GATEWAY_MODEL,
            "messages": messages,
            "max_tokens": LOCKED_FINAL_ANSWER_MAX_NEW_TOKENS,
        }
        self._physical_calls += 1
        started = time.perf_counter()
        assert self._client is not None
        response = self._client.chat.completions.create(**request)
        elapsed = time.perf_counter() - started
        choices = _field(response, "choices", ())
        if not choices:
            raise RuntimeError("final-answer provider returned no completion choices")
        choice = choices[0]
        message = _field(choice, "message")
        completion = str(_field(message, "content", "") or "").strip()
        if not completion:
            raise RuntimeError("final-answer provider returned no answer text")
        usage = _field(response, "usage")
        input_tokens, input_available = _reported_count(usage, "prompt_tokens")
        output_tokens, output_available = _reported_count(
            usage, "completion_tokens"
        )
        total_tokens, total_available = _reported_count(usage, "total_tokens")
        report = FinalAnswerCompletionReport(
            gateway_url=CENTRAL_DEV_GATEWAY_URL,
            caller_model=LOCKED_FINAL_ANSWER_MODEL,
            gateway_model=LOCKED_FINAL_ANSWER_GATEWAY_MODEL,
            call_key_sha256=call_key,
            runtime_identity_sha256=self._runtime_identity_sha256,
            campaign_binding_sha256=self.identity.campaign_binding_sha256,
            prompt_population_sha256=self.identity.prompt_population_sha256,
            request_journal_sha256=request_receipt,
            messages_sha256=identity_sha256(messages),
            completion_sha256=quote_sha256(completion),
            response_id=str(_field(response, "id", "") or ""),
            response_model=str(_field(response, "model", "") or ""),
            finish_reason=str(_field(choice, "finish_reason", "") or ""),
            max_new_tokens=LOCKED_FINAL_ANSWER_MAX_NEW_TOKENS,
            max_prompt_token_proxy=LOCKED_FINAL_ANSWER_MAX_PROMPT_TOKENS,
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
            messages_sha256=identity_sha256(messages),
            input_token_proxy=input_token_proxy,
            runtime_identity_sha256=self._runtime_identity_sha256,
            campaign_binding_sha256=self.identity.campaign_binding_sha256,
            prompt_population_sha256=self.identity.prompt_population_sha256,
            completion=completion,
            report=report,
        )

    @staticmethod
    def _response_body(record: _JournalRecord) -> dict[str, Any]:
        return {
            "format": FINAL_ANSWER_RESPONSE_JOURNAL_FORMAT,
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
            "messages_sha256": record.messages_sha256,
            "completion_sha256": record.report.completion_sha256,
            # Preserve the immutable physical-call report so live and replay
            # orchestration can publish byte-identical result artifacts.
            "completion_report": record.report.model_dump(),
        }
        return record.completion

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        """Complete one authorized prompt under the frozen Terra contract."""

        if max_new_tokens is not None and (
            type(max_new_tokens) is not int
            or max_new_tokens != LOCKED_FINAL_ANSWER_MAX_NEW_TOKENS
        ):
            raise ValueError(
                "max_new_tokens must exactly equal the locked 256-token allowance"
            )
        normalized = _validated_messages(messages, label="final-answer prompt")
        messages_sha = identity_sha256(list(normalized))
        allowed = self._allowed_prompts.get(messages_sha)
        if allowed is None or allowed[0] != normalized:
            raise ValueError(
                "final-answer prompt is outside the preflighted population"
            )
        input_tokens = count_chat_prompt_token_proxy(normalized)
        if input_tokens != allowed[1]:
            raise RuntimeError(
                "final-answer prompt token count changed after population preflight"
            )
        if input_tokens > LOCKED_FINAL_ANSWER_MAX_PROMPT_TOKENS:
            raise RuntimeError("final-answer prompt exceeds the locked 8000-token cap")

        with self._lock:
            self._require_open()
            key_payload = self._key_payload(
                messages_sha256=messages_sha,
                input_token_proxy=input_tokens,
            )
            call_key = identity_sha256(key_payload)
            if call_key not in self._allowed_call_keys:
                raise RuntimeError("final-answer call key escaped its authorization")
            with _checkpoint_lock(self._checkpoint_dir):
                records = _load_checkpoint_journal(self._checkpoint_dir)
                unexpected = set(records) - self._allowed_call_keys
                if unexpected:
                    raise ValueError(
                        "final-answer journal contains an unauthorized call: "
                        + min(unexpected)
                    )
                cached = records.get(call_key)
                if cached is not None:
                    return self._accept(cached, cache_hit=True)
                if self._replay_only:
                    raise RuntimeError(
                        "replay-only final-answer runtime has no verified "
                        "response for call: "
                        + call_key
                    )
                if len(records) >= self._authorized_unique_calls:
                    raise RuntimeError(
                        "authorized unique final-answer-call budget exhausted"
                    )
                request_receipt = _atomic_publish_journal(
                    self._checkpoint_dir / f"{call_key}.request.json",
                    self._request_body(
                        call_key=call_key,
                        key_payload=key_payload,
                    ),
                )
                record = self._invoke(
                    [dict(row) for row in normalized],
                    call_key=call_key,
                    request_receipt=request_receipt,
                    input_token_proxy=input_tokens,
                )
                _atomic_publish_journal(
                    self._checkpoint_dir / f"{call_key}.response.json",
                    self._response_body(record),
                )
                verified = _load_checkpoint_journal(self._checkpoint_dir)[
                    call_key
                ]
                return self._accept(verified, cache_hit=False)

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

    def request_token_state_receipt(self) -> dict[str, Any]:
        """Return the explicit local zero-persisted-transformer-state receipt."""

        return {
            "contract": FINAL_ANSWER_REQUEST_TOKEN_STATE_CONTRACT,
            "persisted_request_token_state": False,
            "retained_request_token_state_bytes": 0,
            "request_token_state_evidence_kind": (
                "stateless_journaled_provider_completion_runtime"
            ),
            "external_provider_persistence_certified": False,
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "RecallGuardedCumulativeFinalAnswerRuntime":
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "FINAL_ANSWER_PROMPT_POPULATION_FORMAT",
    "FINAL_ANSWER_REQUEST_JOURNAL_FORMAT",
    "FINAL_ANSWER_REQUEST_TOKEN_STATE_CONTRACT",
    "FINAL_ANSWER_RESPONSE_JOURNAL_FORMAT",
    "FINAL_ANSWER_RUNTIME_FORMAT",
    "LOCKED_FINAL_ANSWER_GATEWAY_MODEL",
    "LOCKED_FINAL_ANSWER_MAX_NEW_TOKENS",
    "LOCKED_FINAL_ANSWER_MAX_PROMPT_TOKENS",
    "LOCKED_FINAL_ANSWER_MODEL",
    "FinalAnswerCompletionReport",
    "FinalAnswerPromptPopulation",
    "FinalAnswerRuntimeIdentity",
    "RecallGuardedCumulativeFinalAnswerRuntime",
    "preflight_final_answer_prompt_population",
]
