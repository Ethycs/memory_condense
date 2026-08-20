"""Pure, fail-closed validation for campaign identities and question rows."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from memory_condense.domain._discourse_identity import (
    canonical_json as _canonical_json,
)
from memory_condense.eval.campaign_models import (
    CampaignMergeError,
    LockedValidationPlan,
)

_HASH_FIELDS = (
    "dataset_sha256",
    "split_manifest_sha256",
    "implementation_sha256",
    "environment_lock_sha256",
    "policy_manifest_sha256",
)
_QUESTION_ERROR_FIELDS = {
    "error",
    "errors",
    "exception",
    "provider_error",
    "responder_error",
    "judge_error",
}
_BINARY_JUDGE_VERDICT = re.compile(
    r"^\s*(CORRECT|INCORRECT)\b(?P<remainder>.*)$",
    re.IGNORECASE | re.DOTALL,
)


def _json_constant(value: str) -> None:
    raise CampaignMergeError(f"non-finite JSON number {value!r} is not allowed")


def _file_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignMergeError(f"{label} must be a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CampaignMergeError(f"{label} must be a JSON array")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignMergeError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    digest = _require_nonempty_string(value, label)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise CampaignMergeError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise CampaignMergeError(f"{label} must be a boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CampaignMergeError(f"{label} must be an integer >= {minimum}")
    return value


def _require_float(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignMergeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignMergeError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise CampaignMergeError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise CampaignMergeError(f"{label} must be <= {maximum}")
    return result

def _load_report(path: Path) -> tuple[dict[str, Any], str, str, str]:
    resolved = path.resolve()
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise CampaignMergeError(f"cannot read report {resolved}: {exc}") from exc
    digest = _file_sha256(payload)
    try:
        report = json.loads(payload, parse_constant=_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignMergeError(f"invalid JSON report {resolved}: {exc}") from exc
    return (
        _require_mapping(report, f"report {resolved}"),
        digest,
        resolved.as_posix(),
        resolved.name,
    )


def _identity(report: dict[str, Any], label: str) -> dict[str, Any]:
    config = _require_mapping(report.get("config"), f"{label}.config")
    identity: dict[str, Any] = {
        field: _require_sha256(report.get(field), f"{label}.{field}")
        for field in _HASH_FIELDS
    }
    split = _require_nonempty_string(
        report.get("benchmark_split"), f"{label}.benchmark_split"
    )
    if split != "validation":
        raise CampaignMergeError(
            f"{label}.benchmark_split must be 'validation', got {split!r}"
        )
    identity["benchmark_split"] = split
    identity["benchmark"] = _require_nonempty_string(
        report.get("benchmark"), f"{label}.benchmark"
    )
    identity["chunker_config"] = _require_mapping(
        config.get("chunker"), f"{label}.config.chunker"
    )
    identity["retrieval_config"] = _require_mapping(
        config.get("retrieval"), f"{label}.config.retrieval"
    )
    identity["responder_model"] = _require_nonempty_string(
        config.get("responder_model"), f"{label}.config.responder_model"
    )
    identity["judge_model"] = _require_nonempty_string(
        config.get("judge_model"), f"{label}.config.judge_model"
    )
    identity["max_prompt_tokens"] = _require_int(
        config.get("max_prompt_tokens"),
        f"{label}.config.max_prompt_tokens",
        minimum=1,
    )
    embedding_device = config.get("embedding_device")
    if embedding_device is not None and not isinstance(embedding_device, str):
        raise CampaignMergeError(
            f"{label}.config.embedding_device must be a string or null"
        )
    identity["embedding_device"] = embedding_device
    identity["recent_window"] = _require_int(
        config.get("recent_window"), f"{label}.config.recent_window"
    )
    raw_proxy_identity = report.get("prompt_token_proxy_identity")
    identity["prompt_token_proxy_identity"] = (
        {
            "schema": "legacy-cl100k-message-content-only-v0",
            "encoding": "cl100k_base",
        }
        if raw_proxy_identity is None
        else dict(
            _require_mapping(
                raw_proxy_identity,
                f"{label}.prompt_token_proxy_identity",
            )
        )
    )
    identity["responder_output_token_reserve"] = _require_int(
        report.get("responder_output_token_reserve", 0),
        f"{label}.responder_output_token_reserve",
    )
    raw_protocol = report.get("evaluation_protocol", {})
    protocol = _require_mapping(raw_protocol, f"{label}.evaluation_protocol")
    protocol_common = dict(protocol)
    protocol_common.pop("sample_offset", None)
    identity["evaluation_protocol"] = protocol_common
    return identity


def _ensure_same_identity(
    expected: dict[str, Any], actual: dict[str, Any], label: str
) -> None:
    for field, expected_value in expected.items():
        actual_value = actual[field]
        if _canonical_json(actual_value) != _canonical_json(expected_value):
            raise CampaignMergeError(
                f"locked campaign identity drift in {label}.{field}: "
                f"expected {_canonical_json(expected_value)}, "
                f"got {_canonical_json(actual_value)}"
            )


def _assert_policy_retrieval_identity(
    identity: dict[str, Any],
    plan: LockedValidationPlan,
    label: str,
) -> None:
    """Match a report's nested config to the policy's flattened identity."""

    retrieval = identity["retrieval_config"]
    chunker = identity["chunker_config"]
    actual: dict[str, Any] = {}
    for field in plan.retrieval:
        if field == "chunker_min_tokens":
            value = chunker.get("min_tokens")
        elif field == "chunker_max_tokens":
            value = chunker.get("max_tokens")
        elif field == "max_prompt_tokens":
            value = identity["max_prompt_tokens"]
        else:
            value = retrieval.get(field)
        actual[field] = value
    if _canonical_json(actual) != _canonical_json(plan.retrieval):
        raise CampaignMergeError(
            f"{label}.config does not match the frozen retrieval policy"
        )


def _has_error(question: dict[str, Any]) -> str | None:
    for field, value in question.items():
        is_error_field = field in _QUESTION_ERROR_FIELDS or field.endswith("_error")
        if is_error_field and value not in (None, "", False, 0, [], {}):
            return field
    status = question.get("status")
    if isinstance(status, str) and status.casefold() in {
        "error",
        "failed",
        "provider_error",
    }:
        return "status"
    return None


def _locked_judge_verdict(value: Any, label: str) -> bool:
    reasoning = _require_nonempty_string(value, label)
    match = _BINARY_JUDGE_VERDICT.match(reasoning)
    if match is None:
        raise CampaignMergeError(f"{label} has no exact binary judge verdict")
    remainder = match.group("remainder").lstrip(" \t\r\n,.:;-—")
    if remainder.casefold().startswith("or ") or remainder.startswith("/"):
        raise CampaignMergeError(f"{label} contains an ambiguous judge verdict")
    return match.group(1).casefold() == "correct"


def _validate_usage(value: Any, label: str) -> dict[str, int | float]:
    usage = _require_mapping(value, label)
    return {
        "input_tokens": _require_int(
            usage.get("input_tokens"), f"{label}.input_tokens"
        ),
        "output_tokens": _require_int(
            usage.get("output_tokens"), f"{label}.output_tokens"
        ),
        "cache_read_input_tokens": _require_int(
            usage.get("cache_read_input_tokens"),
            f"{label}.cache_read_input_tokens",
        ),
        "elapsed_s": _require_float(
            usage.get("elapsed_s"), f"{label}.elapsed_s", minimum=0.0
        ),
        "calls": _require_int(usage.get("calls"), f"{label}.calls"),
    }


def _validate_question(
    value: Any,
    label: str,
    *,
    prompt_cap: int,
    output_token_reserve: int,
    require_proxy_fields: bool,
) -> tuple[dict[str, Any], dict[str, int | float], dict[str, int | float]]:
    question = dict(_require_mapping(value, label))
    question_id = _require_nonempty_string(
        question.get("question_id"), f"{label}.question_id"
    )
    error_field = _has_error(question)
    if error_field is not None:
        raise CampaignMergeError(
            f"{label} ({question_id}) contains a per-question error in {error_field}"
        )
    _require_float(question.get("f1"), f"{label}.f1", minimum=0.0, maximum=1.0)
    _require_bool(question.get("exact_match"), f"{label}.exact_match")
    _require_bool(question.get("judge_correct"), f"{label}.judge_correct")
    _require_int(question.get("context_tokens"), f"{label}.context_tokens")
    legacy_prompt_tokens = _require_int(
        question.get("prompt_tokens"), f"{label}.prompt_tokens"
    )
    raw_prompt_proxy = question.get("prompt_token_proxy")
    if raw_prompt_proxy is None:
        if require_proxy_fields:
            raise CampaignMergeError(
                f"{label}.prompt_token_proxy is required for locked validation"
            )
        prompt_proxy = legacy_prompt_tokens
    else:
        prompt_proxy = _require_int(
            raw_prompt_proxy,
            f"{label}.prompt_token_proxy",
        )
        if prompt_proxy != legacy_prompt_tokens:
            raise CampaignMergeError(
                f"{label}.prompt_tokens compatibility alias disagrees with "
                "prompt_token_proxy"
            )
    if prompt_proxy > prompt_cap:
        raise CampaignMergeError(
            f"{label} ({question_id}) exceeds the locked prompt cap: "
            f"{prompt_proxy} > {prompt_cap}"
        )
    responder_usage = _validate_usage(
        question.get("responder_usage"), f"{label}.responder_usage"
    )
    judge_usage = _validate_usage(
        question.get("judge_usage"), f"{label}.judge_usage"
    )
    expected_provider_compliance = (
        None
        if int(responder_usage["input_tokens"]) <= 0
        else int(responder_usage["input_tokens"]) <= prompt_cap
    )
    reported_provider_compliance = question.get(
        "provider_prompt_budget_compliant"
    )
    if require_proxy_fields or reported_provider_compliance is not None:
        if reported_provider_compliance is not None:
            _require_bool(
                reported_provider_compliance,
                f"{label}.provider_prompt_budget_compliant",
            )
        if reported_provider_compliance != expected_provider_compliance:
            raise CampaignMergeError(
                f"{label}.provider_prompt_budget_compliant disagrees with "
                "provider-reported input usage"
            )
    if expected_provider_compliance is False:
        raise CampaignMergeError(
            f"{label} ({question_id}) provider input usage exceeds the locked "
            f"prompt cap: {responder_usage['input_tokens']} > {prompt_cap}"
        )
    raw_reserve = question.get("responder_output_token_reserve")
    raw_request_proxy = question.get("request_token_proxy")
    if require_proxy_fields and (raw_reserve is None or raw_request_proxy is None):
        raise CampaignMergeError(
            f"{label} must report responder output reserve and request-token proxy"
        )
    if raw_reserve is not None:
        reported_reserve = _require_int(
            raw_reserve,
            f"{label}.responder_output_token_reserve",
        )
        if reported_reserve != output_token_reserve:
            raise CampaignMergeError(
                f"{label}.responder_output_token_reserve disagrees with the "
                "locked protocol"
            )
    if raw_request_proxy is not None:
        request_proxy = _require_int(
            raw_request_proxy,
            f"{label}.request_token_proxy",
        )
        if request_proxy != prompt_proxy + output_token_reserve:
            raise CampaignMergeError(
                f"{label}.request_token_proxy does not include the exact "
                "prompt proxy plus output reserve"
            )
    question["prompt_token_proxy"] = prompt_proxy
    question["prompt_tokens"] = prompt_proxy
    question["provider_prompt_budget_compliant"] = expected_provider_compliance
    return question, responder_usage, judge_usage
