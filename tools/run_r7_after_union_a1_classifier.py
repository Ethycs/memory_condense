#!/usr/bin/env python3
"""Run the sealed R7 A1 after-union Terra classifier lifecycle.

The lifecycle consumes only the exact sealed A1 v2 construction/replay pair.
It derives the complete classifier request population from that pair, sends
only each request's already-sealed ``messages`` to Terra, and publishes the
``DISPOSITIONS_FORMAT`` artifact consumed by the A1 adapter.

Provider execution requires a separate release, owns a distinct immutable
checkpoint journal, and has zero retries.  Materialization and replay use only
complete checkpoint pairs.  Missing, reordered, malformed, or unknown leaf
decisions are never interpreted as exclusion: they fail materialization, while
an explicit ``unresolved`` decision remains the adapter's fail-open U state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from memory_condense.domain._tokenizer import (  # noqa: E402
    count_chat_prompt_token_proxy,
)
from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRecord,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from tools import run_r7_after_union_a1 as a1_cli  # noqa: E402
from tools.matched_eval import live  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval import r7_after_union_a1 as a1  # noqa: E402


FORMAT = "memory-condense-r7-after-union-a1-classifier-lifecycle-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
RELEASE_FORMAT = f"{FORMAT}-provider-release-v1"
REQUEST_ROW_FORMAT = f"{FORMAT}-request-row-v1"
QUESTION_ROW_FORMAT = f"{FORMAT}-question-row-v1"
RESPONSE_ROW_FORMAT = f"{FORMAT}-response-row-v1"
JOURNAL_OWNER_FORMAT = f"{FORMAT}-journal-owner-v1"
MODEL_PROMPT_POPULATION_FORMAT = f"{FORMAT}-model-prompt-population-v1"
CLASSIFIER_ID = "r7-a1-terra-after-union-leaf-relevance-strict-json-v1"

PREFLIGHT_NAME = "r7-after-union-a1-classifier-preflight-v1.json"
RELEASE_NAME = "r7-after-union-a1-classifier-provider-release-v1.json"
DISPOSITIONS_NAME = "r7-after-union-a1-classifier-dispositions-v1.json"
REPLAY_NAME = "r7-after-union-a1-classifier-dispositions-replay-v1.json"
CHECKPOINT_DIR_NAME = "terra-r7-after-union-a1-classifier-v1-calls"

DEFAULT_A1_ROOT = a1_cli.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = DEFAULT_A1_ROOT / "terra-classifier-v1"
EXPECTED_A1_SHA256 = (
    "ad22a5b9c8d790f843de55c7653abdb9cbda9a7afb2661a67f3e50846bc37dca"
)
DEFAULT_MODEL = live.DEFAULT_TERRA_GATEWAY_MODEL
DEFAULT_GATEWAY_URL = live.DEFAULT_GATEWAY_URL
DEFAULT_API_KEY_ENV = live.DEFAULT_API_KEY_ENV
DEFAULT_MAX_CONCURRENCY = 4
MAX_NEW_TOKENS = a1.CLASSIFIER_OUTPUT_TOKEN_RESERVE
MAX_PROMPT_TOKENS = a1.MAX_TOTAL_TOKENS - MAX_NEW_TOKENS

_JOURNAL_FILENAME_RE = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)
_REQUEST_KEYS = {
    "answer_output_token_reserve",
    "boundary_labels_for_scheduling_only",
    "classifier_output_token_reserve",
    "format",
    "hard_total_token_cap",
    "leaf_handle_ids",
    "messages",
    "payload_class",
    "prompt_token_proxy",
    "question_sha256",
    "request_sha256",
    "selected_union_population_sha256",
    "shard_id",
    "shard_population_sha256",
    "topic_labels_for_scheduling_only",
    "topic_labels_have_exclusion_authority",
}
_CLASSIFIER_PROMPT_KEYS = {
    "cross_boundary_edges",
    "dated_question",
    "format",
    "leaf_population",
    "operator_spec",
    "response_schema",
    "selected_union_population_sha256",
    "topic_labels_have_exclusion_authority",
}
_PROMPT_LEAF_KEYS = {
    "cross_boundary_edge_ids",
    "group_handle",
    "handle_id",
    "leaf_receipt_sha256",
    "summary",
}
_REQUEST_ROW_KEYS = {
    "classifier_request",
    "classifier_request_sha256",
    "format",
    "leaf_bindings",
    "messages_sha256",
    "question_sha256",
    "request_row_receipt_sha256",
    "selected_union_population_sha256",
}
_QUESTION_ROW_KEYS = {
    "classifier_request_sha256s",
    "format",
    "leaf_bindings",
    "question_row_receipt_sha256",
    "question_sha256",
    "selected_union_population_sha256",
}
_PREFLIGHT_KEYS = {
    "a1_classifier_request_population_sha256",
    "a1_construction_artifact_sha256",
    "a1_construction_identity_sha256",
    "a1_replay_artifact_sha256",
    "classifier_id",
    "classifier_payload_class",
    "derived_provider_call_count",
    "format",
    "gateway_url",
    "gold_loaded",
    "max_concurrency",
    "max_new_tokens",
    "max_prompt_tokens",
    "model",
    "model_prompt_population_sha256",
    "ordered_classifier_request_population_sha256",
    "ordinal_cli_routing_available",
    "physical_provider_calls",
    "preflight_identity_sha256",
    "production_ordinal_routing_enabled",
    "prompt_population_sha256",
    "question_count",
    "question_population_sha256",
    "question_rows",
    "request_rows",
    "retained_transformer_token_state_bytes",
    "runtime_firewall",
    "selected_leaf_count",
    "source_artifact_sha256",
    "source_replay_artifact_sha256",
}
_RELEASE_KEYS = {
    "a1_construction_artifact_sha256",
    "approval_opt_in",
    "checkpoint_root",
    "checkpoint_root_sha256",
    "classifier_output_root",
    "classifier_output_root_sha256",
    "derived_provider_call_count",
    "format",
    "gateway_url",
    "gold_loaded",
    "journal_owner_identity_sha256",
    "journal_owner_format",
    "max_concurrency",
    "model",
    "model_prompt_population_sha256",
    "ordinal_cli_routing_available",
    "preflight_artifact_sha256",
    "production_ordinal_routing_enabled",
    "prompt_population_sha256",
    "provider_calls_during_release",
    "release_identity_sha256",
    "release_status",
    "retained_transformer_token_state_bytes",
    "retry_count",
    "source_artifact_sha256",
    "unsafe_retry_policy",
}
_FORBIDDEN_PROVIDER_KEYS = {
    "answer",
    "answers",
    "gold",
    "gold_answer",
    "ordinal",
    "parent_prediction",
    "protected_parent",
    "reference",
    "reference_answer",
    "semantic_atom_manifest",
    "source_allowlist",
}
_A1_RUNTIME_FIREWALL = {
    "benchmark_fields_loaded": False,
    "ordinal_routing_enabled": False,
    "protected_parent_loaded": False,
    "semantic_atom_manifest_loaded": False,
    "source_allowlist_loaded": False,
    "topic_labels_have_exclusion_authority": False,
}
_DISPOSITION_RUNTIME_FIREWALL = {
    "gold_loaded": False,
    "ordinal_routing_enabled": False,
    "protected_parent_loaded": False,
    "reference_loaded": False,
    "semantic_atom_manifest_loaded": False,
    "source_allowlist_loaded": False,
}


class R7AfterUnionA1ClassifierError(MatchedEvalContractError):
    """The A1 request population, release, journal, or response changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise R7AfterUnionA1ClassifierError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact list")
    return value  # type: ignore[return-value]


def _exact_int(value: object, label: str) -> int:
    _require(type(value) is int, f"{label} must be an exact integer")
    return value  # type: ignore[return-value]


def _canonical_root(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def _read_expected(path: str | Path, expected: str, label: str) -> SealedArtifact:
    artifact = read_sealed_json(path)
    _require(
        artifact.sha256 == require_sha256(expected, label),
        f"{label} artifact changed",
    )
    return artifact


def _without_receipt(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in payload.items() if name != key}


def _forbidden_provider_keys(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in _FORBIDDEN_PROVIDER_KEYS:
                result.add(normalized)
            result.update(_forbidden_provider_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            result.update(_forbidden_provider_keys(child))
    return result


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise R7AfterUnionA1ClassifierError(
                f"classifier JSON repeats object key: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise R7AfterUnionA1ClassifierError(
        f"classifier JSON contains non-finite constant: {value}"
    )


def _strict_json(text: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise R7AfterUnionA1ClassifierError(
            f"{label} must be one strict JSON object"
        ) from exc
    return _exact_dict(value, label)


def _messages(request: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    raw = _exact_list(request.get("messages"), "A1 classifier messages")
    messages: list[dict[str, str]] = []
    for value in raw:
        row = _exact_dict(value, "A1 classifier message")
        _require(
            set(row) == {"role", "content"}
            and row.get("role") in {"system", "user"}
            and type(row.get("content")) is str
            and bool(row.get("content")),
            "A1 classifier message envelope changed",
        )
        messages.append(
            {"role": str(row["role"]), "content": str(row["content"])}
        )
    _require(
        len(messages) == 2
        and messages[0]["role"] == "system"
        and messages[1]["role"] == "user",
        "A1 classifier messages must be the sealed system/user pair",
    )
    return tuple(messages)


def _request_projection(request: Mapping[str, Any]) -> dict[str, Any]:
    exact = _exact_dict(request, "A1 classifier request")
    request_sha = require_sha256(
        exact.get("request_sha256"), "A1 classifier request"
    )
    messages = _messages(exact)
    provider_input = _strict_json(
        messages[1]["content"], "A1 classifier provider input"
    )
    handles = tuple(
        require_text(value, "A1 classifier handle")
        for value in _exact_list(
            exact.get("leaf_handle_ids"), "A1 classifier handles"
        )
    )
    leaves = _exact_list(
        provider_input.get("leaf_population"), "A1 classifier leaf population"
    )
    bindings: list[dict[str, str]] = []
    for raw in leaves:
        leaf = _exact_dict(raw, "A1 classifier prompt leaf")
        _require(
            set(leaf) == _PROMPT_LEAF_KEYS,
            "A1 classifier prompt leaf schema changed",
        )
        bindings.append(
            {
                "handle_id": require_text(
                    leaf.get("handle_id"), "A1 classifier prompt handle"
                ),
                "leaf_receipt_sha256": require_sha256(
                    leaf.get("leaf_receipt_sha256"),
                    "A1 classifier prompt leaf receipt",
                ),
            }
        )
    question_sha = require_sha256(
        exact.get("question_sha256"), "A1 classifier question"
    )
    selected_union_sha = require_sha256(
        exact.get("selected_union_population_sha256"),
        "A1 classifier selected union",
    )
    _require(
        set(exact) == _REQUEST_KEYS
        and request_sha
        == identity_sha256(_without_receipt(exact, "request_sha256"))
        and exact.get("format") == f"{a1.FORMAT}-classifier-request-v1"
        and exact.get("payload_class") == a1.CLASSIFIER_PAYLOAD_CLASS
        and exact.get("classifier_output_token_reserve") == MAX_NEW_TOKENS
        and exact.get("hard_total_token_cap") == a1.MAX_TOTAL_TOKENS
        and exact.get("answer_output_token_reserve")
        == a1.ANSWER_OUTPUT_TOKEN_RESERVE
        and exact.get("topic_labels_have_exclusion_authority") is False
        and type(exact.get("boundary_labels_for_scheduling_only")) is list
        and type(exact.get("topic_labels_for_scheduling_only")) is list
        and len(handles) == len(set(handles)) == len(bindings)
        and tuple(row["handle_id"] for row in bindings) == handles
        and exact.get("shard_population_sha256")
        == identity_sha256(list(handles))
        and set(provider_input) == _CLASSIFIER_PROMPT_KEYS
        and provider_input.get("format")
        == f"{a1.FORMAT}-classifier-prompt-v1"
        and provider_input.get("selected_union_population_sha256")
        == selected_union_sha
        and provider_input.get("topic_labels_have_exclusion_authority")
        is False
        and quote_sha256(
            require_text(
                provider_input.get("dated_question"),
                "A1 classifier dated question",
            )
        )
        == question_sha
        and provider_input.get("response_schema")
        == {
            "leaf_dispositions": [
                {
                    "disposition": (
                        "relevant|definitely_irrelevant|unresolved"
                    ),
                    "handle_id": "one supplied opaque H handle",
                }
            ]
        }
        and exact.get("prompt_token_proxy")
        == count_chat_prompt_token_proxy(messages)
        and _exact_int(
            exact.get("prompt_token_proxy"), "A1 classifier prompt tokens"
        )
        + MAX_NEW_TOKENS
        <= a1.MAX_TOTAL_TOKENS
        and not _forbidden_provider_keys([dict(row) for row in messages])
        and not _forbidden_provider_keys(provider_input),
        "A1 classifier request or provider-message contract changed",
    )
    assert_gold_blind(provider_input, path="r7_a1_classifier_lifecycle.provider")
    body = {
        "classifier_request": dict(exact),
        "classifier_request_sha256": request_sha,
        "format": REQUEST_ROW_FORMAT,
        "leaf_bindings": bindings,
        "messages_sha256": identity_sha256([dict(row) for row in messages]),
        "question_sha256": question_sha,
        "selected_union_population_sha256": selected_union_sha,
    }
    return {**body, "request_row_receipt_sha256": identity_sha256(body)}


def _question_projection(question: Mapping[str, Any]) -> tuple[
    dict[str, Any], tuple[dict[str, Any], ...]
]:
    exact = _exact_dict(question, "A1 classifier question")
    question_sha = require_sha256(
        exact.get("question_sha256"), "A1 classifier question"
    )
    selected_union_sha = require_sha256(
        exact.get("selected_population_sha256"), "A1 selected union"
    )
    raw_requests = _exact_list(
        exact.get("classifier_requests"), "A1 classifier requests"
    )
    request_rows = tuple(_request_projection(row) for row in raw_requests)
    request_shas = tuple(
        str(row["classifier_request_sha256"]) for row in request_rows
    )
    _require(
        bool(request_rows)
        and len(set(request_shas)) == len(request_shas)
        and all(row["question_sha256"] == question_sha for row in request_rows)
        and all(
            row["selected_union_population_sha256"] == selected_union_sha
            for row in request_rows
        )
        and exact.get("classifier_request_count") == len(request_rows)
        and exact.get("classifier_request_population_sha256")
        == identity_sha256(list(request_shas))
        and exact.get("missing_classifier_request_sha256s")
        == list(request_shas),
        "A1 question classifier request population changed",
    )
    selection = _exact_dict(
        exact.get("semantic_selection"), "A1 semantic selection"
    )
    selected_leaves = _exact_list(
        selection.get("leaves"), "A1 selected leaves"
    )
    selected_bindings = [
        {
            "handle_id": require_text(
                _exact_dict(row, "A1 selected leaf").get("handle_id"),
                "A1 selected handle",
            ),
            "leaf_receipt_sha256": require_sha256(
                _exact_dict(row, "A1 selected leaf").get("receipt_sha256"),
                "A1 selected leaf receipt",
            ),
        }
        for row in selected_leaves
    ]
    request_bindings = [
        dict(binding)
        for row in request_rows
        for binding in row["leaf_bindings"]
    ]
    _require(
        request_bindings == selected_bindings
        and exact.get("selected_leaf_count") == len(selected_bindings),
        "A1 classifier shards do not exactly cover selected leaves",
    )
    body = {
        "classifier_request_sha256s": list(request_shas),
        "format": QUESTION_ROW_FORMAT,
        "leaf_bindings": selected_bindings,
        "question_sha256": question_sha,
        "selected_union_population_sha256": selected_union_sha,
    }
    return (
        {**body, "question_row_receipt_sha256": identity_sha256(body)},
        request_rows,
    )


def _load_a1_pair(args: argparse.Namespace) -> tuple[SealedArtifact, SealedArtifact]:
    root = Path(args.a1_root)
    construction = _read_expected(
        root / a1_cli.CONSTRUCTION_NAME,
        str(args.expected_a1_construction_sha256),
        "A1 v2 construction",
    )
    replay = _read_expected(
        root / a1_cli.REPLAY_NAME,
        str(args.expected_a1_replay_sha256),
        "A1 v2 replay",
    )
    payload = construction.payload
    _require(
        construction.sha256 == replay.sha256
        and construction.payload == replay.payload
        and payload.get("format") == a1.FORMAT
        and payload.get("construction_identity_sha256")
        == identity_sha256(
            _without_receipt(payload, "construction_identity_sha256")
        )
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls_performed_by_core") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("runtime_firewall") == _A1_RUNTIME_FIREWALL
        and payload.get("union_before_exclusion") is True
        and payload.get("classifier_payload_class")
        == a1.CLASSIFIER_PAYLOAD_CLASS
        and payload.get("disposition_artifact_sha256") is None
        and payload.get("compiler_output_artifact_sha256") is None
        and payload.get("construction_status")
        == "preflight_external_classification_then_compilation_required"
        and payload.get("compiler_workload_status")
        == "provisional_fail_open_pending_classifier",
        "sealed A1 v2 construction/replay contract changed",
    )
    assert_gold_blind(payload, path="r7_a1_classifier_lifecycle.a1")
    return construction, replay


def build_preflight_payload(
    construction: SealedArtifact,
    replay: SealedArtifact,
    *,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> tuple[
    dict[str, Any],
    tuple[tuple[dict[str, str], ...], ...],
]:
    payload = construction.payload
    _require(
        construction.sha256 == replay.sha256
        and construction.payload == replay.payload
        and model == DEFAULT_MODEL
        and gateway_url == DEFAULT_GATEWAY_URL
        and type(max_concurrency) is int
        and max_concurrency > 0,
        "A1 classifier preflight runtime policy changed",
    )
    raw_questions = _exact_list(payload.get("questions"), "A1 questions")
    question_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    for raw in raw_questions:
        question_row, requests = _question_projection(
            _exact_dict(raw, "A1 question")
        )
        question_rows.append(question_row)
        request_rows.extend(requests)
    request_shas = [str(row["classifier_request_sha256"]) for row in request_rows]
    prompts = tuple(
        _messages(row["classifier_request"]) for row in request_rows
    )
    prompt_population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_PROMPT_TOKENS
    )
    source_sha = require_sha256(
        payload.get("source_artifact_sha256"), "A1 source artifact"
    )
    source_replay_sha = require_sha256(
        payload.get("source_replay_artifact_sha256"), "A1 source replay"
    )
    derived_calls = len(request_rows)
    _require(
        bool(request_rows)
        and len(set(request_shas)) == derived_calls
        and prompt_population.logical_prompt_count
        == prompt_population.unique_prompt_count
        == derived_calls
        and payload.get("question_count") == len(question_rows)
        and payload.get("expected_question_count") == len(question_rows)
        and payload.get("classifier_request_count") == derived_calls
        and payload.get("missing_classifier_call_count") == derived_calls
        and payload.get("missing_classifier_request_sha256s") == request_shas
        and payload.get("classifier_request_population_sha256")
        == identity_sha256(sorted(request_shas))
        and payload.get("selected_leaf_count")
        == sum(len(row["leaf_bindings"]) for row in question_rows)
        and len({row["question_sha256"] for row in question_rows})
        == len(question_rows)
        and all(
            row["messages_sha256"]
            == prompt_population.ordered_rows[index].messages_sha256
            for index, row in enumerate(request_rows)
        ),
        "A1 exact classifier request population changed",
    )
    model_prompt_sha = identity_sha256(
        {
            "classifier_request_population_sha256": payload[
                "classifier_request_population_sha256"
            ],
            "format": MODEL_PROMPT_POPULATION_FORMAT,
            "model": model,
            "ordered_classifier_request_sha256s": request_shas,
            "prompt_population_sha256": prompt_population.prompt_population_sha256,
        }
    )
    body = {
        "a1_classifier_request_population_sha256": payload[
            "classifier_request_population_sha256"
        ],
        "a1_construction_artifact_sha256": construction.sha256,
        "a1_construction_identity_sha256": payload[
            "construction_identity_sha256"
        ],
        "a1_replay_artifact_sha256": replay.sha256,
        "classifier_id": CLASSIFIER_ID,
        "classifier_payload_class": a1.CLASSIFIER_PAYLOAD_CLASS,
        "derived_provider_call_count": derived_calls,
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "max_concurrency": max_concurrency,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "model": model,
        "model_prompt_population_sha256": model_prompt_sha,
        "ordered_classifier_request_population_sha256": identity_sha256(
            request_shas
        ),
        "ordinal_cli_routing_available": False,
        "physical_provider_calls": 0,
        "production_ordinal_routing_enabled": False,
        "prompt_population_sha256": prompt_population.prompt_population_sha256,
        "question_count": len(question_rows),
        "question_population_sha256": identity_sha256(
            [row["question_row_receipt_sha256"] for row in question_rows]
        ),
        "question_rows": question_rows,
        "request_rows": request_rows,
        "retained_transformer_token_state_bytes": 0,
        "runtime_firewall": dict(_DISPOSITION_RUNTIME_FIREWALL),
        "selected_leaf_count": payload["selected_leaf_count"],
        "source_artifact_sha256": source_sha,
        "source_replay_artifact_sha256": source_replay_sha,
    }
    result = {**body, "preflight_identity_sha256": identity_sha256(body)}
    assert_gold_blind(result, path="r7_a1_classifier_lifecycle.preflight")
    return result, prompts


def _validate_preflight(
    artifact: SealedArtifact,
) -> tuple[
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    payload = artifact.payload
    body = _without_receipt(payload, "preflight_identity_sha256")
    raw_requests = _exact_list(
        payload.get("request_rows"), "classifier preflight requests"
    )
    raw_questions = _exact_list(
        payload.get("question_rows"), "classifier preflight questions"
    )
    requests: list[dict[str, Any]] = []
    for raw in raw_requests:
        row = _exact_dict(raw, "classifier preflight request")
        declared = require_sha256(
            row.get("request_row_receipt_sha256"),
            "classifier preflight request row",
        )
        expected = _request_projection(
            _exact_dict(row.get("classifier_request"), "classifier request")
        )
        _require(
            row == expected
            and declared
            == identity_sha256(
                _without_receipt(row, "request_row_receipt_sha256")
            ),
            "classifier preflight request row changed",
        )
        requests.append(row)
    request_by_sha = {
        str(row["classifier_request_sha256"]): row for row in requests
    }
    _require(
        len(request_by_sha) == len(requests),
        "classifier preflight request rows repeat",
    )
    questions: list[dict[str, Any]] = []
    for raw in raw_questions:
        row = _exact_dict(raw, "classifier preflight question")
        request_shas = [
            require_sha256(value, "classifier question request")
            for value in _exact_list(
                row.get("classifier_request_sha256s"),
                "classifier question request population",
            )
        ]
        bindings: list[dict[str, str]] = []
        for raw_binding in _exact_list(
            row.get("leaf_bindings"), "classifier question leaf bindings"
        ):
            binding = _exact_dict(
                raw_binding, "classifier question leaf binding"
            )
            _require(
                set(binding) == {"handle_id", "leaf_receipt_sha256"},
                "classifier question leaf binding changed",
            )
            bindings.append(
                {
                    "handle_id": require_text(
                        binding.get("handle_id"), "classifier question handle"
                    ),
                    "leaf_receipt_sha256": require_sha256(
                        binding.get("leaf_receipt_sha256"),
                        "classifier question leaf receipt",
                    ),
                }
            )
        _require(
            set(row) == _QUESTION_ROW_KEYS
            and row.get("format") == QUESTION_ROW_FORMAT
            and len(request_shas) == len(set(request_shas))
            and bool(request_shas)
            and len(bindings)
            == len({binding["handle_id"] for binding in bindings})
            and bool(bindings)
            and require_sha256(
                row.get("question_sha256"), "classifier question"
            )
            and require_sha256(
                row.get("selected_union_population_sha256"),
                "classifier question selected union",
            )
            and all(request_sha in request_by_sha for request_sha in request_shas)
            and all(
                request_by_sha[request_sha]["question_sha256"]
                == row.get("question_sha256")
                and request_by_sha[request_sha][
                    "selected_union_population_sha256"
                ]
                == row.get("selected_union_population_sha256")
                for request_sha in request_shas
            )
            and [
                dict(binding)
                for request_sha in request_shas
                for binding in request_by_sha[request_sha]["leaf_bindings"]
            ]
            == bindings
            and row.get("question_row_receipt_sha256")
            == identity_sha256(
                _without_receipt(row, "question_row_receipt_sha256")
            ),
            "classifier preflight question row changed",
        )
        questions.append(row)
    prompts = tuple(
        _messages(row["classifier_request"]) for row in requests
    )
    prompt_population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_PROMPT_TOKENS
    )
    request_shas = [str(row["classifier_request_sha256"]) for row in requests]
    question_request_shas = [
        sha
        for row in questions
        for sha in _exact_list(
            row.get("classifier_request_sha256s"),
            "classifier question request population",
        )
    ]
    question_bindings = [
        dict(binding)
        for row in questions
        for binding in _exact_list(
            row.get("leaf_bindings"), "classifier question leaf bindings"
        )
    ]
    request_bindings = [
        dict(binding)
        for row in requests
        for binding in _exact_list(
            row.get("leaf_bindings"), "classifier request leaf bindings"
        )
    ]
    derived_calls = len(requests)
    model_prompt_sha = identity_sha256(
        {
            "classifier_request_population_sha256": payload.get(
                "a1_classifier_request_population_sha256"
            ),
            "format": MODEL_PROMPT_POPULATION_FORMAT,
            "model": payload.get("model"),
            "ordered_classifier_request_sha256s": request_shas,
            "prompt_population_sha256": prompt_population.prompt_population_sha256,
        }
    )
    _require(
        set(payload) == _PREFLIGHT_KEYS
        and payload.get("preflight_identity_sha256") == identity_sha256(body)
        and payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("runtime_firewall")
        == _DISPOSITION_RUNTIME_FIREWALL
        and payload.get("ordinal_cli_routing_available") is False
        and payload.get("production_ordinal_routing_enabled") is False
        and payload.get("classifier_id") == CLASSIFIER_ID
        and payload.get("classifier_payload_class")
        == a1.CLASSIFIER_PAYLOAD_CLASS
        and payload.get("model") == DEFAULT_MODEL
        and payload.get("gateway_url") == DEFAULT_GATEWAY_URL
        and type(payload.get("max_concurrency")) is int
        and payload.get("max_concurrency") > 0
        and payload.get("max_new_tokens") == MAX_NEW_TOKENS
        and payload.get("max_prompt_tokens") == MAX_PROMPT_TOKENS
        and payload.get("derived_provider_call_count") == derived_calls
        and derived_calls > 0
        and prompt_population.logical_prompt_count
        == prompt_population.unique_prompt_count
        == derived_calls
        and len(set(request_shas)) == derived_calls
        and payload.get("ordered_classifier_request_population_sha256")
        == identity_sha256(request_shas)
        and payload.get("a1_classifier_request_population_sha256")
        == identity_sha256(sorted(request_shas))
        and payload.get("prompt_population_sha256")
        == prompt_population.prompt_population_sha256
        and payload.get("model_prompt_population_sha256") == model_prompt_sha
        and question_request_shas == request_shas
        and question_bindings == request_bindings
        and payload.get("question_count") == len(questions)
        and payload.get("question_population_sha256")
        == identity_sha256(
            [row["question_row_receipt_sha256"] for row in questions]
        )
        and payload.get("selected_leaf_count") == len(question_bindings)
        and len({row.get("question_sha256") for row in questions})
        == len(questions)
        and all(
            require_sha256(
                payload.get(key), f"classifier preflight {key}"
            )
            for key in (
                "a1_construction_artifact_sha256",
                "a1_construction_identity_sha256",
                "a1_replay_artifact_sha256",
                "source_artifact_sha256",
                "source_replay_artifact_sha256",
            )
        ),
        "sealed A1 classifier preflight changed",
    )
    assert_gold_blind(payload, path="r7_a1_classifier_lifecycle.preflight")
    return prompts, tuple(requests), tuple(questions)


def _read_preflight(
    output_root: str | Path, expected_sha256: str
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    artifact = _read_expected(
        Path(output_root) / PREFLIGHT_NAME,
        expected_sha256,
        "A1 classifier preflight",
    )
    prompts, requests, questions = _validate_preflight(artifact)
    return artifact, prompts, requests, questions


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    _require(
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "A1 classifier preflight requires a fresh absent checkpoint root",
    )
    construction, replay = _load_a1_pair(args)
    payload, _ = build_preflight_payload(
        construction,
        replay,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(output_root / PREFLIGHT_NAME, payload)
    return {
        "a1_construction_sha256": construction.sha256,
        "created": created,
        "derived_provider_call_count": payload["derived_provider_call_count"],
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "retained_transformer_token_state_bytes": 0,
    }


def _journal_owner_body(
    preflight: SealedArtifact, *, output_root: str | Path
) -> dict[str, Any]:
    root = _canonical_root(output_root)
    checkpoint_root = _canonical_root(Path(output_root) / CHECKPOINT_DIR_NAME)
    return {
        "checkpoint_root": checkpoint_root,
        "checkpoint_root_sha256": identity_sha256(
            {"canonical_root": checkpoint_root}
        ),
        "classifier_output_root": root,
        "classifier_output_root_sha256": identity_sha256(
            {"canonical_root": root}
        ),
        "derived_provider_call_count": preflight.payload[
            "derived_provider_call_count"
        ],
        "format": JOURNAL_OWNER_FORMAT,
        "model": preflight.payload["model"],
        "model_prompt_population_sha256": preflight.payload[
            "model_prompt_population_sha256"
        ],
        "preflight_artifact_sha256": preflight.sha256,
        "prompt_population_sha256": preflight.payload[
            "prompt_population_sha256"
        ],
    }


def _release_payload(
    preflight: SealedArtifact, *, output_root: str | Path
) -> dict[str, Any]:
    owner = _journal_owner_body(preflight, output_root=output_root)
    body = {
        "a1_construction_artifact_sha256": preflight.payload[
            "a1_construction_artifact_sha256"
        ],
        "approval_opt_in": True,
        "checkpoint_root": owner["checkpoint_root"],
        "checkpoint_root_sha256": owner["checkpoint_root_sha256"],
        "classifier_output_root": owner["classifier_output_root"],
        "classifier_output_root_sha256": owner[
            "classifier_output_root_sha256"
        ],
        "derived_provider_call_count": preflight.payload[
            "derived_provider_call_count"
        ],
        "format": RELEASE_FORMAT,
        "gateway_url": preflight.payload["gateway_url"],
        "gold_loaded": False,
        "journal_owner_identity_sha256": identity_sha256(owner),
        "journal_owner_format": JOURNAL_OWNER_FORMAT,
        "max_concurrency": preflight.payload["max_concurrency"],
        "model": preflight.payload["model"],
        "model_prompt_population_sha256": preflight.payload[
            "model_prompt_population_sha256"
        ],
        "ordinal_cli_routing_available": False,
        "preflight_artifact_sha256": preflight.sha256,
        "production_ordinal_routing_enabled": False,
        "prompt_population_sha256": preflight.payload[
            "prompt_population_sha256"
        ],
        "provider_calls_during_release": 0,
        "release_status": "approved_for_provider_execution",
        "retained_transformer_token_state_bytes": 0,
        "retry_count": 0,
        "source_artifact_sha256": preflight.payload[
            "source_artifact_sha256"
        ],
        "unsafe_retry_policy": "refuse_incomplete_request_response_pair",
    }
    return {**body, "release_identity_sha256": identity_sha256(body)}


def _validate_release(
    artifact: SealedArtifact,
    *,
    preflight: SealedArtifact,
    output_root: str | Path,
) -> None:
    payload = artifact.payload
    owner = _journal_owner_body(preflight, output_root=output_root)
    body = _without_receipt(payload, "release_identity_sha256")
    _require(
        set(payload) == _RELEASE_KEYS
        and payload.get("release_identity_sha256") == identity_sha256(body)
        and payload.get("format") == RELEASE_FORMAT
        and payload.get("release_status")
        == "approved_for_provider_execution"
        and payload.get("approval_opt_in") is True
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls_during_release") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("retry_count") == 0
        and payload.get("unsafe_retry_policy")
        == "refuse_incomplete_request_response_pair"
        and payload.get("ordinal_cli_routing_available") is False
        and payload.get("production_ordinal_routing_enabled") is False
        and payload.get("preflight_artifact_sha256") == preflight.sha256
        and payload.get("derived_provider_call_count")
        == preflight.payload.get("derived_provider_call_count")
        and payload.get("model") == preflight.payload.get("model")
        and payload.get("gateway_url") == preflight.payload.get("gateway_url")
        and payload.get("max_concurrency")
        == preflight.payload.get("max_concurrency")
        and payload.get("model_prompt_population_sha256")
        == preflight.payload.get("model_prompt_population_sha256")
        and payload.get("prompt_population_sha256")
        == preflight.payload.get("prompt_population_sha256")
        and payload.get("a1_construction_artifact_sha256")
        == preflight.payload.get("a1_construction_artifact_sha256")
        and payload.get("source_artifact_sha256")
        == preflight.payload.get("source_artifact_sha256")
        and payload.get("journal_owner_format") == JOURNAL_OWNER_FORMAT
        and all(
            payload.get(key) == value
            for key, value in owner.items()
            if key != "format"
        )
        and payload.get("journal_owner_identity_sha256")
        == identity_sha256(owner),
        "A1 classifier provider release changed",
    )
    assert_gold_blind(payload, path="r7_a1_classifier_lifecycle.release")


def _read_release(
    output_root: str | Path,
    expected_sha256: str,
    *,
    preflight: SealedArtifact,
) -> SealedArtifact:
    artifact = _read_expected(
        Path(output_root) / RELEASE_NAME,
        expected_sha256,
        "A1 classifier release",
    )
    _validate_release(artifact, preflight=preflight, output_root=output_root)
    return artifact


def run_approve_release(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    _require(
        args.approve_provider_release is True,
        "A1 classifier release requires explicit provider approval",
    )
    _require(
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "A1 classifier release requires an absent checkpoint root",
    )
    preflight, _, _, _ = _read_preflight(
        output_root, str(args.expected_preflight_sha256)
    )
    payload = _release_payload(preflight, output_root=output_root)
    artifact, created = publish_sealed_json(output_root / RELEASE_NAME, payload)
    return {
        "created": created,
        "derived_provider_call_count": payload["derived_provider_call_count"],
        "journal_owner_identity_sha256": payload[
            "journal_owner_identity_sha256"
        ],
        "physical_provider_calls": 0,
        "preflight_sha256": preflight.sha256,
        "release_sha256": artifact.sha256,
        "retained_transformer_token_state_bytes": 0,
    }


def _runtime(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionRuntime:
    call_count = _exact_int(
        preflight.payload.get("derived_provider_call_count"),
        "A1 classifier call count",
    )
    _require(
        str(args.model) == preflight.payload.get("model") == DEFAULT_MODEL
        and str(args.gateway_url)
        == preflight.payload.get("gateway_url")
        == DEFAULT_GATEWAY_URL
        and int(args.max_concurrency)
        == preflight.payload.get("max_concurrency")
        and release.payload.get("preflight_artifact_sha256") == preflight.sha256
        and release.payload.get("release_status")
        == "approved_for_provider_execution"
        and len(prompts) == call_count,
        "A1 classifier runtime differs from sealed release",
    )
    return FastCompletionRuntime(
        checkpoint_dir=Path(args.output_root) / CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=DEFAULT_MODEL,
        client=client,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_new_tokens=MAX_NEW_TOKENS,
        max_concurrency=int(args.max_concurrency),
        retries=0,
        benchmark_provenance={
            "a1_construction_artifact_sha256": preflight.payload[
                "a1_construction_artifact_sha256"
            ],
            "arm": FORMAT,
            "authorized_unique_calls": call_count,
            "classifier_request_population_sha256": preflight.payload[
                "a1_classifier_request_population_sha256"
            ],
            "experiment_format": a1.DISPOSITIONS_FORMAT,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "gold_loaded": False,
            "journal_owner_identity_sha256": release.payload[
                "journal_owner_identity_sha256"
            ],
            "model_prompt_population_sha256": preflight.payload[
                "model_prompt_population_sha256"
            ],
            "preflight_artifact_sha256": preflight.sha256,
            "release_authorization_artifact_sha256": release.sha256,
            "source_artifact_sha256": preflight.payload[
                "source_artifact_sha256"
            ],
        },
    )


def _checkpoint_batch(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionBatch:
    runtime = _runtime(preflight, release, prompts, args=args, client=client)
    try:
        return runtime.run()
    finally:
        runtime.close()


def _read_only_checkpoint_count(output_root: str | Path, call_count: int) -> int:
    root = Path(output_root) / CHECKPOINT_DIR_NAME
    if not root.exists():
        return 0
    _require(
        not root.is_symlink() and root.is_dir(),
        "A1 classifier checkpoint root must be a regular directory",
    )
    requests: set[str] = set()
    responses: set[str] = set()
    for path in root.iterdir():
        _require(
            not path.is_symlink() and path.is_file(),
            "A1 classifier checkpoint root contains foreign state",
        )
        if path.name == ".fast-completion-journal.lock":
            continue
        match = _JOURNAL_FILENAME_RE.fullmatch(path.name)
        _require(
            match is not None,
            "A1 classifier checkpoint root contains foreign journal state",
        )
        assert match is not None
        target = requests if match.group("kind") == "request" else responses
        target.add(match.group("key"))
    _require(
        requests == responses,
        "A1 classifier checkpoint pair is incomplete; unsafe retry forbidden",
    )
    _require(
        len(requests) <= call_count,
        "A1 classifier checkpoint population exceeds sealed calls",
    )
    return len(requests)


def _validated_checkpoint_hits(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
) -> int:
    root = Path(args.output_root) / CHECKPOINT_DIR_NAME
    if not root.exists():
        return 0
    runtime = _runtime(preflight, release, prompts, args=args, client=None)
    try:
        with runtime._journal_guard():  # noqa: SLF001 - runtime owns journals
            records = runtime._load_all_records()  # noqa: SLF001
    finally:
        runtime.close()
    call_count = int(preflight.payload["derived_provider_call_count"])
    _require(
        len(records) <= call_count,
        "A1 classifier checkpoints escaped the sealed prompt population",
    )
    return len(records)


def _make_provider_client(api_key: str, gateway_url: str) -> Any:
    return live._make_provider_client(api_key, gateway_url)  # noqa: SLF001


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, _, _ = _read_preflight(
        args.output_root, str(args.expected_preflight_sha256)
    )
    release = _read_release(
        args.output_root,
        str(args.expected_release_sha256),
        preflight=preflight,
    )
    call_count = int(preflight.payload["derived_provider_call_count"])
    _require(
        args.enable_provider is True
        and type(args.authorized_provider_calls) is int
        and 0 <= args.authorized_provider_calls <= call_count,
        "A1 classifier provider requires bounded Terra authorization",
    )
    candidate_hits = _read_only_checkpoint_count(args.output_root, call_count)
    remaining = call_count - candidate_hits
    _require(
        args.authorized_provider_calls == remaining,
        "A1 classifier authorization must exactly equal remaining calls",
    )
    checkpoint_hits = _validated_checkpoint_hits(
        preflight, release, prompts, args=args
    )
    _require(
        checkpoint_hits == candidate_hits,
        "A1 classifier checkpoint count changed after authorization",
    )
    if remaining == 0:
        batch = _checkpoint_batch(
            preflight, release, prompts, args=args, client=None
        )
        _require(
            batch.usage.logical_calls
            == batch.usage.unique_calls
            == batch.usage.checkpoint_hits
            == call_count
            and batch.usage.physical_calls == 0,
            "A1 classifier completed checkpoint replay changed",
        )
    else:
        load_dotenv()
        api_key = os.environ.get(str(args.api_key_env), "").strip()
        _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
        client = _make_provider_client(api_key, str(args.gateway_url))
        try:
            batch = _checkpoint_batch(
                preflight, release, prompts, args=args, client=client
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        _require(
            batch.usage.logical_calls
            == batch.usage.unique_calls
            == call_count
            and batch.usage.physical_calls + batch.usage.checkpoint_hits
            == call_count
            and batch.usage.physical_calls <= args.authorized_provider_calls
            and batch.usage.checkpoint_hits >= checkpoint_hits,
            "A1 classifier provider population changed",
        )
    return {
        "authorized_remaining_provider_calls": remaining,
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "derived_provider_call_count": call_count,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": preflight.sha256,
        "release_sha256": release.sha256,
        "retained_transformer_token_state_bytes": 0,
    }


def _parse_dispositions(
    completion: str,
    expected_bindings: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    payload = _strict_json(completion, "A1 classifier response")
    _require(
        set(payload) == {"leaf_dispositions"},
        "A1 classifier response envelope changed",
    )
    raw_rows = _exact_list(
        payload.get("leaf_dispositions"), "A1 classifier dispositions"
    )
    expected = tuple(str(row["handle_id"]) for row in expected_bindings)
    rows: list[dict[str, str]] = []
    for raw in raw_rows:
        row = _exact_dict(raw, "A1 classifier disposition")
        _require(
            set(row) == {"disposition", "handle_id"}
            and row.get("disposition")
            in {"relevant", "definitely_irrelevant", "unresolved"},
            "A1 classifier disposition must be exact R/I/U",
        )
        rows.append(
            {
                "disposition": str(row["disposition"]),
                "handle_id": require_text(
                    row.get("handle_id"), "A1 classifier disposition handle"
                ),
            }
        )
    _require(
        tuple(row["handle_id"] for row in rows) == expected,
        "A1 classifier response must cover every handle once in supplied order",
    )
    return [
        {
            **row,
            "leaf_receipt_sha256": str(binding["leaf_receipt_sha256"]),
        }
        for row, binding in zip(rows, expected_bindings, strict=True)
    ]


def _record_by_messages(
    batch: FastCompletionBatch,
) -> dict[str, FastCompletionRecord]:
    result = {row.messages_sha256: row for row in batch.unique_records}
    _require(
        len(result) == len(batch.unique_records),
        "A1 classifier completion records repeat",
    )
    return result


def _dispositions_payload(
    preflight: SealedArtifact,
    release: SealedArtifact,
    request_rows: Sequence[Mapping[str, Any]],
    question_rows: Sequence[Mapping[str, Any]],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    call_count = int(preflight.payload["derived_provider_call_count"])
    _require(
        len(request_rows)
        == len(batch.logical_completions)
        == len(batch.unique_records)
        == call_count
        and batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.checkpoint_hits
        == call_count
        and batch.usage.physical_calls == 0
        and batch.provenance.model == preflight.payload["model"]
        and batch.provenance.retries == 0
        and batch.provenance.prompt_population_sha256
        == preflight.payload["prompt_population_sha256"]
        and batch.provenance.benchmark_provenance.get(
            "model_prompt_population_sha256"
        )
        == preflight.payload["model_prompt_population_sha256"]
        and batch.provenance.benchmark_provenance.get(
            "journal_owner_identity_sha256"
        )
        == release.payload["journal_owner_identity_sha256"],
        "A1 classifier checkpoint-only completion batch changed",
    )
    record_by_messages = _record_by_messages(batch)
    responses: list[dict[str, Any]] = []
    disposition_by_request: dict[str, list[dict[str, str]]] = {}
    for request_row, completion in zip(
        request_rows, batch.logical_completions, strict=True
    ):
        messages_sha = str(request_row["messages_sha256"])
        record = record_by_messages.get(messages_sha)
        _require(
            record is not None
            and record.completion == completion
            and record.completion_sha256 == quote_sha256(completion)
            and record.requested_model == preflight.payload["model"]
            and record.finish_reason == "stop",
            "A1 classifier completion record binding changed",
        )
        bindings = tuple(
            _exact_dict(row, "A1 classifier response leaf binding")
            for row in _exact_list(
                request_row.get("leaf_bindings"),
                "A1 classifier response leaf bindings",
            )
        )
        dispositions = _parse_dispositions(completion, bindings)
        request_sha = str(request_row["classifier_request_sha256"])
        disposition_by_request[request_sha] = dispositions
        body = {
            "call_key_sha256": record.call_key_sha256,
            "classifier_output": completion,
            "classifier_output_sha256": record.completion_sha256,
            "dispositions": dispositions,
            "format": RESPONSE_ROW_FORMAT,
            "leaf_bindings": [dict(row) for row in bindings],
            "messages_sha256": messages_sha,
            "question_sha256": request_row["question_sha256"],
            "request_journal_sha256": record.request_journal_sha256,
            "request_sha256": request_sha,
            "response_journal_sha256": record.response_journal_sha256,
            "selected_union_population_sha256": request_row[
                "selected_union_population_sha256"
            ],
            "source_artifact_sha256": preflight.payload[
                "source_artifact_sha256"
            ],
        }
        responses.append(
            {**body, "response_row_receipt_sha256": identity_sha256(body)}
        )
    questions: list[dict[str, Any]] = []
    for question in question_rows:
        request_shas = [
            str(value)
            for value in _exact_list(
                question.get("classifier_request_sha256s"),
                "A1 classifier materialization request population",
            )
        ]
        dispositions = [
            row for request_sha in request_shas for row in disposition_by_request[request_sha]
        ]
        expected_bindings = _exact_list(
            question.get("leaf_bindings"),
            "A1 classifier materialization leaf population",
        )
        _require(
            [
                {
                    "handle_id": row["handle_id"],
                    "leaf_receipt_sha256": row["leaf_receipt_sha256"],
                }
                for row in dispositions
            ]
            == expected_bindings,
            "A1 classifier materialization lost selected leaf coverage",
        )
        questions.append(
            {
                "classifier_request_sha256s": request_shas,
                "dispositions": dispositions,
                "question_sha256": question["question_sha256"],
                "selected_union_population_sha256": question[
                    "selected_union_population_sha256"
                ],
            }
        )
    response_receipts = [row["response_row_receipt_sha256"] for row in responses]
    value = {
        "a1_construction_artifact_sha256": preflight.payload[
            "a1_construction_artifact_sha256"
        ],
        "a1_replay_artifact_sha256": preflight.payload[
            "a1_replay_artifact_sha256"
        ],
        "classifier_id": CLASSIFIER_ID,
        "classifier_request_population_sha256": preflight.payload[
            "a1_classifier_request_population_sha256"
        ],
        "completion_runtime_identity_sha256": batch.runtime_identity_sha256,
        "derived_provider_call_count": call_count,
        "disposition_population_sha256": identity_sha256(questions),
        "format": a1.DISPOSITIONS_FORMAT,
        "journal_owner_identity_sha256": release.payload[
            "journal_owner_identity_sha256"
        ],
        "lifecycle_format": FORMAT,
        "model": preflight.payload["model"],
        "model_prompt_population_sha256": preflight.payload[
            "model_prompt_population_sha256"
        ],
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "prompt_population_sha256": preflight.payload[
            "prompt_population_sha256"
        ],
        "provider_calls_performed_by_core": 0,
        "question_count": len(questions),
        "questions": questions,
        "release_authorization_artifact_sha256": release.sha256,
        "responses": responses,
        "response_population_sha256": identity_sha256(response_receipts),
        "retained_transformer_token_state_bytes": 0,
        "runtime_firewall": dict(_DISPOSITION_RUNTIME_FIREWALL),
        "source_artifact_sha256": preflight.payload[
            "source_artifact_sha256"
        ],
        "source_replay_artifact_sha256": preflight.payload[
            "source_replay_artifact_sha256"
        ],
    }
    assert_gold_blind(value, path="r7_a1_classifier_lifecycle.dispositions")
    return value


def _complete_checkpoint_batch(
    preflight: SealedArtifact,
    release: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
) -> FastCompletionBatch:
    call_count = int(preflight.payload["derived_provider_call_count"])
    _require(
        _read_only_checkpoint_count(args.output_root, call_count) == call_count,
        "A1 classifier materialization requires every complete checkpoint",
    )
    return _checkpoint_batch(
        preflight, release, prompts, args=args, client=None
    )


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, requests, questions = _read_preflight(
        args.output_root, str(args.expected_preflight_sha256)
    )
    release = _read_release(
        args.output_root,
        str(args.expected_release_sha256),
        preflight=preflight,
    )
    batch = _complete_checkpoint_batch(
        preflight, release, prompts, args=args
    )
    payload = _dispositions_payload(
        preflight, release, requests, questions, batch
    )
    artifact, created = publish_sealed_json(
        Path(args.output_root) / DISPOSITIONS_NAME, payload
    )
    return {
        "checkpoint_hits": payload["derived_provider_call_count"],
        "created": created,
        "derived_provider_call_count": payload[
            "derived_provider_call_count"
        ],
        "dispositions_sha256": artifact.sha256,
        "physical_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, requests, questions = _read_preflight(
        args.output_root, str(args.expected_preflight_sha256)
    )
    release = _read_release(
        args.output_root,
        str(args.expected_release_sha256),
        preflight=preflight,
    )
    batch = _complete_checkpoint_batch(
        preflight, release, prompts, args=args
    )
    rebuilt = _dispositions_payload(
        preflight, release, requests, questions, batch
    )
    root = Path(args.output_root)
    artifact = _read_expected(
        root / DISPOSITIONS_NAME,
        str(args.expected_dispositions_sha256),
        "A1 classifier dispositions",
    )
    _require(
        artifact.payload == rebuilt,
        "A1 classifier dispositions differ from checkpoint-only replay",
    )
    replay, _ = publish_sealed_json(root / REPLAY_NAME, rebuilt)
    _require(
        replay.sha256 == artifact.sha256,
        "A1 classifier replay is not byte-identical",
    )
    return {
        "byte_identical": True,
        "dispositions_sha256": artifact.sha256,
        "physical_provider_calls": 0,
        "replay_sha256": replay.sha256,
        "retained_transformer_token_state_bytes": 0,
    }


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument(
        "--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY
    )


def _add_sealed_lifecycle_args(parser: argparse.ArgumentParser) -> None:
    _add_runtime_args(parser)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--expected-release-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    _add_runtime_args(preflight)
    preflight.add_argument("--a1-root", type=Path, default=DEFAULT_A1_ROOT)
    preflight.add_argument(
        "--expected-a1-construction-sha256", default=EXPECTED_A1_SHA256
    )
    preflight.add_argument(
        "--expected-a1-replay-sha256", default=EXPECTED_A1_SHA256
    )

    release = commands.add_parser("approve-release")
    _add_runtime_args(release)
    release.add_argument("--expected-preflight-sha256", required=True)
    release.add_argument("--approve-provider-release", action="store_true")

    provider = commands.add_parser("provider-run")
    _add_sealed_lifecycle_args(provider)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, required=True)
    provider.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)

    materialize = commands.add_parser("materialize")
    _add_sealed_lifecycle_args(materialize)

    replay = commands.add_parser("replay")
    _add_sealed_lifecycle_args(replay)
    replay.add_argument("--expected-dispositions-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = run_preflight(args)
    elif args.command == "approve-release":
        result = run_approve_release(args)
    elif args.command == "provider-run":
        result = run_provider(args)
    elif args.command == "materialize":
        result = run_materialize(args)
    else:
        result = run_replay(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_DIR_NAME",
    "CLASSIFIER_ID",
    "DEFAULT_A1_ROOT",
    "DEFAULT_GATEWAY_URL",
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT_ROOT",
    "DISPOSITIONS_NAME",
    "EXPECTED_A1_SHA256",
    "PREFLIGHT_NAME",
    "RELEASE_NAME",
    "REPLAY_NAME",
    "R7AfterUnionA1ClassifierError",
    "build_parser",
    "build_preflight_payload",
    "main",
    "run_approve_release",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
]
