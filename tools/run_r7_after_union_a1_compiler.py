#!/usr/bin/env python3
"""Run the sealed R7 A1b Terra typed-fact compiler lifecycle.

The lifecycle consumes the exact classified A1 construction/replay pair and
derives its actionable ``typed_fact_compiler_strict_json_v1`` population.  It
sends only the already-sealed compiler messages to Terra.  Provider execution
requires a separate release, uses immutable zero-retry journals, and is the
only phase allowed to open a provider client.

Materialization is checkpoint-only.  It rejects malformed facts and every
missing, foreign, or non-exact citation before publishing the
``COMPILER_OUTPUTS_FORMAT`` artifact accepted by the A1 adapter.  Leaves not
supported by a validated retained fact are recorded as unresolved; no parser
failure or citation omission silently resolves a leaf.
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
from tools.matched_eval import r7_after_union_a1 as a1  # noqa: E402
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
from tools.matched_eval.typed_fact_compiler import (  # noqa: E402
    COMPILER_OUTPUT_TOKEN_RESERVE,
    COMPILER_PROMPT_FORMAT,
    HARD_PROMPT_TOKEN_CAP,
    MAX_COMPILER_FACTS,
    TypedFactCompilation,
    build_compiler_messages,
    parse_compiler_completion,
)


FORMAT = "memory-condense-r7-after-union-a1b-compiler-lifecycle-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight-v1"
RELEASE_FORMAT = f"{FORMAT}-provider-release-v1"
REQUEST_ROW_FORMAT = f"{FORMAT}-request-row-v1"
QUESTION_ROW_FORMAT = f"{FORMAT}-question-row-v1"
RESPONSE_BINDING_FORMAT = f"{FORMAT}-response-binding-v1"
JOURNAL_OWNER_FORMAT = f"{FORMAT}-journal-owner-v1"
MODEL_PROMPT_POPULATION_FORMAT = f"{FORMAT}-model-prompt-population-v1"
COMPILER_ID = "r7-a1b-terra-typed-fact-compiler-strict-json-v1"
COMPILER_PAYLOAD_CLASS = a1.COMPILER_PAYLOAD_CLASS

PREFLIGHT_NAME = "r7-after-union-a1b-compiler-preflight-v1.json"
RELEASE_NAME = "r7-after-union-a1b-compiler-provider-release-v1.json"
OUTPUTS_NAME = "r7-after-union-a1b-compiler-outputs-v1.json"
REPLAY_NAME = "r7-after-union-a1b-compiler-outputs-replay-v1.json"
CHECKPOINT_DIR_NAME = "terra-r7-after-union-a1b-compiler-v1-calls"

DEFAULT_CLASSIFIED_ROOT = Path(
    "eval_results/matched_eval_100/"
    "locked-r7-after-union-a1-classified-temporal-effective-v1"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_CLASSIFIED_ROOT / "terra-compiler-v1"
EXPECTED_CLASSIFIED_SHA256 = (
    "d9071196d57fedf96516aae38dfe5ed0adb5218858bee32d7f7904353c9c4da1"
)
EXPECTED_DISPOSITION_ARTIFACT_SHA256 = (
    "40a584d6499f3682a89cab1aa272c34a8ccf7ead825d2451192bc2b49114a278"
)
DEFAULT_MODEL = live.DEFAULT_TERRA_GATEWAY_MODEL
DEFAULT_GATEWAY_URL = live.DEFAULT_GATEWAY_URL
DEFAULT_API_KEY_ENV = live.DEFAULT_API_KEY_ENV
DEFAULT_MAX_CONCURRENCY = 4
MAX_NEW_TOKENS = COMPILER_OUTPUT_TOKEN_RESERVE
MAX_PROMPT_TOKENS = HARD_PROMPT_TOKEN_CAP - MAX_NEW_TOKENS

_JOURNAL_FILENAME_RE = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)
_REQUEST_KEYS = {
    "answer_output_token_reserve",
    "compiler_output_token_reserve",
    "format",
    "hard_total_token_cap",
    "leaf_handle_ids",
    "messages",
    "payload_class",
    "prompt_token_proxy",
    "question_sha256",
    "request_sha256",
    "selection_receipt_sha256",
    "shard_id",
    "shard_population_sha256",
    "topic_labels_for_scheduling_only",
}
_PROVIDER_INPUT_KEYS = {
    "dated_question",
    "evidence",
    "format",
    "frontier",
    "generic_obligations",
    "handles",
    "operator_spec",
    "response_schema",
    "story_links",
}
_FACT_KEYS = {
    "citations",
    "date",
    "entity",
    "kind",
    "numeric_value",
    "slot_ids",
    "status",
    "text",
    "unit",
}
_CITATION_KEYS = {"handle_id", "quote"}
_LEAF_BINDING_KEYS = {
    "handle_id",
    "leaf_receipt_sha256",
    "source_summary_sha256",
}
_REQUEST_ROW_KEYS = {
    "compiler_request",
    "compiler_request_sha256",
    "compiler_source_population_sha256",
    "format",
    "leaf_bindings",
    "messages_sha256",
    "question_sha256",
    "request_row_receipt_sha256",
    "selection_receipt_sha256",
}
_QUESTION_ROW_KEYS = {
    "compiler_request_sha256s",
    "format",
    "leaf_bindings",
    "question_row_receipt_sha256",
    "question_sha256",
    "selection_receipt_sha256",
}
_PREFLIGHT_KEYS = {
    "classified_a1_construction_artifact_sha256",
    "classified_a1_construction_identity_sha256",
    "classified_a1_replay_artifact_sha256",
    "compiler_id",
    "compiler_payload_class",
    "compiler_request_population_sha256",
    "compiler_source_population_sha256",
    "derived_provider_call_count",
    "disposition_artifact_sha256",
    "format",
    "gateway_url",
    "gold_loaded",
    "max_concurrency",
    "max_new_tokens",
    "max_prompt_tokens",
    "model",
    "model_prompt_population_sha256",
    "ordered_compiler_request_population_sha256",
    "ordinal_cli_routing_available",
    "physical_provider_calls",
    "preflight_identity_sha256",
    "production_ordinal_routing_enabled",
    "prompt_population_sha256",
    "question_count",
    "question_population_sha256",
    "question_rows",
    "request_rows",
    "retained_leaf_count",
    "retained_transformer_token_state_bytes",
    "runtime_firewall",
    "source_artifact_sha256",
    "source_replay_artifact_sha256",
}
_RELEASE_KEYS = {
    "approval_opt_in",
    "checkpoint_root",
    "checkpoint_root_sha256",
    "classified_a1_construction_artifact_sha256",
    "compiler_output_root",
    "compiler_output_root_sha256",
    "compiler_request_population_sha256",
    "derived_provider_call_count",
    "disposition_artifact_sha256",
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
    "target",
    "targets",
    "target_manifest",
}
_A1_RUNTIME_FIREWALL = {
    "benchmark_fields_loaded": False,
    "ordinal_routing_enabled": False,
    "protected_parent_loaded": False,
    "semantic_atom_manifest_loaded": False,
    "source_allowlist_loaded": False,
    "topic_labels_have_exclusion_authority": False,
}
_OUTPUT_RUNTIME_FIREWALL = {
    "gold_loaded": False,
    "ordinal_routing_enabled": False,
    "protected_parent_loaded": False,
    "reference_loaded": False,
    "semantic_atom_manifest_loaded": False,
    "source_allowlist_loaded": False,
    "targets_loaded": False,
}


class R7AfterUnionA1CompilerError(MatchedEvalContractError):
    """The classified A1 population, release, journal, or compilation changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise R7AfterUnionA1CompilerError(message)


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
            raise R7AfterUnionA1CompilerError(
                f"compiler JSON repeats object key: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise R7AfterUnionA1CompilerError(
        f"compiler JSON contains non-finite constant: {value}"
    )


def _strict_json(text: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise R7AfterUnionA1CompilerError(
            f"{label} must be one strict JSON object"
        ) from exc
    return _exact_dict(value, label)


def _messages(request: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    raw = _exact_list(request.get("messages"), "A1b compiler messages")
    messages: list[dict[str, str]] = []
    for value in raw:
        row = _exact_dict(value, "A1b compiler message")
        _require(
            set(row) == {"role", "content"}
            and row.get("role") in {"system", "user"}
            and type(row.get("content")) is str
            and bool(row.get("content")),
            "A1b compiler message envelope changed",
        )
        messages.append(
            {"role": str(row["role"]), "content": str(row["content"])}
        )
    _require(
        len(messages) == 2
        and messages[0]["role"] == "system"
        and messages[1]["role"] == "user",
        "A1b compiler messages must be the sealed system/user pair",
    )
    return tuple(messages)


def _compiler_source(provider_input: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dated_question": provider_input["dated_question"],
        "story_coherence": provider_input["story_links"],
        "typed_evidence": {
            "frontier": provider_input["frontier"],
            "handles": provider_input["handles"],
            "items": provider_input["evidence"],
            "operator_spec": provider_input["operator_spec"],
        },
    }


def _source_summary_by_handle(
    provider_input: Mapping[str, Any], handles: Sequence[str]
) -> dict[str, str]:
    known = set(handles)
    summaries: dict[str, str] = {}
    for raw in _exact_list(
        provider_input.get("evidence"), "A1b compiler evidence"
    ):
        item = _exact_dict(raw, "A1b compiler evidence item")
        item_handles = _exact_list(
            item.get("handle_ids"), "A1b compiler evidence handles"
        )
        summary = require_text(
            item.get("summary"), "A1b compiler evidence summary"
        )
        _require(
            len(item_handles) == 1
            and type(item_handles[0]) is str
            and item_handles[0] in known
            and item_handles[0] not in summaries,
            "A1b compiler evidence must bind one exact request handle",
        )
        summaries[str(item_handles[0])] = summary
    _require(
        set(summaries) == known,
        "A1b compiler evidence does not exactly cover its request handles",
    )
    return summaries


def _request_base(request: Mapping[str, Any]) -> dict[str, Any]:
    exact = _exact_dict(request, "A1b compiler request")
    request_sha = require_sha256(
        exact.get("request_sha256"), "A1b compiler request"
    )
    messages = _messages(exact)
    provider_input = _strict_json(
        messages[1]["content"], "A1b compiler provider input"
    )
    handles = tuple(
        require_text(value, "A1b compiler handle")
        for value in _exact_list(
            exact.get("leaf_handle_ids"), "A1b compiler handles"
        )
    )
    source = _compiler_source(provider_input)
    summaries = _source_summary_by_handle(provider_input, handles)
    represented = tuple(
        require_text(value, "A1b represented handle")
        for value in _exact_list(
            _exact_dict(
                provider_input.get("frontier"), "A1b compiler frontier"
            ).get("represented_handle_ids"),
            "A1b represented handles",
        )
    )
    _require(
        set(exact) == _REQUEST_KEYS
        and request_sha
        == identity_sha256(_without_receipt(exact, "request_sha256"))
        and exact.get("format") == a1.REQUEST_FORMAT
        and exact.get("payload_class") == COMPILER_PAYLOAD_CLASS
        and exact.get("compiler_output_token_reserve") == MAX_NEW_TOKENS
        and exact.get("hard_total_token_cap") == HARD_PROMPT_TOKEN_CAP
        and exact.get("answer_output_token_reserve")
        == a1.ANSWER_OUTPUT_TOKEN_RESERVE
        and type(exact.get("topic_labels_for_scheduling_only")) is list
        and bool(handles)
        and len(handles) == len(set(handles))
        and exact.get("shard_population_sha256")
        == identity_sha256(list(handles))
        and set(provider_input) == _PROVIDER_INPUT_KEYS
        and provider_input.get("format") == COMPILER_PROMPT_FORMAT
        and represented == handles
        and _exact_dict(
            provider_input.get("frontier"), "A1b compiler frontier"
        ).get("omitted_handle_ids")
        == []
        and _exact_dict(
            provider_input.get("frontier"), "A1b compiler frontier"
        ).get("truncated")
        is False
        and tuple(build_compiler_messages(source)) == messages
        and exact.get("prompt_token_proxy")
        == count_chat_prompt_token_proxy(messages)
        and _exact_int(
            exact.get("prompt_token_proxy"), "A1b compiler prompt tokens"
        )
        + MAX_NEW_TOKENS
        <= HARD_PROMPT_TOKEN_CAP
        and quote_sha256(
            require_text(
                provider_input.get("dated_question"),
                "A1b compiler dated question",
            )
        )
        == exact.get("question_sha256")
        and not _forbidden_provider_keys([dict(row) for row in messages])
        and not _forbidden_provider_keys(provider_input),
        "A1b compiler request or provider-message contract changed",
    )
    assert_gold_blind(provider_input, path="r7_a1b_compiler.provider")
    return {
        "compiler_request": dict(exact),
        "compiler_request_sha256": request_sha,
        "compiler_source": source,
        "messages_sha256": identity_sha256([dict(row) for row in messages]),
        "question_sha256": require_sha256(
            exact.get("question_sha256"), "A1b compiler question"
        ),
        "selection_receipt_sha256": require_sha256(
            exact.get("selection_receipt_sha256"),
            "A1b compiler selection",
        ),
        "source_summaries": summaries,
    }


def _request_projection(
    request: Mapping[str, Any],
    leaf_by_handle: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    base = _request_base(request)
    handles = tuple(base["compiler_request"]["leaf_handle_ids"])
    bindings: list[dict[str, str]] = []
    for handle in handles:
        leaf = _exact_dict(leaf_by_handle.get(handle), "A1b retained leaf")
        summary = str(base["source_summaries"][handle])
        _require(
            leaf.get("text") == summary,
            "A1b compiler evidence differs from its selected leaf",
        )
        bindings.append(
            {
                "handle_id": handle,
                "leaf_receipt_sha256": require_sha256(
                    leaf.get("receipt_sha256"), "A1b retained leaf receipt"
                ),
                "source_summary_sha256": quote_sha256(summary),
            }
        )
    source_population_sha = identity_sha256(bindings)
    body = {
        "compiler_request": base["compiler_request"],
        "compiler_request_sha256": base["compiler_request_sha256"],
        "compiler_source_population_sha256": source_population_sha,
        "format": REQUEST_ROW_FORMAT,
        "leaf_bindings": bindings,
        "messages_sha256": base["messages_sha256"],
        "question_sha256": base["question_sha256"],
        "selection_receipt_sha256": base["selection_receipt_sha256"],
    }
    return {**body, "request_row_receipt_sha256": identity_sha256(body)}


def _question_projection(question: Mapping[str, Any]) -> tuple[
    dict[str, Any], tuple[dict[str, Any], ...]
]:
    exact = _exact_dict(question, "classified A1 question")
    question_sha = require_sha256(
        exact.get("question_sha256"), "classified A1 question"
    )
    selection = _exact_dict(
        exact.get("semantic_selection"), "classified A1 selection"
    )
    selection_sha = require_sha256(
        selection.get("receipt_sha256"), "classified A1 selection"
    )
    raw_leaves = _exact_list(selection.get("leaves"), "classified A1 leaves")
    leaf_by_handle: dict[str, dict[str, Any]] = {}
    for raw in raw_leaves:
        leaf = _exact_dict(raw, "classified A1 leaf")
        handle = require_text(leaf.get("handle_id"), "classified A1 handle")
        _require(handle not in leaf_by_handle, "classified A1 leaves repeat")
        leaf_by_handle[handle] = leaf
    retained = tuple(
        require_text(value, "classified A1 retained handle")
        for value in _exact_list(
            _exact_dict(
                selection.get("semantic_result"),
                "classified A1 semantic result",
            ).get("retained_leaf_cell_ids"),
            "classified A1 retained handles",
        )
    )
    raw_requests = _exact_list(
        exact.get("compiler_requests"), "classified A1 compiler requests"
    )
    requests = tuple(
        _request_projection(_exact_dict(row, "A1b compiler request"), leaf_by_handle)
        for row in raw_requests
    )
    request_shas = tuple(str(row["compiler_request_sha256"]) for row in requests)
    request_handles = tuple(
        binding["handle_id"]
        for row in requests
        for binding in row["leaf_bindings"]
    )
    retained_bindings = [
        {
            "handle_id": handle,
            "leaf_receipt_sha256": require_sha256(
                leaf_by_handle[handle].get("receipt_sha256"),
                "classified A1 retained leaf receipt",
            ),
            "source_summary_sha256": quote_sha256(
                require_text(
                    leaf_by_handle[handle].get("text"),
                    "classified A1 retained leaf text",
                )
            ),
        }
        for handle in retained
    ]
    _require(
        len(set(request_shas)) == len(request_shas)
        and all(row["question_sha256"] == question_sha for row in requests)
        and all(
            row["selection_receipt_sha256"] == selection_sha for row in requests
        )
        and request_handles == retained
        and [binding for row in requests for binding in row["leaf_bindings"]]
        == retained_bindings
        and exact.get("compiler_request_count") == len(requests)
        and exact.get("actionable_compiler_request_count") == len(requests)
        and exact.get("request_population_sha256")
        == identity_sha256(list(request_shas))
        and exact.get("missing_compiler_request_sha256s") == list(request_shas)
        and exact.get("missing_classifier_request_sha256s") == [],
        "classified A1 question compiler population changed",
    )
    body = {
        "compiler_request_sha256s": list(request_shas),
        "format": QUESTION_ROW_FORMAT,
        "leaf_bindings": retained_bindings,
        "question_sha256": question_sha,
        "selection_receipt_sha256": selection_sha,
    }
    return (
        {**body, "question_row_receipt_sha256": identity_sha256(body)},
        requests,
    )


def _load_classified_pair(
    args: argparse.Namespace,
) -> tuple[SealedArtifact, SealedArtifact]:
    root = Path(args.classified_root)
    construction = _read_expected(
        root / a1_cli.CONSTRUCTION_NAME,
        str(args.expected_classified_construction_sha256),
        "classified A1 construction",
    )
    replay = _read_expected(
        root / a1_cli.REPLAY_NAME,
        str(args.expected_classified_replay_sha256),
        "classified A1 replay",
    )
    payload = construction.payload
    expected_disposition_sha = require_sha256(
        args.expected_disposition_artifact_sha256,
        "expected classified A1 disposition artifact",
    )
    _require(
        payload.get("disposition_artifact_sha256")
        == expected_disposition_sha,
        "classified A1 disposition artifact changed",
    )
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
        and payload.get("compiler_payload_class") == COMPILER_PAYLOAD_CLASS
        and payload.get("missing_classifier_call_count") == 0
        and payload.get("missing_classifier_request_sha256s") == []
        and require_sha256(
            payload.get("disposition_artifact_sha256"),
            "classified A1 dispositions",
        )
        and payload.get("compiler_output_artifact_sha256") is None
        and payload.get("construction_status")
        == "preflight_external_compilation_required"
        and payload.get("compiler_workload_status")
        == "sealed_disposition_bound",
        "sealed classified A1 construction/replay contract changed",
    )
    assert_gold_blind(payload, path="r7_a1b_compiler.classified_a1")
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
        "A1b compiler preflight runtime policy changed",
    )
    question_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    for raw in _exact_list(payload.get("questions"), "classified A1 questions"):
        question, requests = _question_projection(
            _exact_dict(raw, "classified A1 question")
        )
        question_rows.append(question)
        request_rows.extend(requests)
    request_shas = [str(row["compiler_request_sha256"]) for row in request_rows]
    prompts = tuple(_messages(row["compiler_request"]) for row in request_rows)
    prompt_population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_PROMPT_TOKENS
    )
    derived_calls = len(request_rows)
    compiler_population_sha = identity_sha256(sorted(request_shas))
    source_population_sha = identity_sha256(
        [row["compiler_source_population_sha256"] for row in request_rows]
    )
    _require(
        derived_calls > 0
        and len(set(request_shas)) == derived_calls
        and prompt_population.logical_prompt_count
        == prompt_population.unique_prompt_count
        == derived_calls
        and payload.get("question_count") == len(question_rows)
        and payload.get("expected_question_count") == len(question_rows)
        and payload.get("compiler_request_count") == derived_calls
        and payload.get("actionable_compiler_request_count") == derived_calls
        and payload.get("missing_compiler_call_count") == derived_calls
        and payload.get("missing_external_request_sha256s") == request_shas
        and payload.get("missing_external_call_count") == derived_calls
        and all(
            row["messages_sha256"]
            == prompt_population.ordered_rows[index].messages_sha256
            for index, row in enumerate(request_rows)
        ),
        "classified A1 exact compiler request population changed",
    )
    model_prompt_sha = identity_sha256(
        {
            "compiler_request_population_sha256": compiler_population_sha,
            "format": MODEL_PROMPT_POPULATION_FORMAT,
            "model": model,
            "ordered_compiler_request_sha256s": request_shas,
            "prompt_population_sha256": prompt_population.prompt_population_sha256,
        }
    )
    retained_count = sum(len(row["leaf_bindings"]) for row in question_rows)
    body = {
        "classified_a1_construction_artifact_sha256": construction.sha256,
        "classified_a1_construction_identity_sha256": payload[
            "construction_identity_sha256"
        ],
        "classified_a1_replay_artifact_sha256": replay.sha256,
        "compiler_id": COMPILER_ID,
        "compiler_payload_class": COMPILER_PAYLOAD_CLASS,
        "compiler_request_population_sha256": compiler_population_sha,
        "compiler_source_population_sha256": source_population_sha,
        "derived_provider_call_count": derived_calls,
        "disposition_artifact_sha256": payload["disposition_artifact_sha256"],
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "max_concurrency": max_concurrency,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "model": model,
        "model_prompt_population_sha256": model_prompt_sha,
        "ordered_compiler_request_population_sha256": identity_sha256(
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
        "retained_leaf_count": retained_count,
        "retained_transformer_token_state_bytes": 0,
        "runtime_firewall": dict(_OUTPUT_RUNTIME_FIREWALL),
        "source_artifact_sha256": require_sha256(
            payload.get("source_artifact_sha256"), "classified A1 source"
        ),
        "source_replay_artifact_sha256": require_sha256(
            payload.get("source_replay_artifact_sha256"),
            "classified A1 source replay",
        ),
    }
    result = {**body, "preflight_identity_sha256": identity_sha256(body)}
    assert_gold_blind(result, path="r7_a1b_compiler.preflight")
    return result, prompts


def _validate_leaf_bindings(
    value: object,
    source_summaries: Mapping[str, str],
    *,
    label: str,
) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []
    for raw in _exact_list(value, label):
        binding = _exact_dict(raw, f"{label} row")
        handle = require_text(binding.get("handle_id"), f"{label} handle")
        _require(
            set(binding) == _LEAF_BINDING_KEYS
            and handle in source_summaries
            and binding.get("source_summary_sha256")
            == quote_sha256(source_summaries[handle]),
            f"{label} source binding changed",
        )
        bindings.append(
            {
                "handle_id": handle,
                "leaf_receipt_sha256": require_sha256(
                    binding.get("leaf_receipt_sha256"), f"{label} leaf receipt"
                ),
                "source_summary_sha256": require_sha256(
                    binding.get("source_summary_sha256"),
                    f"{label} source summary",
                ),
            }
        )
    _require(
        len(bindings) == len({row["handle_id"] for row in bindings}),
        f"{label} handles repeat",
    )
    return bindings


def _validate_preflight(
    artifact: SealedArtifact,
) -> tuple[
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    payload = artifact.payload
    body = _without_receipt(payload, "preflight_identity_sha256")
    requests: list[dict[str, Any]] = []
    for raw in _exact_list(
        payload.get("request_rows"), "A1b compiler preflight requests"
    ):
        row = _exact_dict(raw, "A1b compiler preflight request")
        base = _request_base(
            _exact_dict(row.get("compiler_request"), "A1b compiler request")
        )
        bindings = _validate_leaf_bindings(
            row.get("leaf_bindings"),
            base["source_summaries"],
            label="A1b compiler request bindings",
        )
        _require(
            set(row) == _REQUEST_ROW_KEYS
            and tuple(binding["handle_id"] for binding in bindings)
            == tuple(base["compiler_request"]["leaf_handle_ids"])
            and row.get("compiler_request_sha256")
            == base["compiler_request_sha256"]
            and row.get("messages_sha256") == base["messages_sha256"]
            and row.get("question_sha256") == base["question_sha256"]
            and row.get("selection_receipt_sha256")
            == base["selection_receipt_sha256"]
            and row.get("compiler_source_population_sha256")
            == identity_sha256(bindings)
            and row.get("request_row_receipt_sha256")
            == identity_sha256(
                _without_receipt(row, "request_row_receipt_sha256")
            ),
            "A1b compiler preflight request row changed",
        )
        requests.append(row)
    request_by_sha = {
        str(row["compiler_request_sha256"]): row for row in requests
    }
    _require(
        len(request_by_sha) == len(requests),
        "A1b compiler preflight request rows repeat",
    )
    questions: list[dict[str, Any]] = []
    for raw in _exact_list(
        payload.get("question_rows"), "A1b compiler preflight questions"
    ):
        row = _exact_dict(raw, "A1b compiler preflight question")
        request_shas = [
            require_sha256(value, "A1b compiler question request")
            for value in _exact_list(
                row.get("compiler_request_sha256s"),
                "A1b compiler question requests",
            )
        ]
        question_sha = require_sha256(
            row.get("question_sha256"), "A1b compiler question"
        )
        selection_sha = require_sha256(
            row.get("selection_receipt_sha256"), "A1b compiler selection"
        )
        _require(
            all(request_sha in request_by_sha for request_sha in request_shas),
            "A1b compiler question contains an unknown request",
        )
        source_summaries = {
            str(binding["handle_id"]): str(
                _request_base(request_by_sha[request_sha]["compiler_request"])[
                    "source_summaries"
                ][binding["handle_id"]]
            )
            for request_sha in request_shas
            for binding in request_by_sha[request_sha]["leaf_bindings"]
        }
        bindings = _validate_leaf_bindings(
            row.get("leaf_bindings"),
            source_summaries,
            label="A1b compiler question bindings",
        )
        request_bindings = [
            dict(binding)
            for request_sha in request_shas
            for binding in request_by_sha[request_sha]["leaf_bindings"]
        ]
        _require(
            set(row) == _QUESTION_ROW_KEYS
            and len(request_shas) == len(set(request_shas))
            and all(
                request_by_sha[request_sha]["question_sha256"] == question_sha
                and request_by_sha[request_sha]["selection_receipt_sha256"]
                == selection_sha
                for request_sha in request_shas
            )
            and bindings == request_bindings
            and row.get("question_row_receipt_sha256")
            == identity_sha256(
                _without_receipt(row, "question_row_receipt_sha256")
            ),
            "A1b compiler preflight question row changed",
        )
        questions.append(row)
    prompts = tuple(_messages(row["compiler_request"]) for row in requests)
    prompt_population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_PROMPT_TOKENS
    )
    request_shas = [str(row["compiler_request_sha256"]) for row in requests]
    question_request_shas = [
        request_sha
        for row in questions
        for request_sha in row["compiler_request_sha256s"]
    ]
    request_bindings = [
        dict(binding)
        for row in requests
        for binding in row["leaf_bindings"]
    ]
    question_bindings = [
        dict(binding)
        for row in questions
        for binding in row["leaf_bindings"]
    ]
    derived_calls = len(requests)
    compiler_population_sha = identity_sha256(sorted(request_shas))
    model_prompt_sha = identity_sha256(
        {
            "compiler_request_population_sha256": compiler_population_sha,
            "format": MODEL_PROMPT_POPULATION_FORMAT,
            "model": payload.get("model"),
            "ordered_compiler_request_sha256s": request_shas,
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
        and payload.get("runtime_firewall") == _OUTPUT_RUNTIME_FIREWALL
        and payload.get("ordinal_cli_routing_available") is False
        and payload.get("production_ordinal_routing_enabled") is False
        and payload.get("compiler_id") == COMPILER_ID
        and payload.get("compiler_payload_class") == COMPILER_PAYLOAD_CLASS
        and payload.get("model") == DEFAULT_MODEL
        and payload.get("gateway_url") == DEFAULT_GATEWAY_URL
        and type(payload.get("max_concurrency")) is int
        and payload.get("max_concurrency") > 0
        and payload.get("max_new_tokens") == MAX_NEW_TOKENS
        and payload.get("max_prompt_tokens") == MAX_PROMPT_TOKENS
        and derived_calls > 0
        and payload.get("derived_provider_call_count") == derived_calls
        and len(set(request_shas)) == derived_calls
        and prompt_population.logical_prompt_count
        == prompt_population.unique_prompt_count
        == derived_calls
        and question_request_shas == request_shas
        and question_bindings == request_bindings
        and payload.get("compiler_request_population_sha256")
        == compiler_population_sha
        and payload.get("ordered_compiler_request_population_sha256")
        == identity_sha256(request_shas)
        and payload.get("compiler_source_population_sha256")
        == identity_sha256(
            [row["compiler_source_population_sha256"] for row in requests]
        )
        and payload.get("prompt_population_sha256")
        == prompt_population.prompt_population_sha256
        and payload.get("model_prompt_population_sha256") == model_prompt_sha
        and payload.get("question_count") == len(questions)
        and payload.get("question_population_sha256")
        == identity_sha256(
            [row["question_row_receipt_sha256"] for row in questions]
        )
        and payload.get("retained_leaf_count") == len(question_bindings)
        and len({row["question_sha256"] for row in questions})
        == len(questions)
        and all(
            require_sha256(payload.get(key), f"A1b compiler preflight {key}")
            for key in (
                "classified_a1_construction_artifact_sha256",
                "classified_a1_construction_identity_sha256",
                "classified_a1_replay_artifact_sha256",
                "disposition_artifact_sha256",
                "source_artifact_sha256",
                "source_replay_artifact_sha256",
            )
        ),
        "sealed A1b compiler preflight changed",
    )
    assert_gold_blind(payload, path="r7_a1b_compiler.preflight")
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
        "A1b compiler preflight",
    )
    prompts, requests, questions = _validate_preflight(artifact)
    return artifact, prompts, requests, questions


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    _require(
        _canonical_root(output_root) != _canonical_root(args.classified_root),
        "A1b compiler output root must differ from the classified A1 root",
    )
    _require(
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "A1b compiler preflight requires a fresh absent checkpoint root",
    )
    construction, replay = _load_classified_pair(args)
    payload, _ = build_preflight_payload(
        construction,
        replay,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(output_root / PREFLIGHT_NAME, payload)
    return {
        "classified_a1_construction_sha256": construction.sha256,
        "created": created,
        "derived_provider_call_count": payload["derived_provider_call_count"],
        "disposition_artifact_sha256": payload["disposition_artifact_sha256"],
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
        "compiler_output_root": root,
        "compiler_output_root_sha256": identity_sha256(
            {"canonical_root": root}
        ),
        "compiler_request_population_sha256": preflight.payload[
            "compiler_request_population_sha256"
        ],
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
        "approval_opt_in": True,
        "checkpoint_root": owner["checkpoint_root"],
        "checkpoint_root_sha256": owner["checkpoint_root_sha256"],
        "classified_a1_construction_artifact_sha256": preflight.payload[
            "classified_a1_construction_artifact_sha256"
        ],
        "compiler_output_root": owner["compiler_output_root"],
        "compiler_output_root_sha256": owner["compiler_output_root_sha256"],
        "compiler_request_population_sha256": preflight.payload[
            "compiler_request_population_sha256"
        ],
        "derived_provider_call_count": preflight.payload[
            "derived_provider_call_count"
        ],
        "disposition_artifact_sha256": preflight.payload[
            "disposition_artifact_sha256"
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
        "source_artifact_sha256": preflight.payload["source_artifact_sha256"],
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
    body = _without_receipt(payload, "release_identity_sha256")
    owner = _journal_owner_body(preflight, output_root=output_root)
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
        and payload.get("compiler_request_population_sha256")
        == preflight.payload.get("compiler_request_population_sha256")
        and payload.get("classified_a1_construction_artifact_sha256")
        == preflight.payload.get("classified_a1_construction_artifact_sha256")
        and payload.get("disposition_artifact_sha256")
        == preflight.payload.get("disposition_artifact_sha256")
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
        "A1b compiler provider release changed",
    )
    assert_gold_blind(payload, path="r7_a1b_compiler.release")


def _read_release(
    output_root: str | Path,
    expected_sha256: str,
    *,
    preflight: SealedArtifact,
) -> SealedArtifact:
    artifact = _read_expected(
        Path(output_root) / RELEASE_NAME,
        expected_sha256,
        "A1b compiler release",
    )
    _validate_release(artifact, preflight=preflight, output_root=output_root)
    return artifact


def run_approve_release(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    _require(
        args.approve_provider_release is True,
        "A1b compiler release requires explicit provider approval",
    )
    _require(
        not (output_root / CHECKPOINT_DIR_NAME).exists(),
        "A1b compiler release requires an absent checkpoint root",
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
    call_count = int(preflight.payload["derived_provider_call_count"])
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
        "A1b compiler runtime differs from sealed release",
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
            "arm": FORMAT,
            "authorized_unique_calls": call_count,
            "classified_a1_construction_artifact_sha256": preflight.payload[
                "classified_a1_construction_artifact_sha256"
            ],
            "compiler_request_population_sha256": preflight.payload[
                "compiler_request_population_sha256"
            ],
            "disposition_artifact_sha256": preflight.payload[
                "disposition_artifact_sha256"
            ],
            "experiment_format": a1.COMPILER_OUTPUTS_FORMAT,
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
        "A1b compiler checkpoint root must be a regular directory",
    )
    requests: set[str] = set()
    responses: set[str] = set()
    for path in root.iterdir():
        _require(
            not path.is_symlink() and path.is_file(),
            "A1b compiler checkpoint root contains foreign state",
        )
        if path.name == ".fast-completion-journal.lock":
            continue
        match = _JOURNAL_FILENAME_RE.fullmatch(path.name)
        _require(
            match is not None,
            "A1b compiler checkpoint root contains foreign journal state",
        )
        assert match is not None
        target = requests if match.group("kind") == "request" else responses
        target.add(match.group("key"))
    _require(
        requests == responses,
        "A1b compiler checkpoint pair is incomplete; unsafe retry forbidden",
    )
    _require(
        len(requests) <= call_count,
        "A1b compiler checkpoint population exceeds sealed calls",
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
    _require(
        len(records) <= int(preflight.payload["derived_provider_call_count"]),
        "A1b compiler checkpoints escaped the sealed prompt population",
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
        "A1b compiler provider requires bounded Terra authorization",
    )
    candidate_hits = _read_only_checkpoint_count(args.output_root, call_count)
    remaining = call_count - candidate_hits
    _require(
        args.authorized_provider_calls == remaining,
        "A1b compiler authorization must exactly equal remaining calls",
    )
    checkpoint_hits = _validated_checkpoint_hits(
        preflight, release, prompts, args=args
    )
    _require(
        checkpoint_hits == candidate_hits,
        "A1b compiler checkpoint count changed after authorization",
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
            "A1b compiler completed checkpoint replay changed",
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
            "A1b compiler provider population changed",
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


def _validate_compiler_completion(
    request_row: Mapping[str, Any], response_text: str
) -> tuple[TypedFactCompilation, dict[str, Any]]:
    request = _exact_dict(
        request_row.get("compiler_request"), "A1b materialized compiler request"
    )
    base = _request_base(request)
    source = _exact_dict(base["compiler_source"], "A1b compiler source")
    summaries = {
        str(key): str(value) for key, value in base["source_summaries"].items()
    }
    response = _strict_json(response_text, "A1b compiler response")
    _require(
        set(response) == {"facts"},
        "A1b compiler response must contain exactly facts",
    )
    raw_facts = _exact_list(response.get("facts"), "A1b compiler facts")
    _require(
        len(raw_facts) <= MAX_COMPILER_FACTS,
        "A1b compiler response exceeds its fact bound",
    )
    raw_cited: list[str] = []
    for raw_index, raw_fact in enumerate(raw_facts):
        fact = _exact_dict(raw_fact, f"A1b compiler fact {raw_index}")
        _require(
            set(fact) == _FACT_KEYS,
            "A1b compiler fact schema changed or omitted citations",
        )
        citations = _exact_list(
            fact.get("citations"), f"A1b compiler fact {raw_index} citations"
        )
        _require(
            1 <= len(citations) <= 8,
            "A1b compiler fact must have an exact citation",
        )
        seen: set[tuple[str, str]] = set()
        for raw_citation in citations:
            citation = _exact_dict(raw_citation, "A1b compiler citation")
            handle = require_text(
                citation.get("handle_id"), "A1b compiler citation handle"
            )
            quote = require_text(
                citation.get("quote"), "A1b compiler citation quote"
            )
            _require(
                set(citation) == _CITATION_KEYS
                and handle in summaries
                and quote in summaries[handle]
                and (handle, quote) not in seen,
                "A1b compiler citation is not exact admitted handle evidence",
            )
            seen.add((handle, quote))
            raw_cited.append(handle)
    try:
        compilation = parse_compiler_completion(source, response_text)
    except (MatchedEvalContractError, KeyError, TypeError, ValueError) as exc:
        raise R7AfterUnionA1CompilerError(
            "A1b compiler response failed the typed-fact validator"
        ) from exc
    _require(
        len(compilation.rejected) == 0
        and len(compilation.accepted_before_dedup) == len(raw_facts),
        "A1b compiler response contains a rejected or malformed fact",
    )
    request_handles = tuple(request["leaf_handle_ids"])
    raw_cited_set = set(raw_cited)
    resolved_set = {
        handle
        for fact in compilation.packet.facts
        for handle in fact.handle_ids
    }
    _require(
        raw_cited_set <= set(request_handles)
        and resolved_set <= raw_cited_set,
        "A1b compiler facts escaped their exact request population",
    )
    validation = {
        "accepted_fact_count": len(compilation.accepted_before_dedup),
        "compilation_receipt_sha256": compilation.receipt_sha256,
        "duplicate_fact_count": compilation.duplicate_count,
        "packet_receipt_sha256": compilation.packet.receipt_sha256,
        "packet_valid": compilation.packet.valid,
        "raw_cited_leaf_handle_ids": [
            handle for handle in request_handles if handle in raw_cited_set
        ],
        "rejected_fact_count": 0,
        "resolved_leaf_handle_ids": [
            handle for handle in request_handles if handle in resolved_set
        ],
        "unresolved_leaf_handle_ids": [
            handle for handle in request_handles if handle not in resolved_set
        ],
    }
    return compilation, validation


def _record_by_messages(
    batch: FastCompletionBatch,
) -> dict[str, FastCompletionRecord]:
    result = {row.messages_sha256: row for row in batch.unique_records}
    _require(
        len(result) == len(batch.unique_records),
        "A1b compiler completion records repeat",
    )
    return result


def _outputs_payload(
    preflight: SealedArtifact,
    release: SealedArtifact,
    request_rows: Sequence[Mapping[str, Any]],
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
        "A1b compiler checkpoint-only completion batch changed",
    )
    record_by_messages = _record_by_messages(batch)
    responses: list[dict[str, str]] = []
    response_bindings: list[dict[str, Any]] = []
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
            "A1b compiler completion record binding changed",
        )
        _compilation, validation = _validate_compiler_completion(
            request_row, completion
        )
        request_sha = str(request_row["compiler_request_sha256"])
        responses.append(
            {
                "request_sha256": request_sha,
                "response_sha256": record.completion_sha256,
                "response_text": completion,
            }
        )
        leaf_outcomes = [
            {
                "handle_id": binding["handle_id"],
                "leaf_receipt_sha256": binding["leaf_receipt_sha256"],
                "outcome": (
                    "facts"
                    if binding["handle_id"]
                    in validation["resolved_leaf_handle_ids"]
                    else "unresolved"
                ),
            }
            for binding in request_row["leaf_bindings"]
        ]
        body = {
            "accepted_fact_count": validation["accepted_fact_count"],
            "call_key_sha256": record.call_key_sha256,
            "compilation_receipt_sha256": validation[
                "compilation_receipt_sha256"
            ],
            "compiler_source_population_sha256": request_row[
                "compiler_source_population_sha256"
            ],
            "duplicate_fact_count": validation["duplicate_fact_count"],
            "format": RESPONSE_BINDING_FORMAT,
            "leaf_outcomes": leaf_outcomes,
            "messages_sha256": messages_sha,
            "packet_receipt_sha256": validation["packet_receipt_sha256"],
            "packet_valid": validation["packet_valid"],
            "question_sha256": request_row["question_sha256"],
            "raw_cited_leaf_handle_ids": validation[
                "raw_cited_leaf_handle_ids"
            ],
            "rejected_fact_count": 0,
            "request_journal_sha256": record.request_journal_sha256,
            "request_sha256": request_sha,
            "resolved_leaf_handle_ids": validation[
                "resolved_leaf_handle_ids"
            ],
            "response_journal_sha256": record.response_journal_sha256,
            "response_sha256": record.completion_sha256,
            "selection_receipt_sha256": request_row[
                "selection_receipt_sha256"
            ],
            "source_artifact_sha256": preflight.payload[
                "source_artifact_sha256"
            ],
            "unresolved_leaf_handle_ids": validation[
                "unresolved_leaf_handle_ids"
            ],
        }
        response_bindings.append(
            {**body, "response_binding_receipt_sha256": identity_sha256(body)}
        )
    binding_receipts = [
        row["response_binding_receipt_sha256"] for row in response_bindings
    ]
    value = {
        "classified_a1_construction_artifact_sha256": preflight.payload[
            "classified_a1_construction_artifact_sha256"
        ],
        "classified_a1_replay_artifact_sha256": preflight.payload[
            "classified_a1_replay_artifact_sha256"
        ],
        "compiler_id": COMPILER_ID,
        "compiler_request_population_sha256": preflight.payload[
            "compiler_request_population_sha256"
        ],
        "completion_runtime_identity_sha256": batch.runtime_identity_sha256,
        "derived_provider_call_count": call_count,
        "disposition_artifact_sha256": preflight.payload[
            "disposition_artifact_sha256"
        ],
        "format": a1.COMPILER_OUTPUTS_FORMAT,
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
        "release_authorization_artifact_sha256": release.sha256,
        "response_binding_population_sha256": identity_sha256(binding_receipts),
        "response_bindings": response_bindings,
        "response_count": len(responses),
        "responses": responses,
        "retained_transformer_token_state_bytes": 0,
        "runtime_firewall": dict(_OUTPUT_RUNTIME_FIREWALL),
        "source_artifact_sha256": preflight.payload[
            "source_artifact_sha256"
        ],
        "source_replay_artifact_sha256": preflight.payload[
            "source_replay_artifact_sha256"
        ],
    }
    assert_gold_blind(value, path="r7_a1b_compiler.outputs")
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
        "A1b compiler materialization requires every complete checkpoint",
    )
    return _checkpoint_batch(
        preflight, release, prompts, args=args, client=None
    )


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, requests, _questions = _read_preflight(
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
    payload = _outputs_payload(preflight, release, requests, batch)
    artifact, created = publish_sealed_json(
        Path(args.output_root) / OUTPUTS_NAME, payload
    )
    return {
        "checkpoint_hits": payload["derived_provider_call_count"],
        "compiler_outputs_sha256": artifact.sha256,
        "created": created,
        "derived_provider_call_count": payload["derived_provider_call_count"],
        "physical_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, requests, _questions = _read_preflight(
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
    rebuilt = _outputs_payload(preflight, release, requests, batch)
    root = Path(args.output_root)
    artifact = _read_expected(
        root / OUTPUTS_NAME,
        str(args.expected_compiler_outputs_sha256),
        "A1b compiler outputs",
    )
    _require(
        artifact.payload == rebuilt,
        "A1b compiler outputs differ from checkpoint-only replay",
    )
    replay, _ = publish_sealed_json(root / REPLAY_NAME, rebuilt)
    _require(
        replay.sha256 == artifact.sha256,
        "A1b compiler replay is not byte-identical",
    )
    return {
        "byte_identical": True,
        "compiler_outputs_sha256": artifact.sha256,
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
    preflight.add_argument(
        "--classified-root", type=Path, default=DEFAULT_CLASSIFIED_ROOT
    )
    preflight.add_argument(
        "--expected-classified-construction-sha256",
        default=EXPECTED_CLASSIFIED_SHA256,
    )
    preflight.add_argument(
        "--expected-classified-replay-sha256",
        default=EXPECTED_CLASSIFIED_SHA256,
    )
    preflight.add_argument(
        "--expected-disposition-artifact-sha256",
        "--expected-disposition-sha256",
        dest="expected_disposition_artifact_sha256",
        default=EXPECTED_DISPOSITION_ARTIFACT_SHA256,
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
    replay.add_argument("--expected-compiler-outputs-sha256", required=True)
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
    "COMPILER_ID",
    "DEFAULT_CLASSIFIED_ROOT",
    "DEFAULT_GATEWAY_URL",
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT_ROOT",
    "EXPECTED_CLASSIFIED_SHA256",
    "EXPECTED_DISPOSITION_ARTIFACT_SHA256",
    "OUTPUTS_NAME",
    "PREFLIGHT_NAME",
    "RELEASE_NAME",
    "REPLAY_NAME",
    "R7AfterUnionA1CompilerError",
    "build_parser",
    "build_preflight_payload",
    "main",
    "run_approve_release",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
]
