#!/usr/bin/env python3
"""Checkpointed Terra answers for the frozen terminal semantic assay v3.

The construction validator owns semantic search, compact persistence, and
selected-then-dedup proofs.  This lifecycle independently owns the last
provider boundary: every retained segment must be byte-identically visible in
the fitted typed packet, the protected parent prediction is immutable, the
complete chat envelope is at most 8k tokens, and materialization may only read
already checkpointed completions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from tools import run_locked_specialist_final_answer as specialist_v1  # noqa: E402
from tools.matched_eval import live  # noqa: E402
from tools.matched_eval import typed_memory_final_arm as typed_final  # noqa: E402
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


FORMAT = "memory-condense-locked-semantic-final-terra-answer-v3"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight"
RESULT_ROW_FORMAT = f"{FORMAT}-result-row"
SEMANTIC_PROMPT_FORMAT = f"{typed_final.PROMPT_ROW_FORMAT}-render-final-messages-v1"

CONSTRUCTION_FORMAT = (
    "memory-condense-reduced-semantic-binary-search-assay-v3-construction-v1"
)
CLASSIFIED_CLOSURE_FORMAT = (
    "memory-condense-reduced-semantic-binary-search-assay-v3-"
    "classified-closure-v1"
)
STORED_SEARCH_FORMAT = (
    "memory-condense-reduced-semantic-binary-search-assay-v3-"
    "stored-semantic-search-v1"
)
LOCAL_AUDIT_FORMAT = (
    "memory-condense-reduced-semantic-binary-search-assay-v3-"
    "semantic-local-audit-v1"
)

PREFLIGHT_NAME = "locked-semantic-final-answer-preflight-v3.json"
RUN_NAME = "locked-semantic-final-answer-v3.json"
REPLAY_NAME = "locked-semantic-final-answer-replay-v3.json"
CHECKPOINT_DIR_NAME = "locked-semantic-final-answer-checkpoints-v3"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONSTRUCTION = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/reduced-semantic-binary-search-missing4-v3/"
    "reduced-semantic-binary-search-construction-v3.json"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-semantic-final-answer-v3"
)
DEFAULT_MODEL = "codex_sdk/gpt-5.6-terra"

QUESTION_ORDINALS = (42, 65, 74, 79)
QUESTION_COUNT = len(QUESTION_ORDINALS)
SEMANTIC_MODE = "semantic_residual"
PARENT_PASSTHROUGH_MODE = "parent_passthrough"
_MODES = frozenset({SEMANTIC_MODE, PARENT_PASSTHROUGH_MODE})

HARD_COMPLETE_CHAT_TOKEN_CAP = 8_000
OUTPUT_TOKEN_RESERVE = 768
MAX_CHAT_PROMPT_TOKENS = HARD_COMPLETE_CHAT_TOKEN_CAP - OUTPUT_TOKEN_RESERVE

_SOURCE_FIELDS = frozenset(
    {
        "additive_composition",
        "additive_composition_local_audit",
        "classified_closure",
        "dated_question_sha256",
        "fallback_reason",
        "fitted_typed_prompt",
        "mode",
        "namespace_id",
        "new_provider_calls",
        "ordinal",
        "parent_source",
        "query_vector_artifact_sha256",
        "query_vector_row_receipt_sha256",
        "question_id",
        "question_receipt_sha256",
        "question_sha256",
        "retained_transformer_token_state_bytes",
        "semantic_query",
        "semantic_residual_index_receipt_sha256",
        "semantic_residual_local_audit",
        "semantic_residual_search",
        "terminal_prompt",
    }
)

_FITTED_FIELDS = frozenset(
    {
        "allowed_handle_ids",
        "dropped_binding_receipt_sha256s",
        "dropped_item_receipt_sha256s",
        "execution_receipt_sha256",
        "format",
        "full_chat_plus_output_tokens",
        "handle_group_by_id",
        "hard_prompt_token_cap",
        "local_bindings",
        "local_retention_priority_receipt_sha256",
        "mechanism_by_handle",
        "messages_sha256",
        "output_token_reserve",
        "packet_receipt_sha256",
        "preservation_requirements",
        "prompt_token_proxy",
        "protected_binding_receipt_sha256s",
        "protected_item_receipt_sha256s",
        "protection_source_receipt_sha256",
        "provider_input",
        "receipt_sha256",
        "retained_transformer_token_state_bytes",
        "story_coherence",
        "story_link_local_bindings",
        "validation_contract",
    }
)

_TERMINAL_FIELDS = frozenset(
    {
        "fitted_prompt_receipt_sha256",
        "full_chat_plus_output_tokens",
        "hard_prompt_token_cap",
        "messages",
        "messages_sha256",
        "output_token_reserve",
        "prompt_token_proxy",
        "provider_input",
        "provider_prompt_count",
        "rendered_messages_utf8_byte_count",
        "rendered_messages_utf8_sha256",
        "retained_transformer_token_state_bytes",
        "terminal_prompt_receipt_sha256",
    }
)

_PROVIDER_FIELDS = frozenset(
    {
        "dated_question",
        "deterministic_execution_advisory",
        "format",
        "protected_parent_fallback",
        "response_schema",
        "scalar_validation_advisory",
        "story_coherence",
        "typed_evidence",
    }
)

_CLOSURE_FIELDS = frozenset(
    {
        "all_retained_segments_provider_visible",
        "classified_frontier_receipt_sha256",
        "closed",
        "complete_leaf_partition",
        "fitted_prompt_receipt_sha256",
        "format",
        "post_selection_dedup_audit_receipt_sha256",
        "protection_source_receipt_sha256",
        "receipt_sha256",
        "retained_segment_receipt_sha256s",
        "rows",
        "semantic_residual_search_receipt_sha256",
        "terminal_allowed_handle_ids",
        "terminal_allowed_handle_ids_sha256",
    }
)

_CLOSURE_ROW_FIELDS = frozenset(
    {
        "cell_id",
        "dedup_exclusion_sha256",
        "disposition",
        "exact_text_sha256",
        "residual_binding_receipt_sha256",
        "residual_evidence_receipt_sha256",
        "residual_item_receipt_sha256",
        "segment_receipt_sha256",
        "visible_binding_receipt_sha256s",
        "visible_handle_ids",
        "visible_item_receipt_sha256",
    }
)

_COMMON_PLAN_FIELDS = frozenset(
    {
        "answer_plan_receipt_sha256",
        "construction_question_receipt_sha256",
        "dated_question_sha256",
        "fallback_reason",
        "mode",
        "namespace_id",
        "ordinal",
        "parent_judge_row_sha256",
        "parent_prediction",
        "parent_prediction_sha256",
        "parent_prediction_source",
        "parent_replay_artifact_sha256",
        "parent_run_artifact_sha256",
        "parent_source_receipt_sha256",
        "parent_source_row_sha256",
        "query_vector_artifact_sha256",
        "query_vector_row_receipt_sha256",
        "question_id",
        "question_sha256",
        "route_id",
        "semantic_query_receipt_sha256",
        "semantic_residual_index_receipt_sha256",
        "semantic_residual_local_audit_sha256",
        "semantic_residual_search_receipt_sha256",
    }
)

_SEMANTIC_PLAN_FIELDS = _COMMON_PLAN_FIELDS | frozenset(
    {
        "additive_composition_receipt_sha256",
        "allowed_handle_ids",
        "classified_closure",
        "fitted_prompt_receipt_sha256",
        "handle_group_by_id",
        "messages",
        "messages_sha256",
        "preservation_requirements",
        "prompt_token_proxy",
        "provider_input",
        "story_coherence",
        "terminal_prompt_receipt_sha256",
        "validation_contract",
    }
)

_PREFLIGHT_FIELDS = frozenset(
    {
        "answer_plan_population_sha256",
        "construction_artifact_sha256",
        "construction_format",
        "format",
        "gateway_url",
        "gold_loaded",
        "hard_complete_chat_token_cap",
        "max_chat_prompt_tokens",
        "max_concurrency",
        "model",
        "observed_max_complete_envelope_tokens",
        "ordinals",
        "output_token_reserve",
        "parent_passthrough_count",
        "parent_passthrough_rows",
        "physical_prompt_rows",
        "prompt_population",
        "prompt_population_sha256",
        "provider_calls",
        "question_count",
        "required_authorized_provider_calls",
        "retained_transformer_token_state_bytes",
        "semantic_question_count",
        "semantic_renderer_format",
    }
)

ConstructionLoader = Callable[..., tuple[SealedArtifact, Sequence[Mapping[str, Any]]]]


class LockedSemanticFinalAnswerV3Error(MatchedEvalContractError):
    """A construction, visible closure, prompt, checkpoint, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSemanticFinalAnswerV3Error(message)


def _plain_messages(
    messages: Sequence[Mapping[str, str]],
) -> tuple[dict[str, str], ...]:
    rows = tuple(dict(row) for row in messages)
    _require(
        bool(rows)
        and all(
            set(row) == {"role", "content"}
            and row["role"] in {"system", "user", "assistant"}
            and type(row["content"]) is str
            for row in rows
        ),
        "semantic v3 messages changed schema",
    )
    return rows


def _default_construction_loader(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[SealedArtifact, Sequence[Mapping[str, Any]]]:
    module = importlib.import_module("tools.run_reduced_semantic_binary_search_assay")
    loader = getattr(module, "load_verified_construction", None)
    _require(callable(loader), "semantic construction v3 loader is unavailable")
    return loader(path, expected_sha256=expected_sha256)


def _identity_projection(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} changed type")
    assert type(value) is dict
    body = dict(value)
    declared = require_sha256(body.pop("receipt_sha256", None), label)
    _require(identity_sha256(body) == declared, f"{label} receipt changed")
    return dict(value)


def _stored_projection(
    value: object,
    *,
    label: str,
    expected_format: str,
) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} changed type")
    assert type(value) is dict
    body = dict(value)
    declared = require_sha256(
        body.pop("stored_projection_receipt_sha256", None), label
    )
    _require(
        body.get("format") == expected_format and identity_sha256(body) == declared,
        f"{label} compact receipt changed",
    )
    require_sha256(body.get("receipt_sha256"), f"{label} canonical receipt")
    return dict(value)


def _common_source_plan(raw: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    parent = specialist_v1._verified_parent(raw, ordinal)  # noqa: SLF001
    judge = parent["parent_judge_row"]
    assert type(judge) is dict
    return {
        "construction_question_receipt_sha256": require_sha256(
            raw.get("question_receipt_sha256"), "construction question receipt"
        ),
        "dated_question_sha256": require_sha256(
            raw.get("dated_question_sha256"), "dated question"
        ),
        "fallback_reason": require_text(
            raw.get("fallback_reason"), "semantic fallback reason"
        ),
        "mode": require_text(raw.get("mode"), "semantic answer mode"),
        "namespace_id": require_text(raw.get("namespace_id"), "semantic namespace"),
        "ordinal": ordinal,
        "parent_judge_row_sha256": require_sha256(
            parent.get("parent_judge_row_sha256"), "parent judge row"
        ),
        "parent_prediction": require_text(parent.get("prediction"), "parent prediction"),
        "parent_prediction_sha256": require_sha256(
            parent.get("prediction_sha256"), "parent prediction"
        ),
        "parent_prediction_source": require_text(
            judge.get("prediction_source"), "parent prediction source"
        ),
        "parent_replay_artifact_sha256": require_sha256(
            parent.get("replay_artifact_sha256"), "parent replay artifact"
        ),
        "parent_run_artifact_sha256": require_sha256(
            parent.get("run_artifact_sha256"), "parent run artifact"
        ),
        "parent_source_receipt_sha256": require_sha256(
            parent.get("receipt_sha256"), "parent source receipt"
        ),
        "parent_source_row_sha256": require_sha256(
            parent.get("source_row_sha256"), "parent source row"
        ),
        "query_vector_artifact_sha256": require_sha256(
            raw.get("query_vector_artifact_sha256"), "semantic query vectors"
        ),
        "query_vector_row_receipt_sha256": require_sha256(
            raw.get("query_vector_row_receipt_sha256"), "semantic query-vector row"
        ),
        "question_id": require_text(raw.get("question_id"), "question ID"),
        "question_sha256": require_sha256(raw.get("question_sha256"), "question"),
        "route_id": require_text(judge.get("route_id"), "parent route"),
    }


def _semantic_receipts(raw: Mapping[str, Any]) -> dict[str, Any]:
    query = _identity_projection(raw.get("semantic_query"), "semantic v3 query")
    search = _stored_projection(
        raw.get("semantic_residual_search"),
        label="semantic v3 stored search",
        expected_format=STORED_SEARCH_FORMAT,
    )
    local = _identity_projection(
        raw.get("semantic_residual_local_audit"), "semantic v3 local audit"
    )
    frontier = _identity_projection(
        search.get("classified_frontier"), "semantic v3 classified frontier"
    )
    _require(
        local.get("format") == LOCAL_AUDIT_FORMAT
        and query.get("receipt_sha256") == search.get("query_receipt_sha256")
        and query.get("residual_index_receipt_sha256")
        == search.get("residual_index_receipt_sha256")
        == raw.get("semantic_residual_index_receipt_sha256")
        and query.get("query_vector_artifact_sha256")
        == search.get("query_vector_artifact_sha256")
        == raw.get("query_vector_artifact_sha256")
        and local.get("compact_result_receipt_sha256") == search.get("receipt_sha256")
        and local.get("query") == query
        and local.get("classified_frontier") == frontier
        and local.get("protected_duplicates") == search.get("protected_duplicates")
        and search.get("gold_loaded") is False
        and search.get("new_provider_calls") == 0
        and search.get("retained_transformer_token_state_bytes") == 0
        and search.get("searched_complete_memory_population") is True
        and search.get("dedup_after_semantic_selection") is True
        and search.get("protected_evidence_mutated") is False,
        "semantic v3 query/search/local seam changed",
    )
    return {
        "audit_sha256": require_sha256(local.get("receipt_sha256"), "semantic local audit"),
        "frontier": frontier,
        "query_receipt_sha256": require_sha256(
            query.get("receipt_sha256"), "semantic query"
        ),
        "search": search,
        "search_receipt_sha256": require_sha256(
            search.get("receipt_sha256"), "semantic canonical search"
        ),
    }


def _semantic_evidence_handles(
    provider_input: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, str]]:
    typed = provider_input.get("typed_evidence")
    _require(type(typed) is dict, "semantic v3 typed evidence is missing")
    assert type(typed) is dict
    handles = typed.get("handles")
    items = typed.get("items")
    _require(
        type(handles) is list
        and bool(handles)
        and type(items) is list
        and bool(items),
        "semantic v3 provider evidence is empty",
    )
    ordered: list[str] = []
    groups: dict[str, str] = {}
    for raw in handles:
        _require(
            type(raw) is dict
            and set(raw)
            == {"group_handle", "handle_id", "origin", "provenance_grade"},
            "semantic v3 handle schema changed",
        )
        assert type(raw) is dict
        handle = require_text(raw.get("handle_id"), "semantic handle")
        group = require_text(raw.get("group_handle"), "semantic handle group")
        _require(handle not in groups, "semantic v3 handles repeat")
        ordered.append(handle)
        groups[handle] = group
    represented: set[str] = set()
    for raw in items:
        _require(type(raw) is dict, "semantic v3 provider item changed type")
        assert type(raw) is dict
        cited = raw.get("handle_ids")
        summary = raw.get("summary")
        _require(
            type(cited) is list
            and bool(cited)
            and len(cited) == len(set(cited))
            and all(type(value) is str and value in groups for value in cited)
            and type(summary) is str
            and bool(summary),
            "semantic v3 provider item escaped its handles or exact text",
        )
        represented.update(cited)
    frontier = typed.get("frontier")
    _require(
        represented == set(ordered)
        and type(frontier) is dict
        and frontier.get("available_handle_ids") == ordered
        and frontier.get("represented_handle_ids") == ordered
        and type(frontier.get("closed")) is bool
        and type(frontier.get("truncated")) is bool
        and frontier.get("closed") is not frontier.get("truncated")
        and frontier.get("mode")
        == ("exhaustive" if frontier.get("closed") else "bounded")
        and frontier.get("omitted_handle_ids") == []
        and frontier.get("unresolved_slot_ids") == [],
        "semantic v3 provider frontier changed handle coverage",
    )
    return tuple(ordered), groups


def _parser_self_check(
    *,
    parent: str,
    allowed: Sequence[str],
    groups: Mapping[str, str],
    story: Mapping[str, Any],
    preservation: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    completion = json.dumps(
        {
            "decision": "keep_parent",
            "prediction": parent,
            "used_handle_ids": [],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        parsed = typed_final.parse_typed_final_completion(
            completion,
            parent_prediction=parent,
            allowed_handle_ids=tuple(allowed),
            handle_group_by_id=dict(groups),
            story_coherence=dict(story),
            preservation_requirements=dict(preservation),
            validation_contract=dict(validation),
        )
    except MatchedEvalContractError as exc:
        raise LockedSemanticFinalAnswerV3Error(
            f"semantic v3 parser contract changed: {exc}"
        ) from exc
    _require(
        parsed.valid
        and parsed.decision == "keep_parent"
        and parsed.prediction == parent,
        "semantic v3 keep-parent parser self-check failed",
    )


def _provider_summary_rows(
    provider_input: Mapping[str, Any],
) -> tuple[tuple[tuple[str, ...], str], ...]:
    typed = provider_input.get("typed_evidence")
    _require(type(typed) is dict, "semantic v3 typed evidence disappeared")
    assert type(typed) is dict
    items = typed.get("items")
    _require(type(items) is list, "semantic v3 provider items changed type")
    result: list[tuple[tuple[str, ...], str]] = []
    for raw in items:
        _require(type(raw) is dict, "semantic v3 provider item changed type")
        assert type(raw) is dict
        handles = raw.get("handle_ids")
        summary = raw.get("summary")
        _require(
            type(handles) is list
            and bool(handles)
            and all(type(value) is str and value for value in handles)
            and type(summary) is str
            and bool(summary),
            "semantic v3 provider item lost handles or text",
        )
        result.append((tuple(handles), quote_sha256(summary)))
    return tuple(result)


def _composition_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    composition = _identity_projection(
        raw.get("additive_composition"), "semantic v3 additive composition"
    )
    local = raw.get("additive_composition_local_audit")
    _require(type(local) is dict, "semantic v3 additive local audit changed type")
    assert type(local) is dict
    dedup = _identity_projection(
        local.get("post_selection_dedup"), "semantic v3 post-selection dedup"
    )
    exclusions = dedup.get("exclusions")
    _require(
        type(exclusions) is list
        and composition.get("post_selection_dedup_audit_receipt_sha256")
        == dedup.get("receipt_sha256")
        and composition.get("gold_loaded") is False
        and composition.get("provider_prompt_count") == 0
        and composition.get("retained_transformer_token_state_bytes") == 0,
        "semantic v3 composition/dedup seam changed",
    )
    exclusion_by_sha256: dict[str, dict[str, Any]] = {}
    for value in exclusions:
        _require(type(value) is dict, "semantic v3 dedup exclusion changed type")
        assert type(value) is dict
        receipt = identity_sha256(value)
        _require(receipt not in exclusion_by_sha256, "semantic v3 dedup repeats")
        exclusion_by_sha256[receipt] = dict(value)
    return {
        "composition_receipt_sha256": require_sha256(
            composition.get("receipt_sha256"), "semantic additive composition"
        ),
        "dedup": dedup,
        "dedup_exclusion_by_sha256": exclusion_by_sha256,
    }


def _classified_closure(
    raw: Mapping[str, Any],
    *,
    allowed: Sequence[str],
    provider_input: Mapping[str, Any],
    fitted: Mapping[str, Any],
    fitted_receipt_sha256: str,
    residual: Mapping[str, Any],
    composition: Mapping[str, Any],
) -> dict[str, Any]:
    closure = _identity_projection(
        raw.get("classified_closure"), "semantic v3 classified closure"
    )
    _require(
        set(closure) == _CLOSURE_FIELDS,
        "semantic v3 classified closure schema changed",
    )
    frontier = residual["frontier"]
    search = residual["search"]
    retained = tuple(frontier.get("retained_segment_receipt_sha256s", ()))
    rows = closure.get("rows")
    _require(
        type(rows) is list
        and bool(rows)
        and closure.get("format") == CLASSIFIED_CLOSURE_FORMAT
        and closure.get("semantic_residual_search_receipt_sha256")
        == residual["search_receipt_sha256"]
        and closure.get("classified_frontier_receipt_sha256")
        == frontier.get("receipt_sha256")
        and closure.get("post_selection_dedup_audit_receipt_sha256")
        == composition["dedup"].get("receipt_sha256")
        and closure.get("fitted_prompt_receipt_sha256")
        == fitted_receipt_sha256
        and closure.get("complete_leaf_partition") is True
        and closure.get("closed") is True
        and closure.get("all_retained_segments_provider_visible") is True
        and tuple(closure.get("retained_segment_receipt_sha256s", ())) == retained
        and tuple(
            row.get("segment_receipt_sha256")
            for row in rows
            if type(row) is dict
        )
        == retained
        and tuple(closure.get("terminal_allowed_handle_ids", ()))
        == tuple(allowed),
        "semantic v3 classified closure boundary changed",
    )
    allowed_body = {
        "format": f"{CLASSIFIED_CLOSURE_FORMAT}-terminal-allowed-handles-v1",
        "terminal_allowed_handle_ids": list(allowed),
    }
    protection_body = {
        "classified_frontier_receipt_sha256": frontier.get("receipt_sha256"),
        "format": f"{CLASSIFIED_CLOSURE_FORMAT}-protection-source-v1",
        "post_selection_dedup_audit_receipt_sha256": composition["dedup"].get(
            "receipt_sha256"
        ),
        "retained_segment_receipt_sha256s": list(retained),
        "rows": [dict(row) for row in rows if type(row) is dict],
        "semantic_residual_search_receipt_sha256": residual[
            "search_receipt_sha256"
        ],
    }
    _require(
        closure.get("terminal_allowed_handle_ids_sha256")
        == identity_sha256(allowed_body)
        and closure.get("protection_source_receipt_sha256")
        == identity_sha256(protection_body)
        == fitted.get("protection_source_receipt_sha256"),
        "semantic v3 closure protection receipt changed",
    )

    provider_rows = _provider_summary_rows(provider_input)
    evidence_rows = search.get("evidence")
    duplicate_rows = search.get("protected_duplicates")
    _require(
        type(evidence_rows) is list and type(duplicate_rows) is list,
        "semantic v3 canonical survivor inventories changed type",
    )
    evidence_by_segment: dict[str, dict[str, Any]] = {}
    duplicate_by_segment: dict[str, dict[str, Any]] = {}
    for value in evidence_rows:
        evidence = _identity_projection(value, "semantic v3 exact evidence")
        segment = require_sha256(
            evidence.get("segment_receipt_sha256"), "semantic evidence segment"
        )
        _require(segment not in evidence_by_segment, "semantic evidence repeats segment")
        evidence_by_segment[segment] = evidence
    for value in duplicate_rows:
        duplicate = _identity_projection(value, "semantic v3 protected duplicate")
        segment = require_sha256(
            duplicate.get("segment_receipt_sha256"), "semantic duplicate segment"
        )
        _require(
            segment not in duplicate_by_segment,
            "semantic protected duplicate repeats segment",
        )
        duplicate_by_segment[segment] = duplicate

    protected_items: list[str] = []
    protected_bindings: list[str] = []
    for value in rows:
        _require(
            type(value) is dict and set(value) == _CLOSURE_ROW_FIELDS,
            "semantic v3 closure row schema changed",
        )
        assert type(value) is dict
        segment = require_sha256(value.get("segment_receipt_sha256"), "closure segment")
        exact_text = require_sha256(value.get("exact_text_sha256"), "closure exact text")
        visible_handles = tuple(value.get("visible_handle_ids", ()))
        visible_bindings = tuple(value.get("visible_binding_receipt_sha256s", ()))
        visible_item = require_sha256(
            value.get("visible_item_receipt_sha256"), "closure visible item"
        )
        _require(
            bool(visible_handles)
            and len(visible_handles) == len(set(visible_handles))
            and set(visible_handles) <= set(allowed)
            and bool(visible_bindings)
            and all(
                type(receipt) is str and len(receipt) == 64
                for receipt in visible_bindings
            )
            and provider_rows.count((visible_handles, exact_text)) == 1,
            "semantic v3 retained segment is not byte-identically provider-visible",
        )
        evidence = evidence_by_segment.get(segment)
        duplicate = duplicate_by_segment.get(segment)
        _require(
            (evidence is None) != (duplicate is None),
            "semantic v3 closure segment has no unique canonical owner",
        )
        disposition = value.get("disposition")
        if evidence is not None:
            quote = require_text(evidence.get("quote"), "semantic retained quote")
            require_sha256(
                value.get("residual_binding_receipt_sha256"),
                "semantic residual typed binding",
            )
            _require(
                value.get("residual_evidence_receipt_sha256")
                == evidence.get("receipt_sha256")
                and value.get("cell_id") == evidence.get("cell_id")
                and exact_text == quote_sha256(quote) == evidence.get("quote_sha256"),
                "semantic v3 closure changed canonical residual evidence",
            )
            if disposition == "residual_visible":
                _require(
                    value.get("dedup_exclusion_sha256") is None
                    and visible_bindings
                    == (value.get("residual_binding_receipt_sha256"),)
                    and visible_item == value.get("residual_item_receipt_sha256"),
                    "semantic v3 direct survivor visibility changed",
                )
            else:
                exclusion_sha256 = require_sha256(
                    value.get("dedup_exclusion_sha256"),
                    "semantic composition dedup exclusion",
                )
                exclusion = composition["dedup_exclusion_by_sha256"].get(
                    exclusion_sha256
                )
                _require(
                    disposition == "protected_visible_exact_duplicate"
                    and type(exclusion) is dict
                    and exclusion.get("duplicate_item_receipt_sha256")
                    == value.get("residual_item_receipt_sha256")
                    and exclusion.get("duplicate_binding_receipt_sha256s")
                    == [value.get("residual_binding_receipt_sha256")]
                    and exclusion.get("owner_item_receipt_sha256") == visible_item
                    and exclusion.get("owner_binding_receipt_sha256s")
                    == list(visible_bindings),
                    "semantic v3 post-selection duplicate owner changed",
                )
        else:
            assert duplicate is not None
            _require(
                disposition == "protected_visible_exact_duplicate"
                and value.get("residual_evidence_receipt_sha256")
                == duplicate.get("receipt_sha256")
                and value.get("residual_binding_receipt_sha256")
                == duplicate.get("protected_binding_receipt_sha256")
                and value.get("cell_id") == duplicate.get("cell_id")
                and value.get("dedup_exclusion_sha256")
                == duplicate.get("receipt_sha256"),
                "semantic v3 search-level protected duplicate owner changed",
            )
        if visible_item not in protected_items:
            protected_items.append(visible_item)
        for receipt in visible_bindings:
            require_sha256(receipt, "semantic visible binding")
            if receipt not in protected_bindings:
                protected_bindings.append(receipt)
    _require(
        tuple(protected_items)
        == tuple(fitted.get("protected_item_receipt_sha256s", ()))
        and set(protected_bindings)
        == set(fitted.get("protected_binding_receipt_sha256s", ())),
        "semantic v3 closure/fitter protection inventory changed",
    )
    return closure


def _terminal_prompt(
    terminal: object,
    *,
    fitted: Mapping[str, Any],
    provider_input: Mapping[str, Any],
    fitted_receipt_sha256: str,
) -> tuple[list[dict[str, str]], int, str]:
    _require(
        type(terminal) is dict and set(terminal) == _TERMINAL_FIELDS,
        "semantic v3 terminal prompt schema changed",
    )
    assert type(terminal) is dict
    messages = list(_plain_messages(typed_final.render_final_messages(provider_input)))
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    messages_sha256 = identity_sha256(messages)
    rendered = json.dumps(
        messages,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    terminal_body = {
        "fitted_prompt_receipt_sha256": fitted_receipt_sha256,
        "messages_sha256": messages_sha256,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "prompt_token_proxy": prompt_tokens,
        "provider_input_sha256": identity_sha256(provider_input),
        "rendered_messages_utf8_byte_count": len(rendered),
        "rendered_messages_utf8_sha256": hashlib.sha256(rendered).hexdigest(),
    }
    terminal_receipt = require_sha256(
        terminal.get("terminal_prompt_receipt_sha256"), "semantic terminal prompt"
    )
    _require(
        terminal.get("provider_input") == provider_input
        and terminal.get("fitted_prompt_receipt_sha256") == fitted_receipt_sha256
        and terminal.get("messages") == messages
        and terminal.get("messages_sha256") == messages_sha256
        and terminal.get("prompt_token_proxy") == prompt_tokens
        and terminal.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and terminal.get("full_chat_plus_output_tokens")
        == prompt_tokens + OUTPUT_TOKEN_RESERVE
        and terminal.get("hard_prompt_token_cap") == HARD_COMPLETE_CHAT_TOKEN_CAP
        and terminal.get("provider_prompt_count") == 0
        and terminal.get("retained_transformer_token_state_bytes") == 0
        and terminal.get("rendered_messages_utf8_byte_count") == len(rendered)
        and terminal.get("rendered_messages_utf8_sha256")
        == hashlib.sha256(rendered).hexdigest()
        and terminal_receipt == identity_sha256(terminal_body)
        and fitted.get("messages_sha256") == messages_sha256
        and fitted.get("prompt_token_proxy") == prompt_tokens
        and fitted.get("full_chat_plus_output_tokens")
        == prompt_tokens + OUTPUT_TOKEN_RESERVE
        and fitted.get("hard_prompt_token_cap") == HARD_COMPLETE_CHAT_TOKEN_CAP
        and fitted.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and fitted.get("retained_transformer_token_state_bytes") == 0
        and prompt_tokens <= MAX_CHAT_PROMPT_TOKENS,
        "semantic v3 terminal prompt bytes or 8k budget changed",
    )
    return messages, prompt_tokens, terminal_receipt


def _semantic_source_plan(
    raw: Mapping[str, Any],
    ordinal: int,
    *,
    common: Mapping[str, Any],
    residual: Mapping[str, Any],
) -> dict[str, Any]:
    fitted = raw.get("fitted_typed_prompt")
    terminal = raw.get("terminal_prompt")
    _require(
        raw.get("fallback_reason") == "none"
        and type(fitted) is dict
        and set(fitted) == _FITTED_FIELDS
        and type(terminal) is dict,
        f"semantic v3 success source schema changed at ordinal {ordinal}",
    )
    assert type(fitted) is dict and type(terminal) is dict
    provider_input = fitted.get("provider_input")
    allowed = fitted.get("allowed_handle_ids")
    groups = fitted.get("handle_group_by_id")
    story = fitted.get("story_coherence")
    preservation = fitted.get("preservation_requirements")
    validation = fitted.get("validation_contract")
    _require(
        type(provider_input) is dict
        and set(provider_input) == _PROVIDER_FIELDS
        and provider_input.get("format") == typed_final.PROMPT_ROW_FORMAT
        and type(allowed) is list
        and bool(allowed)
        and len(allowed) == len(set(allowed))
        and type(groups) is dict
        and type(story) is dict
        and type(preservation) is dict
        and type(validation) is dict,
        f"semantic v3 fitted provider seam changed at ordinal {ordinal}",
    )
    assert (
        type(provider_input) is dict
        and type(allowed) is list
        and type(groups) is dict
        and type(story) is dict
        and type(preservation) is dict
        and type(validation) is dict
    )
    handles, provider_groups = _semantic_evidence_handles(provider_input)
    parent_fallback = provider_input.get("protected_parent_fallback")
    dated_question = require_text(
        provider_input.get("dated_question"), "semantic dated question"
    )
    frontier = residual["frontier"]
    search = residual["search"]
    _require(
        len(allowed) == len(handles)
        and set(allowed) == set(handles)
        and groups == provider_groups
        and provider_input.get("story_coherence") == story
        and type(parent_fallback) is dict
        and parent_fallback.get("prediction") == common["parent_prediction"]
        and parent_fallback.get("prediction_sha256")
        == common["parent_prediction_sha256"]
        and set(validation.get("by_handle", {})) == set(allowed)
        and quote_sha256(dated_question) == common["dated_question_sha256"]
        and search.get("fallback_required") is False
        and search.get("fallback_reason") == "none"
        and frontier.get("closed") is True
        and frontier.get("complete_leaf_partition") is True
        and not frontier.get("unresolved_segment_receipt_sha256s")
        and bool(frontier.get("retained_segment_receipt_sha256s")),
        f"semantic v3 success lost parent, scope, or closure at ordinal {ordinal}",
    )
    assert_gold_blind(provider_input, path=f"semantic_v3_provider_{ordinal}")
    fitted_receipt = require_sha256(
        fitted.get("receipt_sha256"), "semantic fitted prompt"
    )
    composition = _composition_contract(raw)
    closure = _classified_closure(
        raw,
        allowed=allowed,
        provider_input=provider_input,
        fitted=fitted,
        fitted_receipt_sha256=fitted_receipt,
        residual=residual,
        composition=composition,
    )
    messages, prompt_tokens, terminal_receipt = _terminal_prompt(
        terminal,
        fitted=fitted,
        provider_input=provider_input,
        fitted_receipt_sha256=fitted_receipt,
    )
    _parser_self_check(
        parent=str(common["parent_prediction"]),
        allowed=allowed,
        groups=groups,
        story=story,
        preservation=preservation,
        validation=validation,
    )
    body = {
        **common,
        "additive_composition_receipt_sha256": composition[
            "composition_receipt_sha256"
        ],
        "allowed_handle_ids": list(allowed),
        "classified_closure": closure,
        "fitted_prompt_receipt_sha256": fitted_receipt,
        "handle_group_by_id": dict(groups),
        "messages": messages,
        "messages_sha256": identity_sha256(messages),
        "preservation_requirements": dict(preservation),
        "prompt_token_proxy": prompt_tokens,
        "provider_input": dict(provider_input),
        "story_coherence": dict(story),
        "terminal_prompt_receipt_sha256": terminal_receipt,
        "validation_contract": dict(validation),
    }
    result = {**body, "answer_plan_receipt_sha256": identity_sha256(body)}
    return _validate_plan(result)


def _fallback_source_plan(
    raw: Mapping[str, Any],
    ordinal: int,
    *,
    common: Mapping[str, Any],
    residual: Mapping[str, Any],
) -> dict[str, Any]:
    reason = raw.get("fallback_reason")
    search = residual["search"]
    frontier = residual["frontier"]
    _require(
        reason
        in {
            "retained_unknowns_exceed_payload_cap",
            "no_novel_semantic_evidence",
            "protected_semantic_residual_exceeds_terminal_cap",
        }
        and all(
            raw.get(key) is None
            for key in (
                "additive_composition",
                "additive_composition_local_audit",
                "classified_closure",
                "fitted_typed_prompt",
                "terminal_prompt",
            )
        )
        and (
            (
                reason == "retained_unknowns_exceed_payload_cap"
                and search.get("fallback_required") is True
                and search.get("fallback_reason") == reason
                and frontier.get("closed") is False
                and not search.get("evidence")
            )
            or (
                reason == "no_novel_semantic_evidence"
                and search.get("fallback_required") is False
                and search.get("fallback_reason") == "none"
                and frontier.get("closed") is True
                and not search.get("evidence")
            )
            or (
                reason == "protected_semantic_residual_exceeds_terminal_cap"
                and search.get("fallback_required") is False
                and search.get("fallback_reason") == "none"
                and frontier.get("closed") is True
                and bool(search.get("evidence"))
            )
        ),
        f"semantic v3 fallback proof changed at ordinal {ordinal}",
    )
    body = dict(common)
    result = {**body, "answer_plan_receipt_sha256": identity_sha256(body)}
    return _validate_plan(result)


def _source_plan(raw: Mapping[str, Any], expected_ordinal: int) -> dict[str, Any]:
    _require(
        set(raw) == _SOURCE_FIELDS
        and raw.get("ordinal") == expected_ordinal
        and raw.get("mode") in _MODES
        and raw.get("new_provider_calls") == 0
        and raw.get("retained_transformer_token_state_bytes") == 0,
        f"semantic v3 source schema changed at ordinal {expected_ordinal}",
    )
    source_body = dict(raw)
    declared = require_sha256(
        source_body.pop("question_receipt_sha256", None),
        "semantic construction question",
    )
    _require(
        declared == identity_sha256(source_body),
        f"semantic v3 construction row seal changed at ordinal {expected_ordinal}",
    )
    common = _common_source_plan(raw, expected_ordinal)
    residual = _semantic_receipts(raw)
    common.update(
        {
            "semantic_query_receipt_sha256": residual[
                "query_receipt_sha256"
            ],
            "semantic_residual_index_receipt_sha256": require_sha256(
                raw.get("semantic_residual_index_receipt_sha256"),
                "semantic residual index",
            ),
            "semantic_residual_local_audit_sha256": residual["audit_sha256"],
            "semantic_residual_search_receipt_sha256": residual[
                "search_receipt_sha256"
            ],
        }
    )
    if raw.get("mode") == SEMANTIC_MODE:
        return _semantic_source_plan(
            raw,
            expected_ordinal,
            common=common,
            residual=residual,
        )
    return _fallback_source_plan(
        raw,
        expected_ordinal,
        common=common,
        residual=residual,
    )


def _validate_plan(raw: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = (
        _SEMANTIC_PLAN_FIELDS
        if raw.get("mode") == SEMANTIC_MODE
        else _COMMON_PLAN_FIELDS
    )
    _require(
        set(raw) == expected_fields and raw.get("mode") in _MODES,
        f"sealed semantic v3 plan schema changed at ordinal {raw.get('ordinal')}",
    )
    body = dict(raw)
    declared = require_sha256(
        body.pop("answer_plan_receipt_sha256", None), "semantic v3 answer plan"
    )
    ordinal = raw.get("ordinal")
    parent = require_text(raw.get("parent_prediction"), "semantic v3 parent")
    _require(
        type(ordinal) is int
        and ordinal in QUESTION_ORDINALS
        and declared == identity_sha256(body)
        and raw.get("parent_prediction_sha256") == quote_sha256(parent),
        f"sealed semantic v3 plan receipt changed at ordinal {ordinal}",
    )
    if raw.get("mode") == SEMANTIC_MODE:
        _require(
            raw.get("fallback_reason") == "none",
            f"semantic v3 physical plan gained fallback at ordinal {ordinal}",
        )
        provider = raw.get("provider_input")
        allowed = raw.get("allowed_handle_ids")
        groups = raw.get("handle_group_by_id")
        story = raw.get("story_coherence")
        preservation = raw.get("preservation_requirements")
        validation = raw.get("validation_contract")
        closure = raw.get("classified_closure")
        _require(
            type(provider) is dict
            and set(provider) == _PROVIDER_FIELDS
            and type(allowed) is list
            and type(groups) is dict
            and type(story) is dict
            and type(preservation) is dict
            and type(validation) is dict
            and type(closure) is dict
            and set(closure) == _CLOSURE_FIELDS,
            f"sealed semantic v3 provider plan changed at ordinal {ordinal}",
        )
        assert (
            type(provider) is dict
            and type(allowed) is list
            and type(groups) is dict
            and type(story) is dict
            and type(preservation) is dict
            and type(validation) is dict
            and type(closure) is dict
        )
        handles, provider_groups = _semantic_evidence_handles(provider)
        closure_rows = closure.get("rows")
        provider_rows = _provider_summary_rows(provider)
        closure_body = dict(closure)
        closure_receipt = require_sha256(
            closure_body.pop("receipt_sha256", None),
            "sealed semantic v3 closure",
        )
        _require(
            len(allowed) == len(handles)
            and set(allowed) == set(handles)
            and groups == provider_groups
            and closure_receipt == identity_sha256(closure_body)
            and closure.get("format") == CLASSIFIED_CLOSURE_FORMAT
            and closure.get("closed") is True
            and closure.get("complete_leaf_partition") is True
            and closure.get("all_retained_segments_provider_visible") is True
            and closure.get("terminal_allowed_handle_ids") == allowed
            and closure.get("semantic_residual_search_receipt_sha256")
            == raw.get("semantic_residual_search_receipt_sha256")
            and closure.get("fitted_prompt_receipt_sha256")
            == raw.get("fitted_prompt_receipt_sha256")
            and type(closure_rows) is list
            and all(
                type(row) is dict
                and set(row) == _CLOSURE_ROW_FIELDS
                and (
                    tuple(row.get("visible_handle_ids", ())),
                    row.get("exact_text_sha256"),
                )
                in provider_rows
                for row in closure_rows
            ),
            f"sealed semantic v3 closure lost provider visibility at ordinal {ordinal}",
        )
        messages = list(_plain_messages(typed_final.render_final_messages(provider)))
        _require(
            raw.get("messages") == messages
            and raw.get("messages_sha256") == identity_sha256(messages)
            and raw.get("prompt_token_proxy")
            == count_chat_prompt_token_proxy(messages)
            and raw["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE
            <= HARD_COMPLETE_CHAT_TOKEN_CAP,
            f"sealed semantic v3 prompt changed at ordinal {ordinal}",
        )
        _parser_self_check(
            parent=parent,
            allowed=allowed,
            groups=groups,
            story=story,
            preservation=preservation,
            validation=validation,
        )
    else:
        _require(
            raw.get("fallback_reason")
            in {
                "retained_unknowns_exceed_payload_cap",
                "no_novel_semantic_evidence",
                "protected_semantic_residual_exceeds_terminal_cap",
            },
            f"sealed semantic v3 fallback reason changed at ordinal {ordinal}",
        )
    assert_gold_blind(raw, path=f"loaded_semantic_v3_plan_{ordinal}")
    return dict(raw)


def load_answer_plans(
    path: str | Path,
    expected_sha256: str,
    *,
    construction_loader: ConstructionLoader | None = None,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    """Load a construction-v3 through its public validator and seal answer plans."""

    expected = require_sha256(expected_sha256, "expected semantic construction v3")
    loader = construction_loader or _default_construction_loader
    artifact, raw_rows = loader(Path(path), expected_sha256=expected)
    _require(
        isinstance(artifact, SealedArtifact)
        and artifact.sha256 == expected
        and artifact.payload.get("format") == CONSTRUCTION_FORMAT
        and isinstance(raw_rows, (tuple, list))
        and len(raw_rows) == QUESTION_COUNT,
        "semantic construction v3 digest or population changed",
    )
    plans = tuple(
        _source_plan(raw, ordinal)
        for ordinal, raw in zip(QUESTION_ORDINALS, raw_rows, strict=True)
        if isinstance(raw, Mapping)
    )
    _require(
        len(plans) == QUESTION_COUNT
        and tuple(row["ordinal"] for row in plans) == QUESTION_ORDINALS
        and len({row["question_id"] for row in plans}) == QUESTION_COUNT,
        "semantic construction v3 identities changed",
    )
    return artifact, plans


def _preflight_projection(
    construction: SealedArtifact,
    plans: tuple[dict[str, Any], ...],
    *,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> dict[str, Any]:
    _require(model == DEFAULT_MODEL, "semantic v3 answer model must be Terra")
    require_text(gateway_url, "semantic v3 Terra gateway")
    _require(
        type(max_concurrency) is int and max_concurrency > 0,
        "semantic v3 concurrency must be positive",
    )
    validated = tuple(_validate_plan(row) for row in plans)
    _require(
        len(validated) == QUESTION_COUNT
        and tuple(row["ordinal"] for row in validated) == QUESTION_ORDINALS,
        "semantic v3 answer-plan population changed",
    )
    physical = tuple(row for row in validated if row["mode"] == SEMANTIC_MODE)
    passthrough = tuple(
        row for row in validated if row["mode"] == PARENT_PASSTHROUGH_MODE
    )
    _require(bool(physical), "semantic v3 provider population is empty")
    prompts = tuple(_plain_messages(row["messages"]) for row in physical)
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == len(physical),
        "semantic v3 physical prompts must be unique",
    )
    observed_max = max(
        row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE for row in physical
    )
    _require(
        observed_max <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "semantic v3 complete prompt envelope exceeds 8k",
    )
    payload = {
        "answer_plan_population_sha256": identity_sha256(
            [row["answer_plan_receipt_sha256"] for row in validated]
        ),
        "construction_artifact_sha256": construction.sha256,
        "construction_format": CONSTRUCTION_FORMAT,
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "max_chat_prompt_tokens": MAX_CHAT_PROMPT_TOKENS,
        "max_concurrency": max_concurrency,
        "model": model,
        "observed_max_complete_envelope_tokens": observed_max,
        "ordinals": list(QUESTION_ORDINALS),
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "parent_passthrough_count": len(passthrough),
        "parent_passthrough_rows": list(passthrough),
        "physical_prompt_rows": list(physical),
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "provider_calls": 0,
        "question_count": QUESTION_COUNT,
        "required_authorized_provider_calls": len(physical),
        "retained_transformer_token_state_bytes": 0,
        "semantic_question_count": len(physical),
        "semantic_renderer_format": SEMANTIC_PROMPT_FORMAT,
    }
    assert_gold_blind(payload, path="semantic_v3_answer_preflight")
    return payload


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    construction, plans = load_answer_plans(
        Path(args.construction), str(args.expected_construction_sha256)
    )
    payload = _preflight_projection(
        construction,
        plans,
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
    )
    artifact, created = publish_sealed_json(
        Path(args.output_root) / PREFLIGHT_NAME, payload
    )
    return {
        "construction_sha256": construction.sha256,
        "created": created,
        "maximum_complete_prompt_envelope": payload[
            "observed_max_complete_envelope_tokens"
        ],
        "parent_passthrough_count": payload["parent_passthrough_count"],
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "question_count": QUESTION_COUNT,
        "required_authorized_provider_calls": payload[
            "required_authorized_provider_calls"
        ],
        "semantic_question_count": payload["semantic_question_count"],
    }


def _validate_preflight(
    artifact: SealedArtifact,
) -> tuple[
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    payload = artifact.payload
    assert_gold_blind(payload, path="loaded_semantic_v3_answer_preflight")
    physical = payload.get("physical_prompt_rows")
    passthrough = payload.get("parent_passthrough_rows")
    _require(
        set(payload) == _PREFLIGHT_FIELDS
        and payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("construction_format") == CONSTRUCTION_FORMAT
        and payload.get("model") == DEFAULT_MODEL
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("hard_complete_chat_token_cap")
        == HARD_COMPLETE_CHAT_TOKEN_CAP
        and payload.get("max_chat_prompt_tokens") == MAX_CHAT_PROMPT_TOKENS
        and payload.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and payload.get("question_count") == QUESTION_COUNT
        and payload.get("ordinals") == list(QUESTION_ORDINALS)
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("semantic_renderer_format") == SEMANTIC_PROMPT_FORMAT
        and type(payload.get("max_concurrency")) is int
        and payload["max_concurrency"] > 0
        and type(physical) is list
        and bool(physical)
        and type(passthrough) is list
        and len(physical) == payload.get("required_authorized_provider_calls")
        == payload.get("semantic_question_count")
        and len(passthrough) == payload.get("parent_passthrough_count")
        and len(physical) + len(passthrough) == QUESTION_COUNT,
        "sealed semantic v3 preflight changed",
    )
    require_sha256(
        payload.get("construction_artifact_sha256"),
        "semantic v3 preflight construction",
    )
    validated_physical = tuple(_validate_plan(row) for row in physical)
    validated_passthrough = tuple(_validate_plan(row) for row in passthrough)
    _require(
        all(row["mode"] == SEMANTIC_MODE for row in validated_physical)
        and all(
            row["mode"] == PARENT_PASSTHROUGH_MODE
            for row in validated_passthrough
        ),
        "semantic v3 preflight modes changed",
    )
    ordered = tuple(
        sorted(
            (*validated_physical, *validated_passthrough),
            key=lambda row: QUESTION_ORDINALS.index(row["ordinal"]),
        )
    )
    _require(
        tuple(row["ordinal"] for row in ordered) == QUESTION_ORDINALS
        and len({row["question_id"] for row in ordered}) == QUESTION_COUNT
        and payload.get("answer_plan_population_sha256")
        == identity_sha256(
            [row["answer_plan_receipt_sha256"] for row in ordered]
        ),
        "sealed semantic v3 answer-plan population changed",
    )
    prompts = tuple(_plain_messages(row["messages"]) for row in validated_physical)
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    observed_max = max(
        row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE
        for row in validated_physical
    )
    _require(
        population.model_dump() == payload.get("prompt_population")
        and population.prompt_population_sha256
        == payload.get("prompt_population_sha256")
        and population.logical_prompt_count
        == population.unique_prompt_count
        == len(validated_physical)
        and payload.get("observed_max_complete_envelope_tokens") == observed_max
        and observed_max <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "sealed semantic v3 prompt population changed",
    )
    return prompts, ordered


def _read_preflight(
    output_root: Path,
    expected_sha256: str,
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    artifact = read_sealed_json(output_root / PREFLIGHT_NAME)
    _require(
        artifact.sha256
        == require_sha256(expected_sha256, "expected semantic v3 preflight"),
        "semantic v3 preflight digest changed",
    )
    prompts, plans = _validate_preflight(artifact)
    return artifact, prompts, plans


def _runtime(
    artifact: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    output_root: Path,
    model: str,
    gateway_url: str,
    max_concurrency: int,
    client: Any | None,
) -> FastCompletionRuntime:
    required = artifact.payload["required_authorized_provider_calls"]
    _require(
        model == DEFAULT_MODEL == artifact.payload.get("model")
        and gateway_url == artifact.payload.get("gateway_url")
        and max_concurrency == artifact.payload.get("max_concurrency")
        and len(prompts) == required,
        "runtime settings differ from sealed semantic v3 preflight",
    )
    return FastCompletionRuntime(
        checkpoint_dir=output_root / CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=model,
        client=client,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
        max_new_tokens=OUTPUT_TOKEN_RESERVE,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "arm": "locked_semantic_final_terra_answer_v3",
            "authorized_unique_calls": required,
            "construction_artifact_sha256": artifact.payload[
                "construction_artifact_sha256"
            ],
            "construction_format": CONSTRUCTION_FORMAT,
            "experiment_format": FORMAT,
            "gateway_url": gateway_url,
            "gold_loaded": False,
            "preflight_artifact_sha256": artifact.sha256,
            "semantic_renderer_format": SEMANTIC_PROMPT_FORMAT,
        },
    )


def _checkpoint_batch(
    artifact: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionBatch:
    runtime = _runtime(
        artifact,
        prompts,
        output_root=Path(args.output_root),
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_concurrency=int(args.max_concurrency),
        client=client,
    )
    try:
        return runtime.run()
    finally:
        runtime.close()


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, _plans = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    required = preflight.payload["required_authorized_provider_calls"]
    _require(
        args.enable_provider is True and args.authorized_provider_calls == required,
        f"provider-run requires exact authorization for {required} Terra calls",
    )
    load_dotenv()
    api_key = os.environ.get(str(args.api_key_env), "").strip()
    _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
    client = live._make_provider_client(api_key, str(args.gateway_url))  # noqa: SLF001
    try:
        batch = _checkpoint_batch(preflight, prompts, args=args, client=client)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    _require(
        batch.usage.logical_calls == batch.usage.unique_calls == required,
        "semantic v3 Terra population changed",
    )
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "gold_loaded": False,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": preflight.sha256,
        "required_authorized_provider_calls": required,
    }


def _stable_batch(batch: FastCompletionBatch) -> dict[str, Any]:
    value = batch.model_dump()
    return {
        "logical_completions": value["logical_completions"],
        "prompt_population": value["prompt_population"],
        "provenance": value["provenance"],
        "runtime_identity_sha256": value["runtime_identity_sha256"],
        "unique_records": [
            {
                key: child
                for key, child in row.items()
                if key not in {"checkpoint_hit", "physical_call"}
            }
            for row in value["unique_records"]
        ],
        "usage": {
            key: child
            for key, child in value["usage"].items()
            if key not in {"checkpoint_hits", "physical_calls"}
        },
    }


def _result_body(
    plan: Mapping[str, Any],
    *,
    prediction: str,
    prediction_source: str,
    decision: str,
    completion_parser: str,
    call_key_sha256: str | None = None,
    completion_receipt_sha256: str | None = None,
    parse_error_code: str | None = None,
    parse_receipt_sha256: str | None = None,
    prompt_row_receipt_sha256: str | None = None,
    request_journal_sha256: str | None = None,
    response_journal_sha256: str | None = None,
    solver_valid: bool | None = None,
    used_handle_ids: Sequence[str] = (),
    validation_basis: str | None = None,
) -> dict[str, Any]:
    parent = plan["parent_prediction"]
    return {
        "answer_mode": plan["mode"],
        "call_key_sha256": call_key_sha256,
        "changed_from_parent": prediction != parent,
        "completion_parser": completion_parser,
        "completion_receipt_sha256": completion_receipt_sha256,
        "construction_question_receipt_sha256": plan[
            "construction_question_receipt_sha256"
        ],
        "dated_question_sha256": plan["dated_question_sha256"],
        "decision": decision,
        "format": RESULT_ROW_FORMAT,
        "ordinal": plan["ordinal"],
        "parent_judge_row_sha256": plan["parent_judge_row_sha256"],
        "parent_prediction_sha256": plan["parent_prediction_sha256"],
        "parent_prediction_source": plan["parent_prediction_source"],
        "parent_replay_artifact_sha256": plan[
            "parent_replay_artifact_sha256"
        ],
        "parent_run_artifact_sha256": plan["parent_run_artifact_sha256"],
        "parent_source_receipt_sha256": plan["parent_source_receipt_sha256"],
        "parent_source_row_sha256": plan["parent_source_row_sha256"],
        "parse_error_code": parse_error_code,
        "parse_receipt_sha256": parse_receipt_sha256,
        "prediction": prediction,
        "prediction_sha256": quote_sha256(prediction),
        "prediction_source": prediction_source,
        "prompt_row_receipt_sha256": prompt_row_receipt_sha256,
        "question_id": plan["question_id"],
        "question_sha256": plan["question_sha256"],
        "request_journal_sha256": request_journal_sha256,
        "response_journal_sha256": response_journal_sha256,
        "retained_transformer_token_state_bytes": 0,
        "route_id": plan["route_id"],
        "solver_valid": solver_valid,
        "used_handle_ids": list(used_handle_ids),
        "validation_basis": validation_basis,
    }


def _materialization_projection(
    preflight: SealedArtifact,
    plans: tuple[dict[str, Any], ...],
    batch: FastCompletionBatch,
) -> dict[str, Any]:
    required = preflight.payload["required_authorized_provider_calls"]
    _require(
        batch.usage.logical_calls
        == batch.usage.unique_calls
        == batch.usage.checkpoint_hits
        == required
        and batch.usage.physical_calls == 0
        and len(batch.logical_completions) == required
        and len(batch.unique_records) == required,
        "materialization requires every semantic v3 checkpoint and no provider calls",
    )
    physical = tuple(row for row in plans if row["mode"] == SEMANTIC_MODE)
    _require(
        len(plans) == QUESTION_COUNT
        and tuple(row["ordinal"] for row in plans) == QUESTION_ORDINALS
        and len(physical) == required,
        "semantic v3 materialization population changed",
    )
    records = {row.messages_sha256: row for row in batch.unique_records}
    completions = {
        plan["ordinal"]: completion
        for plan, completion in zip(
            physical, batch.logical_completions, strict=True
        )
    }
    _require(len(records) == required, "semantic v3 completions repeat")
    results: list[dict[str, Any]] = []
    for plan in plans:
        parent = plan["parent_prediction"]
        if plan["mode"] == PARENT_PASSTHROUGH_MODE:
            body = _result_body(
                plan,
                prediction=parent,
                prediction_source=plan["parent_prediction_source"],
                decision=PARENT_PASSTHROUGH_MODE,
                completion_parser="none",
            )
        else:
            completion = completions[plan["ordinal"]]
            record = records.get(plan["messages_sha256"])
            _require(
                record is not None
                and record.completion == completion
                and record.checkpoint_hit is True
                and record.physical_call is False,
                f"semantic v3 checkpoint changed at ordinal {plan['ordinal']}",
            )
            assert record is not None
            parsed = typed_final.parse_typed_final_completion(
                completion,
                parent_prediction=parent,
                allowed_handle_ids=tuple(plan["allowed_handle_ids"]),
                handle_group_by_id=dict(plan["handle_group_by_id"]),
                story_coherence=dict(plan["story_coherence"]),
                preservation_requirements=dict(plan["preservation_requirements"]),
                validation_contract=dict(plan["validation_contract"]),
            )
            valid_replace = parsed.valid and parsed.decision == "replace"
            prediction = parsed.prediction if valid_replace else parent
            if valid_replace:
                source = "locked_semantic_v3_validated_replacement"
                decision = "replace"
                used = parsed.used_handle_ids
            elif parsed.valid:
                source = "locked_semantic_v3_validated_keep_parent"
                decision = "keep_parent"
                used = ()
            else:
                source = "locked_semantic_v3_invalid_keep_parent"
                decision = "invalid_keep_parent"
                used = ()
            body = _result_body(
                plan,
                prediction=prediction,
                prediction_source=source,
                decision=decision,
                completion_parser="typed_final_v1",
                call_key_sha256=record.call_key_sha256,
                completion_receipt_sha256=record.completion_sha256,
                parse_error_code=parsed.error_code,
                parse_receipt_sha256=parsed.receipt_sha256,
                prompt_row_receipt_sha256=plan["answer_plan_receipt_sha256"],
                request_journal_sha256=record.request_journal_sha256,
                response_journal_sha256=record.response_journal_sha256,
                solver_valid=parsed.valid,
                used_handle_ids=used,
                validation_basis=parsed.validation_basis,
            )
        row = {**body, "source_row_sha256": identity_sha256(body)}
        results.append(row)
    _require(
        tuple(row["ordinal"] for row in results) == QUESTION_ORDINALS,
        "semantic v3 result order changed",
    )
    payload = {
        "changed_prediction_count": sum(row["changed_from_parent"] for row in results),
        "completion_batch": _stable_batch(batch),
        "construction_artifact_sha256": preflight.payload[
            "construction_artifact_sha256"
        ],
        "construction_format": CONSTRUCTION_FORMAT,
        "format": FORMAT,
        "gold_loaded": False,
        "invalid_completion_parent_fallback_count": sum(
            row["prediction_source"] == "locked_semantic_v3_invalid_keep_parent"
            for row in results
        ),
        "judge_rows": [typed_final.judge_row_projection(row) for row in results],
        "model": DEFAULT_MODEL,
        "ordinals": list(QUESTION_ORDINALS),
        "parent_passthrough_count": sum(
            row["answer_mode"] == PARENT_PASSTHROUGH_MODE for row in results
        ),
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "question_count": QUESTION_COUNT,
        "questions": results,
        "required_authorized_provider_calls": required,
        "retained_transformer_token_state_bytes": 0,
        "semantic_question_count": sum(
            row["answer_mode"] == SEMANTIC_MODE for row in results
        ),
        "semantic_renderer_format": SEMANTIC_PROMPT_FORMAT,
        "validated_keep_parent_count": sum(
            row["prediction_source"] == "locked_semantic_v3_validated_keep_parent"
            for row in results
        ),
        "validated_replacement_count": sum(
            row["prediction_source"] == "locked_semantic_v3_validated_replacement"
            for row in results
        ),
    }
    assert_gold_blind(payload, path="locked_semantic_final_terra_answer_v3")
    return payload


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    preflight, prompts, plans = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    batch = _checkpoint_batch(preflight, prompts, args=args, client=None)
    payload = _materialization_projection(preflight, plans, batch)
    artifact, created = publish_sealed_json(Path(args.output_root) / RUN_NAME, payload)
    return {
        "changed_prediction_count": payload["changed_prediction_count"],
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "parent_passthrough_count": payload["parent_passthrough_count"],
        "physical_provider_calls": 0,
        "run_sha256": artifact.sha256,
        "terminal_run_replayed": not created,
        "validated_replacement_count": payload["validated_replacement_count"],
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    construction, source_plans = load_answer_plans(
        Path(args.construction), str(args.expected_construction_sha256)
    )
    preflight, prompts, plans = _read_preflight(
        Path(args.output_root), str(args.expected_preflight_sha256)
    )
    _require(
        preflight.payload.get("construction_artifact_sha256") == construction.sha256
        and source_plans == plans,
        "semantic v3 construction/preflight binding changed",
    )
    batch = _checkpoint_batch(preflight, prompts, args=args, client=None)
    rebuilt = _materialization_projection(preflight, plans, batch)
    terminal = read_sealed_json(Path(args.output_root) / RUN_NAME)
    _require(
        terminal.sha256
        == require_sha256(args.expected_run_sha256, "expected semantic v3 run")
        and terminal.payload == rebuilt,
        "semantic v3 run differs from checkpoint-only replay",
    )
    replay, _created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME, terminal.payload
    )
    _require(
        replay.sha256 == terminal.sha256,
        "semantic v3 replay is not byte-identical",
    )
    return {
        "byte_identical": True,
        "physical_provider_calls": 0,
        "replay_sha256": replay.sha256,
        "run_sha256": terminal.sha256,
    }


def _add_runtime_settings(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)
    parser.add_argument("--max-concurrency", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="seal semantic v3 prompts")
    _add_runtime_settings(preflight)
    preflight.add_argument("--construction", type=Path, default=DEFAULT_CONSTRUCTION)
    preflight.add_argument("--expected-construction-sha256", required=True)

    provider = commands.add_parser("provider-run", help="execute sealed Terra prompts")
    _add_runtime_settings(provider)
    provider.add_argument("--expected-preflight-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, default=0)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)

    materialize = commands.add_parser(
        "materialize", help="materialize the four terminal semantic answers"
    )
    _add_runtime_settings(materialize)
    materialize.add_argument("--expected-preflight-sha256", required=True)

    replay = commands.add_parser("replay", help="prove byte-identical v3 replay")
    _add_runtime_settings(replay)
    replay.add_argument("--construction", type=Path, default=DEFAULT_CONSTRUCTION)
    replay.add_argument("--expected-construction-sha256", required=True)
    replay.add_argument("--expected-preflight-sha256", required=True)
    replay.add_argument("--expected-run-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = run_preflight(args)
    elif args.command == "provider-run":
        result = run_provider(args)
    elif args.command == "materialize":
        result = run_materialize(args)
    elif args.command == "replay":
        result = run_replay(args)
    else:  # pragma: no cover
        raise AssertionError("unreachable command")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_DIR_NAME",
    "CLASSIFIED_CLOSURE_FORMAT",
    "CONSTRUCTION_FORMAT",
    "DEFAULT_CONSTRUCTION",
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT",
    "FORMAT",
    "HARD_COMPLETE_CHAT_TOKEN_CAP",
    "LOCAL_AUDIT_FORMAT",
    "LockedSemanticFinalAnswerV3Error",
    "MAX_CHAT_PROMPT_TOKENS",
    "OUTPUT_TOKEN_RESERVE",
    "PARENT_PASSTHROUGH_MODE",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_NAME",
    "QUESTION_COUNT",
    "QUESTION_ORDINALS",
    "REPLAY_NAME",
    "RESULT_ROW_FORMAT",
    "RUN_NAME",
    "SEMANTIC_MODE",
    "SEMANTIC_PROMPT_FORMAT",
    "STORED_SEARCH_FORMAT",
    "build_parser",
    "load_answer_plans",
    "main",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
]
