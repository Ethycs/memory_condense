#!/usr/bin/env python3
"""Combined v2 answer lifecycle for specialist and semantic-search evidence.

The future combined construction supplies one hundred replay-bound rows in
three modes. ``specialist`` rows use the closed specialist-v3 renderer and
parser, ``semantic_residual`` rows use the ordinary typed-final renderer and
parser over the terminal binary-search survivors, and ``parent_passthrough``
rows never enter the provider population. Every invalid or keep decision
returns the sealed parent byte-for-byte.

Construction is loaded through a lazy, injectable adapter. Preflight is the
only prompt-authority boundary; provider execution is checkpointed and
materialization/replay are provider-free.
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
from tools.matched_eval.population import EXPECTED_QUESTION_COUNT  # noqa: E402
from tools.matched_eval.specialist_scoped_completion import (  # noqa: E402
    FORMAT as SCOPED_COMPLETION_FORMAT,
    parse_specialist_scoped_completion,
)


FORMAT = "memory-condense-locked-semantic-final-terra-answer-v2"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight"
RESULT_ROW_FORMAT = f"{FORMAT}-result-row"
SEMANTIC_PROMPT_FORMAT = (
    f"{typed_final.PROMPT_ROW_FORMAT}-render-final-messages-v1"
)
CLASSIFIED_CLOSURE_FORMAT = (
    "memory-condense-reduced-semantic-binary-search-assay-v1-"
    "classified-closure-v1"
)

PREFLIGHT_NAME = "locked-semantic-final-answer-preflight-v2.json"
RUN_NAME = "locked-semantic-final-answer-v2.json"
REPLAY_NAME = "locked-semantic-final-answer-replay-v2.json"
CHECKPOINT_DIR_NAME = "locked-semantic-final-answer-checkpoints-v2"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONSTRUCTION = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-semantic-final-v2/"
    "locked-semantic-final-construction-v2.json"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-semantic-final-answer-v2"
)
DEFAULT_MODEL = "codex_sdk/gpt-5.6-terra"

SPECIALIST_MODE = specialist_v1.SPECIALIST_MODE
SEMANTIC_MODE = "semantic_residual"
PARENT_PASSTHROUGH_MODE = specialist_v1.PARENT_PASSTHROUGH_MODE
_MODES = frozenset({SPECIALIST_MODE, SEMANTIC_MODE, PARENT_PASSTHROUGH_MODE})

HARD_COMPLETE_CHAT_TOKEN_CAP = typed_final.HARD_PROMPT_TOKEN_CAP
MAX_CHAT_PROMPT_TOKENS = typed_final.MAX_CHAT_PROMPT_TOKENS
OUTPUT_TOKEN_RESERVE = typed_final.OUTPUT_TOKEN_RESERVE

_COMMON_PLAN_FIELDS = specialist_v1._COMMON_PLAN_FIELDS  # noqa: SLF001
_SPECIALIST_PLAN_FIELDS = specialist_v1._SPECIALIST_PLAN_FIELDS  # noqa: SLF001
_SEMANTIC_PLAN_FIELDS = _COMMON_PLAN_FIELDS | frozenset(
    {
        "additive_composition_local_audit_sha256",
        "additive_composition_receipt_sha256",
        "allowed_handle_ids",
        "classified_closure",
        "fitted_prompt_receipt_sha256",
        "handle_group_by_id",
        "messages",
        "messages_sha256",
        "namespace_id",
        "preservation_requirements",
        "prompt_token_proxy",
        "provider_input",
        "query_vector_artifact_sha256",
        "query_vector_row_receipt_sha256",
        "semantic_query_receipt_sha256",
        "semantic_residual_index_receipt_sha256",
        "semantic_residual_local_audit_sha256",
        "semantic_residual_search_receipt_sha256",
        "story_coherence",
        "terminal_prompt_receipt_sha256",
        "validation_contract",
    }
)

_SEMANTIC_FALLBACK_PLAN_FIELDS = _COMMON_PLAN_FIELDS | frozenset(
    {
        "fallback_reason",
        "namespace_id",
        "query_vector_artifact_sha256",
        "query_vector_row_receipt_sha256",
        "semantic_query_receipt_sha256",
        "semantic_residual_index_receipt_sha256",
        "semantic_residual_local_audit_sha256",
        "semantic_residual_search_receipt_sha256",
    }
)

_SEMANTIC_SOURCE_FIELDS = frozenset(
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

_CLASSIFIED_CLOSURE_FIELDS = frozenset(
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
_CLASSIFIED_CLOSURE_ROW_FIELDS = frozenset(
    {
        "cell_id",
        "dedup_exclusion_sha256",
        "disposition",
        "exact_text_sha256",
        "residual_evidence_receipt_sha256",
        "residual_binding_receipt_sha256",
        "residual_item_receipt_sha256",
        "segment_receipt_sha256",
        "visible_binding_receipt_sha256s",
        "visible_handle_ids",
        "visible_item_receipt_sha256",
    }
)

_SEMANTIC_FITTED_FIELDS = frozenset(
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
_SEMANTIC_TERMINAL_FIELDS = frozenset(
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
_SEMANTIC_PROVIDER_FIELDS = frozenset(
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

_PREFLIGHT_FIELDS = frozenset(
    {
        "answer_plan_population_sha256",
        "construction_artifact_sha256",
        "format",
        "gateway_url",
        "gold_loaded",
        "hard_complete_chat_token_cap",
        "max_chat_prompt_tokens",
        "max_concurrency",
        "model",
        "observed_max_complete_envelope_tokens",
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
        "scoped_completion_format",
        "semantic_question_count",
        "semantic_renderer_format",
        "specialist_question_count",
    }
)

ConstructionLoader = Callable[..., tuple[SealedArtifact, Sequence[Mapping[str, Any]]]]


class LockedSemanticFinalAnswerError(MatchedEvalContractError):
    """Raised when a v2 source, prompt, checkpoint, or replay diverges."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSemanticFinalAnswerError(message)


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
        "semantic final messages changed schema",
    )
    return rows


def _default_construction_loader(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[SealedArtifact, Sequence[Mapping[str, Any]]]:
    module = importlib.import_module("tools.run_locked_semantic_final_construction")
    loader = getattr(module, "load_verified_construction", None)
    _require(callable(loader), "combined semantic construction loader is unavailable")
    return loader(path, expected_sha256=expected_sha256)


def _semantic_evidence_handles(
    provider_input: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, str]]:
    typed = provider_input.get("typed_evidence")
    _require(type(typed) is dict, "semantic typed evidence is missing")
    assert type(typed) is dict
    handles = typed.get("handles")
    items = typed.get("items")
    _require(
        type(handles) is list
        and bool(handles)
        and type(items) is list
        and bool(items),
        "semantic survivor evidence is empty",
    )
    ordered: list[str] = []
    groups: dict[str, str] = {}
    for raw in handles:
        _require(
            type(raw) is dict
            and set(raw)
            == {"group_handle", "handle_id", "origin", "provenance_grade"}
            and type(raw.get("handle_id")) is str
            and bool(raw["handle_id"])
            and type(raw.get("group_handle")) is str
            and bool(raw["group_handle"]),
            "semantic survivor handle schema changed",
        )
        assert type(raw) is dict
        handle = str(raw["handle_id"])
        _require(handle not in groups, "semantic survivor handles repeat")
        ordered.append(handle)
        groups[handle] = str(raw["group_handle"])
    represented: set[str] = set()
    for raw in items:
        _require(type(raw) is dict, "semantic survivor item changed type")
        assert type(raw) is dict
        cited = raw.get("handle_ids")
        _require(
            type(cited) is list
            and bool(cited)
            and all(type(value) is str and value in groups for value in cited),
            "semantic survivor item escaped its exact handles",
        )
        represented.update(cited)
    _require(
        represented == set(ordered),
        "semantic fitted prompt contains unrepresented survivor handles",
    )
    frontier = typed.get("frontier")
    _require(
        type(frontier) is dict
        and frontier.get("available_handle_ids") == ordered
        and frontier.get("represented_handle_ids") == ordered,
        "semantic fitted frontier differs from its visible handles",
    )
    return tuple(ordered), groups


def _semantic_parser_contract(
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
        raise LockedSemanticFinalAnswerError(
            f"semantic typed-final parser contract changed: {exc}"
        ) from exc
    _require(
        parsed.valid
        and parsed.decision == "keep_parent"
        and parsed.prediction == parent,
        "semantic keep-parent parser self-check failed",
    )


def _semantic_common_plan(
    raw: Mapping[str, Any],
    ordinal: int,
    *,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    """Project v1's parent seam without applying its two-mode gate."""

    judge = parent.get("parent_judge_row")
    _require(type(judge) is dict, f"semantic parent judge is missing at ordinal {ordinal}")
    assert type(judge) is dict
    _require(
        raw.get("mode") == SEMANTIC_MODE,
        f"semantic answer mode changed at ordinal {ordinal}",
    )
    return {
        "construction_question_receipt_sha256": require_sha256(
            raw.get("question_receipt_sha256"), "construction question receipt"
        ),
        "dated_question_sha256": require_sha256(
            raw.get("dated_question_sha256"), "dated question"
        ),
        "mode": SEMANTIC_MODE,
        "ordinal": ordinal,
        "parent_judge_row_sha256": require_sha256(
            parent.get("parent_judge_row_sha256"), "parent judge row"
        ),
        "parent_prediction": require_text(
            parent.get("prediction"), "parent prediction"
        ),
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
        "question_id": require_text(raw.get("question_id"), "question ID"),
        "question_sha256": require_sha256(raw.get("question_sha256"), "question"),
        "route_id": require_text(judge.get("route_id"), "parent route"),
    }


def _semantic_search_binding(
    raw: Mapping[str, Any],
    *,
    allowed: Sequence[str],
    provider_input: Mapping[str, Any],
) -> dict[str, Any]:
    binding = raw.get("semantic_search_binding")
    _require(type(binding) is dict, "semantic search binding is missing")
    assert type(binding) is dict
    _require(
        set(binding)
        == {
            "activation_reason",
            "dedup_receipt_sha256",
            "definitely_no_leaf_receipt_sha256s",
            "excluded_overlap_rows",
            "frontier_closed",
            "frontier_complete",
            "full_leaf_partition_receipt_sha256",
            "full_leaf_receipt_sha256s",
            "may_survivor_leaf_receipt_sha256s",
            "no_silent_top_k_uncertainty_drop",
            "omitted_unknown_leaf_receipt_sha256s",
            "receipt_sha256",
            "retained_leaf_rows",
            "search_result_receipt_sha256",
            "selected_leaf_receipt_sha256s",
            "selection_precedes_dedup",
            "specialists_evaluated_first",
        },
        "semantic search binding schema changed",
    )
    body = dict(binding)
    declared = body.pop("receipt_sha256", None)
    reason = binding.get("activation_reason")
    applicable = raw.get("applicable_specialist_ids")
    leaves = binding.get("retained_leaf_rows")
    selected = binding.get("selected_leaf_receipt_sha256s")
    excluded = binding.get("excluded_overlap_rows")
    full_leaves = binding.get("full_leaf_receipt_sha256s")
    definitely_no = binding.get("definitely_no_leaf_receipt_sha256s")
    may_survivors = binding.get("may_survivor_leaf_receipt_sha256s")
    omitted_unknown = binding.get("omitted_unknown_leaf_receipt_sha256s")
    _require(
        declared == identity_sha256(body)
        and reason in {"no_applicable_specialist", "specialist_proofless"}
        and type(applicable) is list
        and (
            not applicable
            if reason == "no_applicable_specialist"
            else bool(applicable)
        )
        and binding.get("specialists_evaluated_first") is True
        and binding.get("selection_precedes_dedup") is True
        and binding.get("no_silent_top_k_uncertainty_drop") is True
        and binding.get("frontier_closed") is True
        and binding.get("frontier_complete") is True
        and type(leaves) is list
        and bool(leaves)
        and type(selected) is list
        and bool(selected)
        and len(selected) == len(set(selected))
        and type(excluded) is list
        and type(full_leaves) is list
        and bool(full_leaves)
        and len(full_leaves) == len(set(full_leaves))
        and type(definitely_no) is list
        and len(definitely_no) == len(set(definitely_no))
        and type(may_survivors) is list
        and bool(may_survivors)
        and len(may_survivors) == len(set(may_survivors))
        and omitted_unknown == [],
        "semantic search activation/order contract changed",
    )
    require_sha256(
        binding.get("search_result_receipt_sha256"),
        "semantic search result",
    )
    require_sha256(declared, "semantic search binding")
    require_sha256(binding.get("dedup_receipt_sha256"), "semantic selected-leaf dedup")
    partition_receipt = require_sha256(
        binding.get("full_leaf_partition_receipt_sha256"),
        "semantic full leaf partition",
    )
    _require(
        partition_receipt
        == identity_sha256({"full_leaf_receipt_sha256s": full_leaves}),
        "semantic full leaf partition receipt changed",
    )
    for receipt in (*full_leaves, *definitely_no, *may_survivors):
        require_sha256(receipt, "semantic frontier leaf")
    _require(
        not set(definitely_no) & set(may_survivors)
        and set(full_leaves) == set(definitely_no) | set(may_survivors)
        and [receipt for receipt in full_leaves if receipt not in set(definitely_no)]
        == may_survivors
        and selected == may_survivors,
        "semantic closed frontier leaf partition changed",
    )
    for receipt in selected:
        require_sha256(receipt, "semantic selected leaf")
    leaf_text_by_handle: dict[str, str] = {}
    for leaf in leaves:
        _require(
            type(leaf) is dict
            and set(leaf)
            == {
                "exact_text",
                "exact_text_sha256",
                "handle_id",
                "leaf_receipt_sha256",
            },
            "semantic retained leaf schema changed",
        )
        assert type(leaf) is dict
        handle = require_text(leaf.get("handle_id"), "semantic leaf handle")
        exact_text = require_text(leaf.get("exact_text"), "semantic exact leaf text")
        _require(
            handle not in leaf_text_by_handle
            and leaf.get("exact_text_sha256") == quote_sha256(exact_text),
            "semantic retained leaf text seal changed",
        )
        require_sha256(leaf.get("leaf_receipt_sha256"), "semantic leaf receipt")
        leaf_text_by_handle[handle] = exact_text
    excluded_receipts: list[str] = []
    for overlap in excluded:
        _require(
            type(overlap) is dict
            and set(overlap)
            == {
                "overlap_local_receipt_sha256",
                "receipt_sha256",
                "selected_leaf_receipt_sha256",
            },
            "semantic selected overlap schema changed",
        )
        assert type(overlap) is dict
        overlap_body = dict(overlap)
        overlap_receipt = overlap_body.pop("receipt_sha256", None)
        selected_receipt = require_sha256(
            overlap.get("selected_leaf_receipt_sha256"),
            "semantic excluded selected leaf",
        )
        require_sha256(
            overlap.get("overlap_local_receipt_sha256"),
            "semantic overlap local receipt",
        )
        _require(
            overlap_receipt == identity_sha256(overlap_body),
            "semantic selected overlap receipt changed",
        )
        excluded_receipts.append(selected_receipt)
    retained_receipts = [
        str(leaf["leaf_receipt_sha256"])
        for leaf in leaves
        if type(leaf) is dict
    ]
    _require(
        len(excluded_receipts) == len(set(excluded_receipts))
        and not set(excluded_receipts) & set(retained_receipts)
        and set(selected) == set(excluded_receipts) | set(retained_receipts),
        "semantic selected-then-dedup partition changed",
    )
    _require(
        [receipt for receipt in selected if receipt not in set(excluded_receipts)]
        == retained_receipts,
        "semantic dedup changed retained selection order",
    )
    _require(
        set(leaf_text_by_handle) <= set(allowed),
        "semantic retained leaf escaped the fitted handles",
    )
    typed = provider_input.get("typed_evidence")
    assert type(typed) is dict
    frontier = typed.get("frontier")
    _require(
        type(frontier) is dict
        and frontier.get("closed") is True
        and frontier.get("truncated") is False
        and frontier.get("omitted_handle_ids") == []
        and frontier.get("unresolved_slot_ids") == [],
        "semantic terminal prompt lost its closed complete frontier",
    )
    items = typed.get("items")
    assert type(items) is list
    provider_leaf_rows: dict[str, list[str]] = {}
    for item in items:
        assert type(item) is dict
        cited = item.get("handle_ids")
        summary = item.get("summary")
        _require(
            type(cited) is list
            and bool(cited)
            and type(summary) is str
            and bool(summary),
            "semantic provider item is not exact visible evidence",
        )
        for handle in cited:
            provider_leaf_rows.setdefault(handle, []).append(summary)
    _require(
        all(
            exact_text in provider_leaf_rows.get(handle, [])
            for handle, exact_text in leaf_text_by_handle.items()
        ),
        "semantic final prompt substituted or summarized retained leaf text",
    )
    return dict(binding)


def _semantic_classified_closure(
    raw: Mapping[str, Any],
    *,
    allowed: Sequence[str],
    provider_input: Mapping[str, Any],
    search_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove every selected may leaf remains visible after terminal dedup."""

    closure = raw.get("classified_closure")
    _require(
        type(closure) is dict and set(closure) == _CLASSIFIED_CLOSURE_FIELDS,
        "semantic classified closure schema changed",
    )
    assert type(closure) is dict
    body = dict(closure)
    declared = body.pop("receipt_sha256", None)
    rows = closure.get("rows")
    selected = search_binding.get("selected_leaf_receipt_sha256s")
    definitely_no = search_binding.get("definitely_no_leaf_receipt_sha256s")
    retained = search_binding.get("retained_leaf_rows")
    excluded = search_binding.get("excluded_overlap_rows")
    _require(
        declared == identity_sha256(body)
        and closure.get("format") == CLASSIFIED_CLOSURE_FORMAT
        and closure.get("semantic_search_result_receipt_sha256")
        == search_binding.get("search_result_receipt_sha256")
        and closure.get("leaf_partition_receipt_sha256")
        == search_binding.get("full_leaf_partition_receipt_sha256")
        and type(selected) is list
        and type(definitely_no) is list
        and closure.get("selected_may_leaf_count") == len(selected)
        and closure.get("definitely_no_leaf_count") == len(definitely_no)
        and closure.get("terminal_allowed_handle_ids_sha256")
        == identity_sha256({"allowed_handle_ids": list(allowed)})
        and closure.get("all_selected_may_leaves_provider_visible") is True
        and closure.get("typed_frontier_closed") is True
        and type(rows) is list
        and len(rows) == len(selected),
        "semantic classified closure seal changed",
    )
    require_sha256(declared, "semantic classified closure")
    assert type(retained) is list and type(excluded) is list
    retained_by_cell = {
        str(row["leaf_receipt_sha256"]): row
        for row in retained
        if type(row) is dict
    }
    excluded_by_cell = {
        str(row["selected_leaf_receipt_sha256"]): row
        for row in excluded
        if type(row) is dict
    }
    typed = provider_input.get("typed_evidence")
    assert type(typed) is dict
    items = typed.get("items")
    frontier = typed.get("frontier")
    _require(
        type(items) is list
        and type(frontier) is dict
        and frontier.get("closed") is True
        and frontier.get("truncated") is False
        and frontier.get("omitted_handle_ids") == []
        and frontier.get("unresolved_slot_ids") == [],
        "semantic classified closure lost terminal frontier visibility",
    )
    visible_summaries: dict[str, list[str]] = {}
    for item in items:
        _require(type(item) is dict, "semantic terminal evidence item changed type")
        assert type(item) is dict
        handles = item.get("handle_ids")
        summary = item.get("summary")
        _require(
            type(handles) is list
            and bool(handles)
            and type(summary) is str
            and bool(summary),
            "semantic terminal evidence item lost exact text",
        )
        for handle in handles:
            visible_summaries.setdefault(str(handle), []).append(summary)

    cell_ids: list[str] = []
    cell_receipts: list[str] = []
    visible_handles: list[str] = []
    direct_handles = {
        str(row["handle_id"])
        for row in retained
        if type(row) is dict
    }
    for classified in rows:
        _require(
            type(classified) is dict
            and set(classified) == _CLASSIFIED_CLOSURE_ROW_FIELDS,
            "semantic classified closure row schema changed",
        )
        assert type(classified) is dict
        cell_id = require_text(classified.get("cell_id"), "semantic leaf cell ID")
        cell_receipt = require_sha256(
            classified.get("cell_receipt_sha256"), "semantic leaf cell"
        )
        exact_text_sha = require_sha256(
            classified.get("exact_text_sha256"), "semantic leaf exact text"
        )
        disposition = classified.get("disposition")
        visible_handle = require_text(
            classified.get("visible_handle_id"), "semantic visible owner handle"
        )
        residual_binding = require_sha256(
            classified.get("residual_binding_receipt_sha256"),
            "semantic residual binding",
        )
        residual_item = require_sha256(
            classified.get("residual_item_receipt_sha256"),
            "semantic residual item",
        )
        visible_binding = require_sha256(
            classified.get("visible_binding_receipt_sha256"),
            "semantic visible binding",
        )
        visible_item = require_sha256(
            classified.get("visible_item_receipt_sha256"),
            "semantic visible item",
        )
        dedup_exclusion = classified.get("dedup_exclusion_sha256")
        _require(
            disposition in {"residual_visible", "protected_visible_exact_duplicate"}
            and visible_handle in allowed
            and any(
                quote_sha256(summary) == exact_text_sha
                for summary in visible_summaries.get(visible_handle, [])
            ),
            "semantic selected leaf is not byte-identically provider-visible",
        )
        if disposition == "residual_visible":
            leaf = retained_by_cell.get(cell_receipt)
            _require(
                type(leaf) is dict
                and leaf.get("handle_id") == visible_handle
                and leaf.get("exact_text_sha256") == exact_text_sha
                and residual_binding == visible_binding
                and residual_item == visible_item
                and dedup_exclusion is None,
                "semantic residual visibility classification changed",
            )
        else:
            overlap = excluded_by_cell.get(cell_receipt)
            _require(
                type(overlap) is dict
                and dedup_exclusion == overlap.get("receipt_sha256")
                and visible_handle not in direct_handles,
                "semantic protected-owner dedup classification changed",
            )
            require_sha256(dedup_exclusion, "semantic dedup exclusion")
        cell_ids.append(cell_id)
        cell_receipts.append(cell_receipt)
        visible_handles.append(visible_handle)
    _require(
        len(cell_ids) == len(set(cell_ids))
        and cell_receipts == selected
        and len(visible_handles) == len(set(visible_handles)),
        "semantic selected leaf classification is incomplete or duplicated",
    )
    return dict(closure)


def _sealed_projection(value: object, label: str) -> tuple[dict[str, Any], str]:
    _require(type(value) is dict, f"{label} changed type")
    assert type(value) is dict
    body = dict(value)
    receipt = require_sha256(body.pop("receipt_sha256", None), label)
    _require(receipt == identity_sha256(body), f"{label} receipt changed")
    return dict(value), receipt


def _semantic_residual_contract(
    raw: Mapping[str, Any],
    *,
    require_closed: bool,
) -> dict[str, Any]:
    """Validate the canonical residual result and its local replay seam."""

    query, query_receipt = _sealed_projection(
        raw.get("semantic_query"), "semantic residual query"
    )
    search, search_receipt = _sealed_projection(
        raw.get("semantic_residual_search"), "semantic residual search"
    )
    expected_search_fields = {
        "attempted_evidence_count",
        "attempted_provider_payload_tokens",
        "classified_frontier",
        "core_result",
        "decision_audits",
        "dedup_after_semantic_selection",
        "evidence",
        "fallback_reason",
        "fallback_required",
        "format",
        "gold_loaded",
        "local_binding_receipt_sha256s",
        "new_provider_calls",
        "protected_duplicates",
        "protected_evidence_mutated",
        "protected_evidence_population_receipt_sha256",
        "provider_payload_tokens",
        "query_receipt_sha256",
        "query_vector_artifact_sha256",
        "receipt_sha256",
        "residual_index_receipt_sha256",
        "retained_transformer_token_state_bytes",
        "searched_complete_memory_population",
        "terminal_after_specialist_selection",
    }
    _require(
        set(search) == expected_search_fields
        and search.get("format")
        == "memory-condense-semantic-residual-terminal-result-v1"
        and search.get("gold_loaded") is False
        and search.get("new_provider_calls") == 0
        and search.get("retained_transformer_token_state_bytes") == 0
        and search.get("searched_complete_memory_population") is True
        and search.get("terminal_after_specialist_selection") is True
        and search.get("dedup_after_semantic_selection") is True
        and search.get("protected_evidence_mutated") is False
        and search.get("query_receipt_sha256") == query_receipt
        and search.get("query_vector_artifact_sha256")
        == raw.get("query_vector_artifact_sha256")
        and search.get("residual_index_receipt_sha256")
        == raw.get("semantic_residual_index_receipt_sha256")
        and type(search.get("decision_audits")) is list
        and type(search.get("evidence")) is list
        and type(search.get("local_binding_receipt_sha256s")) is list
        and search.get("protected_duplicates") == [],
        "canonical semantic residual result changed",
    )
    require_sha256(
        search.get("protected_evidence_population_receipt_sha256"),
        "semantic empty protected population",
    )
    frontier, frontier_receipt = _sealed_projection(
        search.get("classified_frontier"), "semantic classified frontier"
    )
    expected_frontier_fields = {
        "all_novel_survivors_protected",
        "certified_negative_leaf_cell_ids",
        "classified_leaf_count",
        "closed",
        "complete_leaf_partition",
        "core_result_receipt_sha256",
        "format",
        "packed_segment_receipt_sha256s",
        "protected_duplicate_audit_receipt_sha256s",
        "protected_duplicate_segment_receipt_sha256s",
        "receipt_sha256",
        "residual_index_receipt_sha256",
        "retained_leaf_cell_ids",
        "retained_segment_receipt_sha256s",
        "unresolved_segment_receipt_sha256s",
    }
    retained_cells = frontier.get("retained_leaf_cell_ids")
    negative_cells = frontier.get("certified_negative_leaf_cell_ids")
    retained_segments = frontier.get("retained_segment_receipt_sha256s")
    packed_segments = frontier.get("packed_segment_receipt_sha256s")
    duplicate_segments = frontier.get(
        "protected_duplicate_segment_receipt_sha256s"
    )
    duplicate_audits = frontier.get(
        "protected_duplicate_audit_receipt_sha256s"
    )
    unresolved_segments = frontier.get("unresolved_segment_receipt_sha256s")
    _require(
        set(frontier) == expected_frontier_fields
        and frontier.get("format")
        == "memory-condense-semantic-residual-classified-frontier-v1"
        and frontier.get("complete_leaf_partition") is True
        and type(retained_cells) is list
        and type(negative_cells) is list
        and type(retained_segments) is list
        and type(packed_segments) is list
        and type(duplicate_segments) is list
        and type(duplicate_audits) is list
        and type(unresolved_segments) is list
        and all(
            len(values) == len(set(values))
            for values in (
                retained_cells,
                negative_cells,
                retained_segments,
                packed_segments,
                duplicate_segments,
                duplicate_audits,
                unresolved_segments,
            )
        )
        and not set(retained_cells) & set(negative_cells)
        and frontier.get("classified_leaf_count")
        == len(retained_cells) + len(negative_cells)
        and not set(packed_segments) & set(duplicate_segments)
        and not set(packed_segments) & set(unresolved_segments)
        and not set(duplicate_segments) & set(unresolved_segments)
        and set(retained_segments)
        == set(packed_segments) | set(duplicate_segments) | set(unresolved_segments)
        and duplicate_segments == []
        and duplicate_audits == []
        and frontier.get("all_novel_survivors_protected")
        == (not unresolved_segments)
        and frontier.get("closed") == (not unresolved_segments)
        and frontier.get("residual_index_receipt_sha256")
        == search.get("residual_index_receipt_sha256"),
        "canonical semantic residual frontier changed",
    )
    for value in (*retained_segments, *packed_segments, *unresolved_segments):
        require_sha256(value, "semantic residual segment")
    core, core_receipt = _sealed_projection(
        search.get("core_result"), "semantic binary-search core result"
    )
    _require(
        frontier.get("core_result_receipt_sha256") == core_receipt
        and core.get("retained_leaf_cell_ids") == retained_cells
        and core.get("pruned_leaf_cell_ids") == negative_cells
        and core.get("provider_calls_performed_by_core") == 0
        and core.get("retained_transformer_token_state_bytes") == 0
        and core.get("gold_loaded") is False,
        "semantic residual leaf partition differs from the search core",
    )
    if require_closed:
        _require(
            frontier.get("closed") is True
            and unresolved_segments == []
            and search.get("fallback_required") is False
            and search.get("fallback_reason") == "none",
            "semantic provider row does not have a closed complete frontier",
        )

    evidence_by_segment: dict[str, dict[str, Any]] = {}
    evidence_receipts: list[str] = []
    binding_receipts = search.get("local_binding_receipt_sha256s")
    assert type(binding_receipts) is list
    for evidence in search.get("evidence", []):
        sealed, evidence_receipt = _sealed_projection(
            evidence, "semantic residual exact evidence"
        )
        segment = require_sha256(
            sealed.get("segment_receipt_sha256"), "semantic evidence segment"
        )
        quote = require_text(sealed.get("quote"), "semantic residual exact quote")
        _require(
            sealed.get("format")
            == "memory-condense-semantic-residual-exact-evidence-v1"
            and sealed.get("quote_sha256") == quote_sha256(quote)
            and segment not in evidence_by_segment,
            "semantic residual evidence lost exact segment bytes",
        )
        require_sha256(
            sealed.get("citation_binding_receipt_sha256"),
            "semantic residual citation binding",
        )
        evidence_by_segment[segment] = sealed
        evidence_receipts.append(evidence_receipt)
    _require(
        list(evidence_by_segment) == packed_segments
        and len(binding_receipts) == len(evidence_by_segment)
        and binding_receipts
        == [
            row["citation_binding_receipt_sha256"]
            for row in evidence_by_segment.values()
        ],
        "semantic residual evidence differs from packed segments",
    )

    audit = raw.get("semantic_residual_local_audit")
    _require(
        type(audit) is dict
        and set(audit)
        == {
            "classified_frontier",
            "compact_result_receipt_sha256",
            "local_bindings",
            "protected_duplicates",
            "query",
        }
        and audit.get("classified_frontier") == frontier
        and audit.get("compact_result_receipt_sha256") == search_receipt
        and audit.get("protected_duplicates") == []
        and audit.get("query") == query
        and type(audit.get("local_bindings")) is list,
        "semantic residual local replay audit changed",
    )
    local_binding_receipts: list[str] = []
    for binding in audit.get("local_bindings", []):
        _sealed, receipt = _sealed_projection(
            binding, "semantic residual local binding"
        )
        local_binding_receipts.append(receipt)
    _require(
        local_binding_receipts == binding_receipts,
        "semantic residual local bindings differ from provider evidence",
    )
    return {
        "audit_sha256": identity_sha256(audit),
        "evidence_by_segment": evidence_by_segment,
        "frontier": frontier,
        "frontier_receipt_sha256": frontier_receipt,
        "query_receipt_sha256": query_receipt,
        "search_receipt_sha256": search_receipt,
    }


def _composition_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    composition, composition_receipt = _sealed_projection(
        raw.get("additive_composition"), "semantic additive composition"
    )
    audit = raw.get("additive_composition_local_audit")
    _require(type(audit) is dict, "semantic additive local audit changed type")
    assert type(audit) is dict
    dedup = audit.get("post_selection_dedup")
    dedup_projection, dedup_receipt = _sealed_projection(
        dedup, "semantic post-selection dedup audit"
    )
    _require(
        composition.get("post_selection_dedup_audit_receipt_sha256")
        == dedup_receipt
        and composition.get("gold_loaded") is False
        and composition.get("provider_prompt_count") == 0
        and composition.get("retained_transformer_token_state_bytes") == 0
        and dedup_projection.get("operation_position")
        == "after_all_mechanism_selection",
        "semantic additive composition or selected-then-dedup seam changed",
    )
    exclusions = dedup_projection.get("exclusions")
    _require(type(exclusions) is list, "semantic dedup exclusions changed type")
    assert type(exclusions) is list
    return {
        "audit_sha256": identity_sha256(audit),
        "composition_receipt_sha256": composition_receipt,
        "dedup_exclusion_by_sha256": {
            identity_sha256(row): row for row in exclusions if type(row) is dict
        },
        "dedup_receipt_sha256": dedup_receipt,
    }


def _classified_segment_closure(
    raw: Mapping[str, Any],
    *,
    allowed: Sequence[str],
    provider_input: Mapping[str, Any],
    fitted_receipt: str,
    residual_contract: Mapping[str, Any],
    composition_contract: Mapping[str, Any],
) -> dict[str, Any]:
    closure = raw.get("classified_closure")
    _require(
        type(closure) is dict and set(closure) == _CLASSIFIED_CLOSURE_FIELDS,
        "semantic classified closure schema changed",
    )
    assert type(closure) is dict
    body = dict(closure)
    declared = require_sha256(
        body.pop("receipt_sha256", None), "semantic classified closure"
    )
    rows = closure.get("rows")
    retained_segments = residual_contract["frontier"].get(
        "retained_segment_receipt_sha256s"
    )
    _require(
        declared == identity_sha256(body)
        and closure.get("format") == CLASSIFIED_CLOSURE_FORMAT
        and closure.get("semantic_residual_search_receipt_sha256")
        == residual_contract["search_receipt_sha256"]
        and closure.get("classified_frontier_receipt_sha256")
        == residual_contract["frontier_receipt_sha256"]
        and closure.get("post_selection_dedup_audit_receipt_sha256")
        == composition_contract["dedup_receipt_sha256"]
        and closure.get("fitted_prompt_receipt_sha256") == fitted_receipt
        and closure.get("retained_segment_receipt_sha256s") == retained_segments
        and closure.get("terminal_allowed_handle_ids") == list(allowed)
        and closure.get("complete_leaf_partition") is True
        and closure.get("all_retained_segments_provider_visible") is True
        and closure.get("closed") is True
        and type(rows) is list
        and len(rows) == len(retained_segments),
        "semantic classified closure seal changed",
    )
    allowed_body = {
        "format": f"{CLASSIFIED_CLOSURE_FORMAT}-terminal-allowed-handles-v1",
        "terminal_allowed_handle_ids": list(allowed),
    }
    protection_body = {
        "format": f"{CLASSIFIED_CLOSURE_FORMAT}-protection-source-v1",
        "semantic_residual_search_receipt_sha256": residual_contract[
            "search_receipt_sha256"
        ],
        "classified_frontier_receipt_sha256": residual_contract[
            "frontier_receipt_sha256"
        ],
        "post_selection_dedup_audit_receipt_sha256": composition_contract[
            "dedup_receipt_sha256"
        ],
        "retained_segment_receipt_sha256s": list(retained_segments),
        "rows": rows,
    }
    _require(
        closure.get("terminal_allowed_handle_ids_sha256")
        == identity_sha256(allowed_body)
        and closure.get("protection_source_receipt_sha256")
        == identity_sha256(protection_body),
        "semantic terminal visibility protection receipt changed",
    )
    typed = provider_input.get("typed_evidence")
    assert type(typed) is dict
    items = typed.get("items")
    _require(type(items) is list, "semantic terminal typed items changed type")
    visible_items: set[tuple[tuple[str, ...], str]] = set()
    for item in items:
        _require(type(item) is dict, "semantic terminal typed item changed type")
        assert type(item) is dict
        handles = item.get("handle_ids")
        summary = item.get("summary")
        _require(
            type(handles) is list
            and bool(handles)
            and type(summary) is str
            and bool(summary),
            "semantic terminal typed item lost exact bytes",
        )
        visible_items.add((tuple(handles), quote_sha256(summary)))

    evidence_by_segment = residual_contract["evidence_by_segment"]
    seen_handles: set[str] = set()
    observed_segments: list[str] = []
    for row in rows:
        _require(
            type(row) is dict and set(row) == _CLASSIFIED_CLOSURE_ROW_FIELDS,
            "semantic classified segment row schema changed",
        )
        assert type(row) is dict
        segment = require_sha256(
            row.get("segment_receipt_sha256"), "semantic classified segment"
        )
        evidence = evidence_by_segment.get(segment)
        handles = row.get("visible_handle_ids")
        binding_receipts = row.get("visible_binding_receipt_sha256s")
        exact_text_sha = require_sha256(
            row.get("exact_text_sha256"), "semantic classified exact text"
        )
        _require(
            type(evidence) is dict
            and row.get("cell_id") == evidence.get("cell_id")
            and row.get("residual_evidence_receipt_sha256")
            == evidence.get("receipt_sha256")
            and row.get("residual_binding_receipt_sha256")
            == evidence.get("citation_binding_receipt_sha256")
            and exact_text_sha == evidence.get("quote_sha256")
            and type(handles) is list
            and bool(handles)
            and len(handles) == len(set(handles))
            and set(handles) <= set(allowed)
            and type(binding_receipts) is list
            and len(binding_receipts) == len(handles)
            and len(binding_receipts) == len(set(binding_receipts))
            and (tuple(handles), exact_text_sha) in visible_items,
            "semantic retained segment is not byte-identically provider-visible",
        )
        for value in binding_receipts:
            require_sha256(value, "semantic visible binding")
        residual_item = require_sha256(
            row.get("residual_item_receipt_sha256"), "semantic residual typed item"
        )
        visible_item = require_sha256(
            row.get("visible_item_receipt_sha256"), "semantic visible typed item"
        )
        disposition = row.get("disposition")
        exclusion_sha = row.get("dedup_exclusion_sha256")
        if disposition == "residual_visible":
            _require(
                handles and len(handles) == 1
                and binding_receipts
                == [row.get("residual_binding_receipt_sha256")]
                and visible_item == residual_item
                and exclusion_sha is None,
                "semantic direct residual visibility changed",
            )
        else:
            _require(
                disposition == "protected_visible_exact_duplicate"
                and type(exclusion_sha) is str
                and exclusion_sha
                in composition_contract["dedup_exclusion_by_sha256"],
                "semantic protected duplicate lacks a selected-then-dedup proof",
            )
            exclusion = composition_contract["dedup_exclusion_by_sha256"][
                exclusion_sha
            ]
            _require(
                exclusion.get("duplicate_item_receipt_sha256") == residual_item
                and exclusion.get("duplicate_binding_receipt_sha256s")
                == [row.get("residual_binding_receipt_sha256")]
                and exclusion.get("owner_item_receipt_sha256") == visible_item
                and exclusion.get("owner_binding_receipt_sha256s")
                == binding_receipts,
                "semantic protected duplicate owner proof changed",
            )
        _require(
            not seen_handles & set(handles),
            "semantic classified closure repeats a visible owner handle",
        )
        seen_handles.update(handles)
        observed_segments.append(segment)
    _require(
        observed_segments == retained_segments,
        "semantic classified closure changed retained segment order",
    )
    return dict(closure)


def _semantic_source_plan(
    raw: Mapping[str, Any],
    ordinal: int,
) -> dict[str, Any]:
    _require(
        set(raw) == _SEMANTIC_SOURCE_FIELDS
        and raw.get("mode") == SEMANTIC_MODE
        and raw.get("fallback_reason") == "none"
        and raw.get("new_provider_calls") == 0
        and raw.get("retained_transformer_token_state_bytes") == 0,
        f"semantic residual source schema changed at ordinal {ordinal}",
    )
    parent = specialist_v1._verified_parent(raw, ordinal)  # noqa: SLF001
    common = _semantic_common_plan(raw, ordinal, parent=parent)
    fitted = raw.get("fitted_typed_prompt")
    terminal = raw.get("terminal_prompt")
    _require(
        type(fitted) is dict
        and set(fitted) == _SEMANTIC_FITTED_FIELDS
        and type(terminal) is dict
        and set(terminal) == _SEMANTIC_TERMINAL_FIELDS,
        f"semantic fitted/terminal schema changed at ordinal {ordinal}",
    )
    assert type(fitted) is dict and type(terminal) is dict
    provider = fitted.get("provider_input")
    terminal_provider = terminal.get("provider_input")
    allowed = fitted.get("allowed_handle_ids")
    groups = fitted.get("handle_group_by_id")
    story = fitted.get("story_coherence")
    preservation = fitted.get("preservation_requirements")
    validation = fitted.get("validation_contract")
    _require(
        type(provider) is dict
        and set(provider) == _SEMANTIC_PROVIDER_FIELDS
        and terminal_provider == provider
        and type(allowed) is list
        and bool(allowed)
        and len(allowed) == len(set(allowed))
        and type(groups) is dict
        and type(story) is dict
        and type(preservation) is dict
        and type(validation) is dict,
        f"semantic fitted prompt changed at ordinal {ordinal}",
    )
    assert (
        type(provider) is dict
        and type(allowed) is list
        and type(groups) is dict
        and type(story) is dict
        and type(preservation) is dict
        and type(validation) is dict
    )
    assert_gold_blind(provider, path=f"semantic_final_provider_{ordinal}")
    evidence_handles, evidence_groups = _semantic_evidence_handles(provider)
    residual_contract = _semantic_residual_contract(raw, require_closed=True)
    composition_contract = _composition_contract(raw)
    parent_fallback = provider.get("protected_parent_fallback")
    dated_question = require_text(provider.get("dated_question"), "semantic dated question")
    _require(
        provider.get("format") == typed_final.PROMPT_ROW_FORMAT
        and provider.get("story_coherence") == story
        and type(parent_fallback) is dict
        and parent_fallback.get("prediction") == common["parent_prediction"]
        and parent_fallback.get("prediction_sha256")
        == common["parent_prediction_sha256"]
        and tuple(allowed) == evidence_handles
        and groups == evidence_groups
        and set(validation.get("by_handle", {})) == set(allowed)
        and common["dated_question_sha256"] == quote_sha256(dated_question),
        f"semantic survivors escaped their parent or parser scope at ordinal {ordinal}",
    )
    messages = _plain_messages(typed_final.render_final_messages(provider))
    messages_sha = identity_sha256(list(messages))
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    fitted_receipt = require_sha256(
        fitted.get("receipt_sha256"), "semantic fitted prompt"
    )
    classified_closure = _classified_segment_closure(
        raw,
        allowed=allowed,
        provider_input=provider,
        fitted_receipt=fitted_receipt,
        residual_contract=residual_contract,
        composition_contract=composition_contract,
    )
    rendered_bytes = json.dumps(
        list(messages),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    terminal_receipt_body = {
        "fitted_prompt_receipt_sha256": fitted_receipt,
        "messages_sha256": messages_sha,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "prompt_token_proxy": prompt_tokens,
        "provider_input_sha256": identity_sha256(provider),
        "rendered_messages_utf8_byte_count": len(rendered_bytes),
        "rendered_messages_utf8_sha256": hashlib.sha256(rendered_bytes).hexdigest(),
    }
    terminal_receipt = require_sha256(
        terminal.get("terminal_prompt_receipt_sha256"),
        "semantic terminal prompt",
    )
    _require(
        terminal.get("fitted_prompt_receipt_sha256") == fitted_receipt
        and terminal.get("messages") == list(messages)
        and terminal.get("messages_sha256") == messages_sha
        and terminal.get("prompt_token_proxy") == prompt_tokens
        and terminal.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and terminal.get("hard_prompt_token_cap")
        == HARD_COMPLETE_CHAT_TOKEN_CAP
        and terminal.get("full_chat_plus_output_tokens")
        == prompt_tokens + OUTPUT_TOKEN_RESERVE
        and terminal.get("provider_prompt_count") == 0
        and terminal.get("rendered_messages_utf8_byte_count")
        == len(rendered_bytes)
        and terminal.get("rendered_messages_utf8_sha256")
        == hashlib.sha256(rendered_bytes).hexdigest()
        and terminal.get("retained_transformer_token_state_bytes") == 0
        and terminal_receipt == identity_sha256(terminal_receipt_body)
        and fitted.get("format") == typed_final.PROMPT_ROW_FORMAT
        and fitted.get("messages_sha256") == messages_sha
        and fitted.get("prompt_token_proxy") == prompt_tokens
        and fitted.get("full_chat_plus_output_tokens")
        == prompt_tokens + OUTPUT_TOKEN_RESERVE
        and fitted.get("hard_prompt_token_cap") == HARD_COMPLETE_CHAT_TOKEN_CAP
        and fitted.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and fitted.get("retained_transformer_token_state_bytes") == 0
        and fitted.get("protection_source_receipt_sha256")
        == classified_closure.get("protection_source_receipt_sha256")
        and prompt_tokens <= MAX_CHAT_PROMPT_TOKENS,
        f"semantic terminal prompt seal or hard budget changed at ordinal {ordinal}",
    )
    _semantic_parser_contract(
        parent=common["parent_prediction"],
        allowed=allowed,
        groups=groups,
        story=story,
        preservation=preservation,
        validation=validation,
    )
    body = {
        **common,
        "additive_composition_local_audit_sha256": composition_contract[
            "audit_sha256"
        ],
        "additive_composition_receipt_sha256": composition_contract[
            "composition_receipt_sha256"
        ],
        "allowed_handle_ids": list(allowed),
        "classified_closure": classified_closure,
        "fitted_prompt_receipt_sha256": fitted_receipt,
        "handle_group_by_id": dict(groups),
        "messages": list(messages),
        "messages_sha256": messages_sha,
        "namespace_id": require_text(raw.get("namespace_id"), "semantic namespace"),
        "preservation_requirements": dict(preservation),
        "prompt_token_proxy": prompt_tokens,
        "provider_input": dict(provider),
        "query_vector_artifact_sha256": require_sha256(
            raw.get("query_vector_artifact_sha256"), "semantic query vectors"
        ),
        "query_vector_row_receipt_sha256": require_sha256(
            raw.get("query_vector_row_receipt_sha256"), "semantic query-vector row"
        ),
        "semantic_query_receipt_sha256": residual_contract[
            "query_receipt_sha256"
        ],
        "semantic_residual_index_receipt_sha256": require_sha256(
            raw.get("semantic_residual_index_receipt_sha256"),
            "semantic residual index",
        ),
        "semantic_residual_local_audit_sha256": residual_contract["audit_sha256"],
        "semantic_residual_search_receipt_sha256": residual_contract[
            "search_receipt_sha256"
        ],
        "story_coherence": dict(story),
        "terminal_prompt_receipt_sha256": terminal_receipt,
        "validation_contract": dict(validation),
    }
    result = {**body, "answer_plan_receipt_sha256": identity_sha256(body)}
    assert_gold_blind(result, path=f"semantic_final_plan_{ordinal}")
    return result


def _semantic_fallback_source_plan(
    raw: Mapping[str, Any], ordinal: int
) -> dict[str, Any]:
    reason = raw.get("fallback_reason")
    _require(
        set(raw) == _SEMANTIC_SOURCE_FIELDS
        and raw.get("mode") == PARENT_PASSTHROUGH_MODE
        and reason
        in {
            "retained_unknowns_exceed_payload_cap",
            "no_novel_semantic_evidence",
            "protected_semantic_residual_exceeds_terminal_cap",
        }
        and raw.get("new_provider_calls") == 0
        and raw.get("retained_transformer_token_state_bytes") == 0
        and all(
            raw.get(key) is None
            for key in (
                "additive_composition",
                "additive_composition_local_audit",
                "classified_closure",
                "fitted_typed_prompt",
                "terminal_prompt",
            )
        ),
        f"semantic fallback source schema changed at ordinal {ordinal}",
    )
    parent = specialist_v1._verified_parent(raw, ordinal)  # noqa: SLF001
    common = specialist_v1._common_plan(  # noqa: SLF001
        raw, ordinal, parent=parent
    )
    residual_contract = _semantic_residual_contract(raw, require_closed=False)
    search = raw["semantic_residual_search"]
    frontier = residual_contract["frontier"]
    evidence = residual_contract["evidence_by_segment"]
    _require(
        (
            reason == "retained_unknowns_exceed_payload_cap"
            and search.get("fallback_required") is True
            and search.get("fallback_reason")
            == "retained_unknowns_exceed_payload_cap"
            and frontier.get("closed") is False
        )
        or (
            reason == "no_novel_semantic_evidence"
            and search.get("fallback_required") is False
            and not evidence
            and not frontier.get("retained_segment_receipt_sha256s")
        )
        or (
            reason == "protected_semantic_residual_exceeds_terminal_cap"
            and search.get("fallback_required") is False
            and frontier.get("closed") is True
            and bool(evidence)
        ),
        f"semantic fallback reason lost its residual proof at ordinal {ordinal}",
    )
    body = {
        **common,
        "fallback_reason": reason,
        "namespace_id": require_text(raw.get("namespace_id"), "semantic namespace"),
        "query_vector_artifact_sha256": require_sha256(
            raw.get("query_vector_artifact_sha256"), "semantic query vectors"
        ),
        "query_vector_row_receipt_sha256": require_sha256(
            raw.get("query_vector_row_receipt_sha256"), "semantic query-vector row"
        ),
        "semantic_query_receipt_sha256": residual_contract[
            "query_receipt_sha256"
        ],
        "semantic_residual_index_receipt_sha256": require_sha256(
            raw.get("semantic_residual_index_receipt_sha256"),
            "semantic residual index",
        ),
        "semantic_residual_local_audit_sha256": residual_contract["audit_sha256"],
        "semantic_residual_search_receipt_sha256": residual_contract[
            "search_receipt_sha256"
        ],
    }
    result = {**body, "answer_plan_receipt_sha256": identity_sha256(body)}
    assert_gold_blind(result, path=f"semantic_fallback_plan_{ordinal}")
    return result


def _source_plan(raw: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    source_body = dict(raw)
    declared = source_body.pop("question_receipt_sha256", None)
    _require(
        raw.get("ordinal") == ordinal
        and raw.get("mode") in _MODES
        and declared == identity_sha256(source_body),
        f"combined construction question seal/order changed at ordinal {ordinal}",
    )
    if raw.get("mode") == SEMANTIC_MODE:
        return _semantic_source_plan(raw, ordinal)
    if set(raw) == _SEMANTIC_SOURCE_FIELDS:
        return _semantic_fallback_source_plan(raw, ordinal)
    try:
        return specialist_v1._source_plan(raw, ordinal)  # noqa: SLF001
    except MatchedEvalContractError as exc:
        raise LockedSemanticFinalAnswerError(
            f"combined specialist/passthrough row changed at ordinal {ordinal}: {exc}"
        ) from exc


def _validate_semantic_plan(raw: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        raw.get("mode") == SEMANTIC_MODE
        and set(raw) == _SEMANTIC_PLAN_FIELDS,
        f"sealed semantic answer plan schema changed at ordinal {raw.get('ordinal')}",
    )
    body = dict(raw)
    declared = body.pop("answer_plan_receipt_sha256", None)
    ordinal = raw.get("ordinal")
    _require(
        type(ordinal) is int
        and 0 <= ordinal < EXPECTED_QUESTION_COUNT
        and declared == identity_sha256(body)
        and raw.get("parent_prediction_sha256")
        == quote_sha256(require_text(raw.get("parent_prediction"), "semantic parent")),
        f"sealed semantic answer plan receipt changed at ordinal {ordinal}",
    )
    provider = raw.get("provider_input")
    allowed = raw.get("allowed_handle_ids")
    groups = raw.get("handle_group_by_id")
    story = raw.get("story_coherence")
    preservation = raw.get("preservation_requirements")
    validation = raw.get("validation_contract")
    classified_closure = raw.get("classified_closure")
    _require(
        type(provider) is dict
        and set(provider) == _SEMANTIC_PROVIDER_FIELDS
        and type(allowed) is list
        and type(groups) is dict
        and type(story) is dict
        and type(preservation) is dict
        and type(validation) is dict,
        f"sealed semantic parser inputs changed at ordinal {ordinal}",
    )
    assert (
        type(provider) is dict
        and type(allowed) is list
        and type(groups) is dict
        and type(story) is dict
        and type(preservation) is dict
        and type(validation) is dict
    )
    handles, evidence_groups = _semantic_evidence_handles(provider)
    _require(
        type(classified_closure) is dict
        and set(classified_closure) == _CLASSIFIED_CLOSURE_FIELDS,
        "sealed semantic classified closure changed schema",
    )
    assert type(classified_closure) is dict
    closure_body = dict(classified_closure)
    closure_receipt = closure_body.pop("receipt_sha256", None)
    _require(
        closure_receipt == identity_sha256(closure_body)
        and classified_closure.get("closed") is True
        and classified_closure.get("complete_leaf_partition") is True
        and classified_closure.get("all_retained_segments_provider_visible") is True
        and classified_closure.get("terminal_allowed_handle_ids") == allowed
        and classified_closure.get("semantic_residual_search_receipt_sha256")
        == raw.get("semantic_residual_search_receipt_sha256")
        and classified_closure.get("fitted_prompt_receipt_sha256")
        == raw.get("fitted_prompt_receipt_sha256"),
        "sealed semantic classified closure changed",
    )
    typed = provider.get("typed_evidence")
    assert type(typed) is dict
    provider_items = {
        (tuple(item["handle_ids"]), quote_sha256(item["summary"]))
        for item in typed.get("items", [])
        if type(item) is dict
        and type(item.get("handle_ids")) is list
        and type(item.get("summary")) is str
    }
    closure_rows = classified_closure.get("rows")
    _require(type(closure_rows) is list, "sealed semantic closure rows changed type")
    for row in closure_rows:
        _require(
            type(row) is dict
            and set(row) == _CLASSIFIED_CLOSURE_ROW_FIELDS
            and (
                tuple(row.get("visible_handle_ids", ())),
                row.get("exact_text_sha256"),
            )
            in provider_items,
            "sealed semantic closure lost provider-visible exact segment text",
        )
    messages = _plain_messages(typed_final.render_final_messages(provider))
    _require(
        tuple(allowed) == handles
        and groups == evidence_groups
        and raw.get("messages") == list(messages)
        and raw.get("messages_sha256") == identity_sha256(list(messages))
        and raw.get("prompt_token_proxy") == count_chat_prompt_token_proxy(messages)
        and raw["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE
        <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        f"sealed semantic survivor prompt changed at ordinal {ordinal}",
    )
    _semantic_parser_contract(
        parent=str(raw["parent_prediction"]),
        allowed=allowed,
        groups=groups,
        story=story,
        preservation=preservation,
        validation=validation,
    )
    assert_gold_blind(raw, path=f"loaded_semantic_final_plan_{ordinal}")
    return dict(raw)


def _validate_semantic_fallback_plan(raw: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        raw.get("mode") == PARENT_PASSTHROUGH_MODE
        and set(raw) == _SEMANTIC_FALLBACK_PLAN_FIELDS,
        f"sealed semantic fallback schema changed at ordinal {raw.get('ordinal')}",
    )
    body = dict(raw)
    declared = body.pop("answer_plan_receipt_sha256", None)
    _require(
        declared == identity_sha256(body)
        and raw.get("fallback_reason")
        in {
            "retained_unknowns_exceed_payload_cap",
            "no_novel_semantic_evidence",
            "protected_semantic_residual_exceeds_terminal_cap",
        }
        and raw.get("parent_prediction_sha256")
        == quote_sha256(require_text(raw.get("parent_prediction"), "fallback parent")),
        f"sealed semantic fallback receipt changed at ordinal {raw.get('ordinal')}",
    )
    assert_gold_blind(raw, path="loaded_semantic_fallback_plan")
    return dict(raw)


def _validate_plan(raw: Mapping[str, Any]) -> dict[str, Any]:
    if raw.get("mode") == SEMANTIC_MODE:
        return _validate_semantic_plan(raw)
    if set(raw) == _SEMANTIC_FALLBACK_PLAN_FIELDS:
        return _validate_semantic_fallback_plan(raw)
    try:
        return specialist_v1._validate_stored_plan(raw)  # noqa: SLF001
    except MatchedEvalContractError as exc:
        raise LockedSemanticFinalAnswerError(
            f"sealed combined answer plan changed at ordinal {raw.get('ordinal')}: {exc}"
        ) from exc


def load_answer_plans(
    path: str | Path,
    expected_sha256: str,
    *,
    construction_loader: ConstructionLoader | None = None,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    """Load the combined construction and derive its sealed answer plans."""

    expected = require_sha256(expected_sha256, "expected semantic construction")
    loader = construction_loader or _default_construction_loader
    artifact, raw_rows = loader(Path(path), expected_sha256=expected)
    _require(
        isinstance(artifact, SealedArtifact)
        and artifact.sha256 == expected
        and isinstance(raw_rows, (tuple, list))
        and len(raw_rows) == EXPECTED_QUESTION_COUNT,
        "combined semantic construction digest or population changed",
    )
    plans = tuple(
        _source_plan(raw, ordinal)
        for ordinal, raw in enumerate(raw_rows)
        if isinstance(raw, Mapping)
    )
    _require(
        len(plans) == EXPECTED_QUESTION_COUNT
        and tuple(row["ordinal"] for row in plans)
        == tuple(range(EXPECTED_QUESTION_COUNT))
        and len({row["question_id"] for row in plans}) == EXPECTED_QUESTION_COUNT,
        "combined semantic construction identities changed",
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
    _require(model == DEFAULT_MODEL, "combined semantic answer model must be Terra")
    require_text(gateway_url, "combined semantic Terra gateway")
    _require(
        type(max_concurrency) is int and max_concurrency > 0,
        "combined semantic concurrency must be positive",
    )
    validated = tuple(_validate_plan(row) for row in plans)
    _require(
        len(validated) == EXPECTED_QUESTION_COUNT
        and tuple(row["ordinal"] for row in validated)
        == tuple(range(EXPECTED_QUESTION_COUNT)),
        "combined semantic answer plan population changed",
    )
    physical = tuple(
        row for row in validated if row["mode"] in {SPECIALIST_MODE, SEMANTIC_MODE}
    )
    passthrough = tuple(
        row for row in validated if row["mode"] == PARENT_PASSTHROUGH_MODE
    )
    specialist = sum(row["mode"] == SPECIALIST_MODE for row in physical)
    semantic = sum(row["mode"] == SEMANTIC_MODE for row in physical)
    _require(bool(physical), "combined semantic provider population is empty")
    prompts = tuple(_plain_messages(row["messages"]) for row in physical)
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == len(physical),
        "combined semantic prompts must be unique",
    )
    observed_max = max(
        row["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE for row in physical
    )
    _require(
        observed_max <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "combined semantic complete prompt envelope exceeds 8k",
    )
    payload = {
        "answer_plan_population_sha256": identity_sha256(
            [row["answer_plan_receipt_sha256"] for row in validated]
        ),
        "construction_artifact_sha256": construction.sha256,
        "format": PREFLIGHT_FORMAT,
        "gateway_url": gateway_url,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "max_chat_prompt_tokens": MAX_CHAT_PROMPT_TOKENS,
        "max_concurrency": max_concurrency,
        "model": model,
        "observed_max_complete_envelope_tokens": observed_max,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "parent_passthrough_count": len(passthrough),
        "parent_passthrough_rows": list(passthrough),
        "physical_prompt_rows": list(physical),
        "prompt_population": population.model_dump(),
        "prompt_population_sha256": population.prompt_population_sha256,
        "provider_calls": 0,
        "question_count": EXPECTED_QUESTION_COUNT,
        "required_authorized_provider_calls": len(physical),
        "retained_transformer_token_state_bytes": 0,
        "scoped_completion_format": SCOPED_COMPLETION_FORMAT,
        "semantic_question_count": semantic,
        "semantic_renderer_format": SEMANTIC_PROMPT_FORMAT,
        "specialist_question_count": specialist,
    }
    assert_gold_blind(payload, path="combined_semantic_answer_preflight")
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
        "question_count": EXPECTED_QUESTION_COUNT,
        "required_authorized_provider_calls": payload[
            "required_authorized_provider_calls"
        ],
        "semantic_question_count": payload["semantic_question_count"],
        "specialist_question_count": payload["specialist_question_count"],
    }


def _validate_preflight(
    artifact: SealedArtifact,
) -> tuple[
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    payload = artifact.payload
    assert_gold_blind(payload, path="loaded_combined_semantic_answer_preflight")
    physical = payload.get("physical_prompt_rows")
    passthrough = payload.get("parent_passthrough_rows")
    _require(
        set(payload) == _PREFLIGHT_FIELDS
        and payload.get("format") == PREFLIGHT_FORMAT
        and payload.get("model") == DEFAULT_MODEL
        and payload.get("gold_loaded") is False
        and payload.get("provider_calls") == 0
        and payload.get("hard_complete_chat_token_cap")
        == HARD_COMPLETE_CHAT_TOKEN_CAP
        and payload.get("max_chat_prompt_tokens") == MAX_CHAT_PROMPT_TOKENS
        and payload.get("output_token_reserve") == OUTPUT_TOKEN_RESERVE
        and payload.get("question_count") == EXPECTED_QUESTION_COUNT
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("scoped_completion_format") == SCOPED_COMPLETION_FORMAT
        and payload.get("semantic_renderer_format") == SEMANTIC_PROMPT_FORMAT
        and type(physical) is list
        and bool(physical)
        and type(passthrough) is list
        and len(physical) == payload.get("required_authorized_provider_calls")
        and len(passthrough) == payload.get("parent_passthrough_count")
        and len(physical) + len(passthrough) == EXPECTED_QUESTION_COUNT,
        "sealed combined semantic answer preflight changed",
    )
    require_sha256(
        payload.get("construction_artifact_sha256"),
        "combined semantic preflight construction",
    )
    validated_physical = tuple(_validate_plan(row) for row in physical)
    validated_passthrough = tuple(_validate_plan(row) for row in passthrough)
    _require(
        all(row["mode"] in {SPECIALIST_MODE, SEMANTIC_MODE} for row in validated_physical)
        and all(
            row["mode"] == PARENT_PASSTHROUGH_MODE
            for row in validated_passthrough
        )
        and sum(row["mode"] == SPECIALIST_MODE for row in validated_physical)
        == payload.get("specialist_question_count")
        and sum(row["mode"] == SEMANTIC_MODE for row in validated_physical)
        == payload.get("semantic_question_count"),
        "combined semantic preflight modes changed",
    )
    ordered = tuple(
        sorted((*validated_physical, *validated_passthrough), key=lambda row: row["ordinal"])
    )
    _require(
        tuple(row["ordinal"] for row in ordered)
        == tuple(range(EXPECTED_QUESTION_COUNT))
        and len({row["question_id"] for row in ordered}) == EXPECTED_QUESTION_COUNT
        and payload.get("answer_plan_population_sha256")
        == identity_sha256(
            [row["answer_plan_receipt_sha256"] for row in ordered]
        ),
        "sealed combined semantic answer plan population changed",
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
        "sealed combined semantic prompt population changed",
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
        == require_sha256(expected_sha256, "expected combined semantic preflight"),
        "combined semantic preflight digest changed",
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
        "runtime settings differ from sealed combined semantic preflight",
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
            "arm": "locked_semantic_final_terra_answer_v2",
            "authorized_unique_calls": required,
            "construction_artifact_sha256": artifact.payload[
                "construction_artifact_sha256"
            ],
            "experiment_format": FORMAT,
            "gateway_url": gateway_url,
            "gold_loaded": False,
            "preflight_artifact_sha256": artifact.sha256,
            "scoped_completion_format": SCOPED_COMPLETION_FORMAT,
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
        "combined semantic Terra population changed",
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
    proof_kind: str | None = None,
    proof_receipt_sha256: str | None = None,
    prompt_row_receipt_sha256: str | None = None,
    request_journal_sha256: str | None = None,
    response_journal_sha256: str | None = None,
    solver_valid: bool | None = None,
    specialist_scope_receipt_sha256: str | None = None,
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
        "proof_kind": proof_kind,
        "proof_receipt_sha256": proof_receipt_sha256,
        "prompt_row_receipt_sha256": prompt_row_receipt_sha256,
        "question_id": plan["question_id"],
        "question_sha256": plan["question_sha256"],
        "request_journal_sha256": request_journal_sha256,
        "response_journal_sha256": response_journal_sha256,
        "retained_transformer_token_state_bytes": 0,
        "route_id": plan["route_id"],
        "solver_valid": solver_valid,
        "specialist_scope_receipt_sha256": specialist_scope_receipt_sha256,
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
        "materialization requires every combined checkpoint and no provider calls",
    )
    physical = tuple(
        row for row in plans if row["mode"] in {SPECIALIST_MODE, SEMANTIC_MODE}
    )
    _require(
        len(plans) == EXPECTED_QUESTION_COUNT and len(physical) == required,
        "combined semantic materialization population changed",
    )
    records = {row.messages_sha256: row for row in batch.unique_records}
    completions = {
        plan["ordinal"]: completion
        for plan, completion in zip(physical, batch.logical_completions, strict=True)
    }
    _require(len(records) == required, "combined semantic completions repeat")
    results: list[dict[str, Any]] = []
    for plan in plans:
        parent = plan["parent_prediction"]
        if plan["mode"] == PARENT_PASSTHROUGH_MODE:
            body = _result_body(
                plan,
                prediction=parent,
                prediction_source="locked_semantic_parent_passthrough_v2",
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
                f"combined checkpoint changed at ordinal {plan['ordinal']}",
            )
            assert record is not None
            if plan["mode"] == SPECIALIST_MODE:
                _prompt, scope = specialist_v1._scoped_prompt_and_scope(plan)  # noqa: SLF001
                parsed = parse_specialist_scoped_completion(
                    completion,
                    parent_prediction=parent,
                    scope=scope,
                )
                valid_replace = parsed.valid and parsed.decision == "replace"
                prediction = parsed.prediction if valid_replace else parent
                if valid_replace:
                    source = "locked_semantic_specialist_validated_replacement_v2"
                    decision = "replace"
                    used = parsed.used_handle_ids
                elif parsed.valid:
                    source = "locked_semantic_specialist_validated_keep_parent_v2"
                    decision = "keep_parent"
                    used = ()
                else:
                    source = "locked_semantic_specialist_invalid_keep_parent_v2"
                    decision = "invalid_keep_parent"
                    used = ()
                body = _result_body(
                    plan,
                    prediction=prediction,
                    prediction_source=source,
                    decision=decision,
                    completion_parser="specialist_scoped_v3",
                    call_key_sha256=record.call_key_sha256,
                    completion_receipt_sha256=record.completion_sha256,
                    parse_error_code=parsed.error_code,
                    parse_receipt_sha256=parsed.receipt_sha256,
                    proof_kind=parsed.proof_kind,
                    proof_receipt_sha256=parsed.proof_receipt_sha256,
                    prompt_row_receipt_sha256=plan["answer_plan_receipt_sha256"],
                    request_journal_sha256=record.request_journal_sha256,
                    response_journal_sha256=record.response_journal_sha256,
                    solver_valid=parsed.valid,
                    specialist_scope_receipt_sha256=parsed.scope_receipt_sha256,
                    used_handle_ids=used,
                    validation_basis=parsed.validation_basis,
                )
            else:
                parsed = typed_final.parse_typed_final_completion(
                    completion,
                    parent_prediction=parent,
                    allowed_handle_ids=tuple(plan["allowed_handle_ids"]),
                    handle_group_by_id=dict(plan["handle_group_by_id"]),
                    story_coherence=dict(plan["story_coherence"]),
                    preservation_requirements=dict(
                        plan["preservation_requirements"]
                    ),
                    validation_contract=dict(plan["validation_contract"]),
                )
                valid_replace = parsed.valid and parsed.decision == "replace"
                prediction = parsed.prediction if valid_replace else parent
                if valid_replace:
                    source = "locked_semantic_search_validated_replacement_v2"
                    decision = "replace"
                    used = parsed.used_handle_ids
                elif parsed.valid:
                    source = "locked_semantic_search_validated_keep_parent_v2"
                    decision = "keep_parent"
                    used = ()
                else:
                    source = "locked_semantic_search_invalid_keep_parent_v2"
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
        tuple(row["ordinal"] for row in results)
        == tuple(range(EXPECTED_QUESTION_COUNT)),
        "combined semantic result order changed",
    )
    payload = {
        "changed_prediction_count": sum(row["changed_from_parent"] for row in results),
        "completion_batch": _stable_batch(batch),
        "construction_artifact_sha256": preflight.payload[
            "construction_artifact_sha256"
        ],
        "format": FORMAT,
        "gold_loaded": False,
        "invalid_completion_parent_fallback_count": sum(
            row["prediction_source"].endswith("invalid_keep_parent_v2")
            for row in results
        ),
        "judge_rows": [typed_final.judge_row_projection(row) for row in results],
        "model": DEFAULT_MODEL,
        "parent_passthrough_count": sum(
            row["answer_mode"] == PARENT_PASSTHROUGH_MODE for row in results
        ),
        "physical_provider_calls_during_materialization": 0,
        "preflight_artifact_sha256": preflight.sha256,
        "question_count": EXPECTED_QUESTION_COUNT,
        "questions": results,
        "required_authorized_provider_calls": required,
        "retained_transformer_token_state_bytes": 0,
        "scoped_completion_format": SCOPED_COMPLETION_FORMAT,
        "semantic_question_count": sum(
            row["answer_mode"] == SEMANTIC_MODE for row in results
        ),
        "semantic_renderer_format": SEMANTIC_PROMPT_FORMAT,
        "specialist_question_count": sum(
            row["answer_mode"] == SPECIALIST_MODE for row in results
        ),
        "validated_replacement_count": sum(
            row["prediction_source"].endswith("validated_replacement_v2")
            for row in results
        ),
    }
    assert_gold_blind(payload, path="locked_semantic_final_terra_answer_v2")
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
        "combined semantic construction/preflight binding changed",
    )
    batch = _checkpoint_batch(preflight, prompts, args=args, client=None)
    rebuilt = _materialization_projection(preflight, plans, batch)
    terminal = read_sealed_json(Path(args.output_root) / RUN_NAME)
    _require(
        terminal.sha256
        == require_sha256(args.expected_run_sha256, "expected combined semantic run")
        and terminal.payload == rebuilt,
        "combined semantic run differs from checkpoint-only replay",
    )
    replay, _created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME, terminal.payload
    )
    _require(replay.sha256 == terminal.sha256, "combined semantic replay is not byte-identical")
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

    preflight = commands.add_parser("preflight", help="seal combined v2 prompts")
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
        "materialize", help="materialize all 100 combined v2 answers"
    )
    _add_runtime_settings(materialize)
    materialize.add_argument("--expected-preflight-sha256", required=True)

    replay = commands.add_parser("replay", help="prove byte-identical v2 replay")
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
    "DEFAULT_CONSTRUCTION",
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT",
    "FORMAT",
    "LockedSemanticFinalAnswerError",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_NAME",
    "REPLAY_NAME",
    "RESULT_ROW_FORMAT",
    "RUN_NAME",
    "SEMANTIC_MODE",
    "SEMANTIC_PROMPT_FORMAT",
    "build_parser",
    "load_answer_plans",
    "main",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
]
