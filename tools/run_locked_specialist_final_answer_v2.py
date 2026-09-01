#!/usr/bin/env python3
"""Checkpoint the locked full-100 specialist/operator v2 answer population.

The sealed construction contains 100 rows: 69 ordinary specialist prompts,
three repaired operator prompts, and 28 byte-bound parent passthroughs.  This
module is a thin version adapter over the audited v1 full-population lifecycle.
It reuses v1 checkpoint journals, materialization, and replay; the only new
logic is authenticated construction loading, parser dispatch, the receipt-
bound q42 prompt transform, and strict refusal of invalid provider fallbacks.

Gold is never opened.  Exactly 72 unique Terra prompts are eligible for the
separately authorized provider phase, every complete envelope is at most 8,000
tokens, and no transformer token state is retained.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionRuntime,
)
from tools import run_locked_specialist_final_answer as base  # noqa: E402
from tools import run_locked_specialist_final_construction_v2 as construction_v2  # noqa: E402
from tools import run_reduced_missing4_v4_answer as repaired_v4  # noqa: E402
from tools import run_reduced_specialist_answer_v2 as generic_answer  # noqa: E402
from tools import run_reduced_specialist_retrieval_assay as terminal_builder  # noqa: E402
from tools.matched_eval.artifacts import SealedArtifact  # noqa: E402
from tools.matched_eval import local_temporal_pair as local_pair  # noqa: E402
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    judge_row_projection,
)
from tools.matched_eval.specialist_scoped_completion import (  # noqa: E402
    PROMPT_FORMAT as SPECIALIST_PROMPT_FORMAT,
    render_specialist_scoped_prompt,
)


FORMAT = "memory-condense-locked-specialist-final-terra-answer-v2"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight"
RESULT_ROW_FORMAT = f"{FORMAT}-result-row"
MIXED_PARSER_FORMAT = f"{FORMAT}-mixed-parser-v1"
ORDINARY_TYPED_PROMPT_TRANSFORM_FORMAT = (
    f"{FORMAT}-ordinary-typed-prompt-transform-v1"
)

PREFLIGHT_NAME = "locked-specialist-final-answer-preflight-v2.json"
RUN_NAME = "locked-specialist-final-answer-v2.json"
REPLAY_NAME = "locked-specialist-final-answer-replay-v2.json"
CHECKPOINT_DIR_NAME = "locked-specialist-final-answer-checkpoints-v2"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONSTRUCTION = construction_v2.DEFAULT_OUTPUT_ROOT / construction_v2.CONSTRUCTION_NAME
DEFAULT_OUTPUT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-specialist-final-answer-v2"
)
EXPECTED_CONSTRUCTION_SHA256 = (
    "663d3b34c463c5e28243b8408c17fa431ea7eb9d7720f61b46bb68ba862629fb"
)
DEFAULT_MODEL = base.DEFAULT_MODEL

EXPECTED_QUESTION_COUNT = 100
EXPECTED_SPECIALIST_COUNT = construction_v2.EXPECTED_SPECIALIST_COUNT
EXPECTED_REPAIRED_OPERATOR_COUNT = construction_v2.EXPECTED_REPAIRED_OPERATOR_COUNT
EXPECTED_PROVIDER_PROMPT_COUNT = construction_v2.EXPECTED_PROVIDER_PROMPT_COUNT
EXPECTED_PASSTHROUGH_COUNT = construction_v2.EXPECTED_PASSTHROUGH_COUNT
REPAIRED_OPERATOR_ORDINALS = tuple(construction_v2.REPAIRED_OPERATOR_ORDINALS)
ORDINARY_TYPED_ORDINALS = (3, 14, 18, 28, 32, 64, 68, 69, 75, 92, 97)
EXPECTED_SCOPED_SPECIALIST_COUNT = EXPECTED_SPECIALIST_COUNT - len(
    ORDINARY_TYPED_ORDINALS
)

SPECIALIST_MODE = base.SPECIALIST_MODE
PARENT_PASSTHROUGH_MODE = base.PARENT_PASSTHROUGH_MODE
REPAIRED_OPERATOR_MODE = "repaired_operator"
SPECIALIST_PARSER = "specialist_scoped"
ORDINARY_TYPED_PARSER = "ordinary_typed_final"
REPAIRED_OPERATOR_PARSER = "v4_repaired_operator"
PASSTHROUGH_PARSER = "parent_passthrough"

HARD_COMPLETE_CHAT_TOKEN_CAP = base.HARD_COMPLETE_CHAT_TOKEN_CAP
MAX_CHAT_PROMPT_TOKENS = base.MAX_CHAT_PROMPT_TOKENS
OUTPUT_TOKEN_RESERVE = base.OUTPUT_TOKEN_RESERVE

_PLAN_EXTRA_FIELDS = frozenset(
    {"adapter_prompt_transform", "answer_parser_kind", "construction_mode"}
)
_PASSTHROUGH_PLAN_FIELDS = base._COMMON_PLAN_FIELDS | _PLAN_EXTRA_FIELDS  # noqa: SLF001
_SCOPED_PLAN_FIELDS = base._SPECIALIST_PLAN_FIELDS | _PLAN_EXTRA_FIELDS  # noqa: SLF001
_REPAIRED_PLAN_FIELDS = base._COMMON_PLAN_FIELDS | _PLAN_EXTRA_FIELDS | frozenset(  # noqa: SLF001
    {
        "allowed_handle_ids",
        "handle_group_by_id",
        "messages",
        "messages_sha256",
        "preservation_requirements",
        "prompt_token_proxy",
        "story_coherence",
        "terminal_prompt_receipt_sha256",
        "validation_contract",
    }
)

_BASE_SOURCE_PLAN = base._source_plan  # noqa: SLF001
_BASE_VALIDATE_STORED_PLAN = base._validate_stored_plan  # noqa: SLF001
_BASE_SCOPED_PROMPT_AND_SCOPE = base._scoped_prompt_and_scope  # noqa: SLF001
_BASE_PARSE_SPECIALIST = base.parse_specialist_scoped_completion
_BASE_PREFLIGHT_PROJECTION = base._preflight_projection  # noqa: SLF001
_BASE_VALIDATE_PREFLIGHT = base._validate_preflight  # noqa: SLF001
_BASE_MATERIALIZATION_PROJECTION = base._materialization_projection  # noqa: SLF001


class LockedSpecialistFinalAnswerV2Error(MatchedEvalContractError):
    """Raised when the full-100 v2 answer contract changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSpecialistFinalAnswerV2Error(message)


@dataclass(frozen=True, slots=True)
class _RepairedOperatorScope:
    ordinal: int
    allowed_handle_ids: tuple[str, ...]
    handle_group_by_id: dict[str, str]
    story_coherence: dict[str, Any]
    preservation_requirements: dict[str, Any]
    validation_contract: dict[str, Any]
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class _OrdinaryTypedScope:
    ordinal: int
    allowed_handle_ids: tuple[str, ...]
    handle_group_by_id: dict[str, str]
    story_coherence: dict[str, Any]
    preservation_requirements: dict[str, Any]
    validation_contract: dict[str, Any]
    prompt_transform_receipt_sha256: str
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class _ScopedSpecialistV2Scope:
    ordinal: int
    base_scope: Any
    allowed_handle_ids: tuple[str, ...]
    provider_input: dict[str, Any]
    validation_contract: dict[str, Any]
    answer_plan_receipt_sha256: str
    receipt_sha256: str


def _default_construction_loader(
    path: Path,
    *,
    expected_sha256: str,
):
    return construction_v2.load_verified_construction(
        path,
        expected_sha256=expected_sha256,
    )


def _base_plan_without_v2_fields(raw: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        key: value
        for key, value in raw.items()
        if key not in _PLAN_EXTRA_FIELDS and key != "answer_plan_receipt_sha256"
    }
    return {**body, "answer_plan_receipt_sha256": identity_sha256(body)}


def _augment_base_plan(
    plan: Mapping[str, Any],
    *,
    construction_mode: str,
    parser_kind: str,
) -> dict[str, Any]:
    body = dict(plan)
    body.pop("answer_plan_receipt_sha256", None)
    body.update(
        {
            "adapter_prompt_transform": None,
            "answer_parser_kind": parser_kind,
            "construction_mode": construction_mode,
        }
    )
    result = {**body, "answer_plan_receipt_sha256": identity_sha256(body)}
    _validate_stored_plan(result)
    return result


def _repaired_scope_projection(
    *,
    plan: Mapping[str, Any],
    raw: Mapping[str, Any],
    ordinal: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, str]], int]:
    terminal = raw.get("terminal_prompt")
    _require(type(terminal) is dict, f"repaired terminal changed at {ordinal}")
    assert type(terminal) is dict
    provider_input = terminal.get("provider_input")
    _require(type(provider_input) is dict, f"repaired provider changed at {ordinal}")
    assert type(provider_input) is dict
    advisories = provider_input.get("specialist_advisories")
    _require(
        type(advisories) is list
        and len(advisories) == 1
        and type(advisories[0]) is dict,
        f"repaired advisory changed at {ordinal}",
    )
    terminal_receipt = require_sha256(
        terminal.get("terminal_prompt_receipt_sha256"),
        "repaired terminal prompt",
    )
    transform: dict[str, Any] | None = None
    messages = [dict(row) for row in plan["messages"]]
    prompt_tokens = int(plan["prompt_token_proxy"])
    if ordinal == 42:
        transform, transformed, prompt_tokens = repaired_v4._q42_prompt_transform(  # noqa: SLF001
            plan,
            advisories[0],
            terminal_receipt,
        )
        messages = [dict(row) for row in transformed]
    scope_body = {
        "adapter_prompt_transform": transform,
        "allowed_handle_ids": list(plan["allowed_handle_ids"]),
        "format": repaired_v4.ADVISORY_SCOPE_FORMAT,
        "ordinal": ordinal,
        "specialist_advisories": advisories,
        "specialist_advisories_sha256": identity_sha256(advisories),
        "terminal_kind": require_text(raw.get("terminal_kind"), "repaired terminal kind"),
        "terminal_prompt_receipt_sha256": terminal_receipt,
    }
    scope = {**scope_body, "receipt_sha256": identity_sha256(scope_body)}
    return scope, transform, messages, prompt_tokens


def _ordinary_typed_terminal_transform(
    raw: Mapping[str, Any],
    ordinal: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the scoped source terminal and seal a typed-final rerender.

    The provider input is preserved byte-for-byte.  Only the renderer changes;
    both the authoritative source terminal and the newly rendered target are
    independently receipt-bound.
    """

    terminal = raw.get("terminal_prompt")
    fitted = raw.get("fitted_typed_prompt")
    _require(
        type(terminal) is dict and type(fitted) is dict,
        f"ordinary typed terminal/fitted prompt changed at {ordinal}",
    )
    assert type(terminal) is dict and type(fitted) is dict
    provider_input = terminal.get("provider_input")
    fitted_input = fitted.get("provider_input")
    _require(
        type(provider_input) is dict and type(fitted_input) is dict,
        f"ordinary typed provider input changed at {ordinal}",
    )
    assert type(provider_input) is dict and type(fitted_input) is dict
    advisories = provider_input.get("specialist_advisories")
    _require(
        type(advisories) is list
        and provider_input == {**dict(fitted_input), "specialist_advisories": advisories},
        f"ordinary typed terminal escaped fitted evidence at {ordinal}",
    )
    fitted_receipt = require_sha256(
        fitted.get("receipt_sha256"), "ordinary typed fitted prompt"
    )
    source_terminal = terminal_builder._terminal_projection(  # noqa: SLF001
        provider_input=fitted_input,
        specialist_advisories=advisories,
        fitted_prompt_receipt_sha256=fitted_receipt,
        message_renderer_format=SPECIALIST_PROMPT_FORMAT,
        prompt_envelope_renderer=render_specialist_scoped_prompt,
    )
    _require(
        source_terminal == terminal,
        f"ordinary typed source terminal receipt changed at {ordinal}",
    )
    target_terminal = terminal_builder._terminal_projection(  # noqa: SLF001
        provider_input=fitted_input,
        specialist_advisories=advisories,
        fitted_prompt_receipt_sha256=fitted_receipt,
    )
    _require(
        target_terminal.get("provider_input") == provider_input
        and target_terminal.get("full_chat_plus_output_tokens", 8_001)
        <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        f"ordinary typed target escaped evidence or 8k at {ordinal}",
    )
    transform_body = {
        "fitted_prompt_receipt_sha256": fitted_receipt,
        "format": ORDINARY_TYPED_PROMPT_TRANSFORM_FORMAT,
        "ordinal": ordinal,
        "provider_input_sha256": identity_sha256(provider_input),
        "source_complete_envelope_tokens": terminal["full_chat_plus_output_tokens"],
        "source_message_renderer_format": terminal["message_renderer_format"],
        "source_messages_sha256": require_sha256(
            terminal.get("messages_sha256"), "ordinary typed source messages"
        ),
        "source_prompt_token_proxy": terminal["prompt_token_proxy"],
        "source_specialist_prompt_envelope_receipt_sha256": require_sha256(
            terminal.get("specialist_prompt_envelope_receipt_sha256"),
            "ordinary typed source specialist envelope",
        ),
        "source_terminal_prompt_receipt_sha256": require_sha256(
            terminal.get("terminal_prompt_receipt_sha256"),
            "ordinary typed source terminal",
        ),
        "specialist_advisories_sha256": require_sha256(
            terminal.get("specialist_advisories_sha256"),
            "ordinary typed advisories",
        ),
        "target_complete_envelope_tokens": target_terminal[
            "full_chat_plus_output_tokens"
        ],
        "target_message_renderer_format": generic_answer.FORMAT,
        "target_messages_sha256": require_sha256(
            target_terminal.get("messages_sha256"), "ordinary typed target messages"
        ),
        "target_prompt_token_proxy": target_terminal["prompt_token_proxy"],
        "target_terminal_prompt_receipt_sha256": require_sha256(
            target_terminal.get("terminal_prompt_receipt_sha256"),
            "ordinary typed target terminal",
        ),
        "transform": "rerender_identical_provider_input_as_ordinary_typed_final",
    }
    transform = {
        **transform_body,
        "receipt_sha256": identity_sha256(transform_body),
    }
    return target_terminal, transform


def _ordinary_typed_source_plan(
    raw: Mapping[str, Any], ordinal: int
) -> dict[str, Any]:
    source_body = dict(raw)
    declared = source_body.pop("question_receipt_sha256", None)
    _require(
        ordinal in ORDINARY_TYPED_ORDINALS
        and raw.get("ordinal") == ordinal
        and declared == identity_sha256(source_body),
        f"ordinary typed construction row seal changed at {ordinal}",
    )
    parent = base._verified_parent(raw, ordinal)  # noqa: SLF001
    common = base._common_plan(raw, ordinal, parent=parent)  # noqa: SLF001
    target_terminal, transform = _ordinary_typed_terminal_transform(raw, ordinal)
    synthetic_body = {**source_body, "terminal_prompt": target_terminal}
    synthetic = {
        **synthetic_body,
        "question_receipt_sha256": identity_sha256(synthetic_body),
    }
    try:
        generic = generic_answer._prompt_plan_row(synthetic, ordinal)  # noqa: SLF001
    except MatchedEvalContractError as exc:
        raise LockedSpecialistFinalAnswerV2Error(
            f"ordinary typed target prompt changed at {ordinal}: {exc}"
        ) from exc
    _require(
        generic["parent_prediction"] == common["parent_prediction"]
        and generic["parent_prediction_sha256"]
        == common["parent_prediction_sha256"]
        and generic["question_id"] == common["question_id"]
        and generic["question_sha256"] == common["question_sha256"]
        and generic["dated_question_sha256"] == common["dated_question_sha256"],
        f"ordinary typed target escaped its parent at {ordinal}",
    )
    provider_fields = {
        key: value
        for key, value in generic.items()
        if key
        not in {
            "construction_question_receipt_sha256",
            "dated_question_sha256",
            "ordinal",
            "parent_prediction",
            "parent_prediction_sha256",
            "prompt_row_receipt_sha256",
            "question_id",
            "question_sha256",
        }
    }
    body = {
        **common,
        **provider_fields,
        "adapter_prompt_transform": transform,
        "answer_parser_kind": ORDINARY_TYPED_PARSER,
        "construction_mode": SPECIALIST_MODE,
        "terminal_prompt_receipt_sha256": transform[
            "source_terminal_prompt_receipt_sha256"
        ],
    }
    result = {**body, "answer_plan_receipt_sha256": identity_sha256(body)}
    _validate_stored_plan(result)
    assert_gold_blind(result, path=f"locked_specialist_v2_ordinary_typed_{ordinal}")
    return result


def _repaired_source_plan(raw: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    unsigned = dict(raw)
    declared = unsigned.pop("question_receipt_sha256", None)
    _require(
        raw.get("ordinal") == ordinal and declared == identity_sha256(unsigned),
        f"repaired construction row seal changed at {ordinal}",
    )
    parent = base._verified_parent(raw, ordinal)  # noqa: SLF001
    normalized = {**dict(raw), "mode": SPECIALIST_MODE}
    common = base._common_plan(normalized, ordinal, parent=parent)  # noqa: SLF001
    try:
        generic = generic_answer._prompt_plan_row(raw, ordinal)  # noqa: SLF001
    except MatchedEvalContractError as exc:
        raise LockedSpecialistFinalAnswerV2Error(
            f"repaired generic prompt changed at {ordinal}: {exc}"
        ) from exc
    _require(
        generic["parent_prediction"] == common["parent_prediction"]
        and generic["parent_prediction_sha256"]
        == common["parent_prediction_sha256"]
        and generic["question_id"] == common["question_id"]
        and generic["question_sha256"] == common["question_sha256"]
        and generic["dated_question_sha256"] == common["dated_question_sha256"]
        and generic["construction_question_receipt_sha256"]
        == common["construction_question_receipt_sha256"],
        f"repaired prompt escaped its parent at {ordinal}",
    )
    provider_fields = {
        key: value
        for key, value in generic.items()
        if key
        not in {
            "construction_question_receipt_sha256",
            "dated_question_sha256",
            "ordinal",
            "parent_prediction",
            "parent_prediction_sha256",
            "prompt_row_receipt_sha256",
            "question_id",
            "question_sha256",
        }
    }
    draft = {**common, **provider_fields}
    scope, transform, messages, prompt_tokens = _repaired_scope_projection(
        plan=draft,
        raw=raw,
        ordinal=ordinal,
    )
    validation = dict(draft["validation_contract"])
    _require(
        repaired_v4.ADVISORY_SCOPE_KEY not in validation,
        "repaired advisory key collided with base validation",
    )
    validation[repaired_v4.ADVISORY_SCOPE_KEY] = scope
    body = {
        **draft,
        "adapter_prompt_transform": transform,
        "answer_parser_kind": REPAIRED_OPERATOR_PARSER,
        "construction_mode": REPAIRED_OPERATOR_MODE,
        "messages": messages,
        "messages_sha256": identity_sha256(messages),
        "prompt_token_proxy": prompt_tokens,
        "validation_contract": validation,
    }
    result = {**body, "answer_plan_receipt_sha256": identity_sha256(body)}
    _validate_stored_plan(result)
    assert_gold_blind(result, path=f"locked_specialist_v2_repaired_plan_{ordinal}")
    return result


def _source_plan(raw: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    mode = raw.get("mode")
    if mode == REPAIRED_OPERATOR_MODE:
        _require(
            ordinal in REPAIRED_OPERATOR_ORDINALS,
            f"unexpected repaired operator ordinal {ordinal}",
        )
        return _repaired_source_plan(raw, ordinal)
    if mode == SPECIALIST_MODE and ordinal in ORDINARY_TYPED_ORDINALS:
        return _ordinary_typed_source_plan(raw, ordinal)
    _require(
        mode in {SPECIALIST_MODE, PARENT_PASSTHROUGH_MODE},
        f"full-100 v2 answer mode changed at {ordinal}",
    )
    # The v1 source-plan builder calls its validator dynamically.  Use the
    # captured v1 validator for this inner call, then seal the v2 fields.
    previous = base._validate_stored_plan  # noqa: SLF001
    previous_scope = base._scoped_prompt_and_scope  # noqa: SLF001
    base._validate_stored_plan = _BASE_VALIDATE_STORED_PLAN  # noqa: SLF001
    base._scoped_prompt_and_scope = _BASE_SCOPED_PROMPT_AND_SCOPE  # noqa: SLF001
    try:
        plan = _BASE_SOURCE_PLAN(raw, ordinal)
    finally:
        base._validate_stored_plan = previous  # noqa: SLF001
        base._scoped_prompt_and_scope = previous_scope  # noqa: SLF001
    parser = SPECIALIST_PARSER if mode == SPECIALIST_MODE else PASSTHROUGH_PARSER
    return _augment_base_plan(plan, construction_mode=str(mode), parser_kind=parser)


def _validate_common_plan(raw: Mapping[str, Any]) -> None:
    body = dict(raw)
    declared = body.pop("answer_plan_receipt_sha256", None)
    ordinal = raw.get("ordinal")
    _require(
        type(ordinal) is int
        and 0 <= ordinal < EXPECTED_QUESTION_COUNT
        and declared == identity_sha256(body)
        and raw.get("parent_prediction_sha256")
        == repaired_v4.quote_sha256(
            require_text(raw.get("parent_prediction"), "parent prediction")
        ),
        f"v2 answer plan receipt changed at {ordinal}",
    )
    for key in (
        "construction_question_receipt_sha256",
        "dated_question_sha256",
        "parent_judge_row_sha256",
        "parent_prediction_sha256",
        "parent_replay_artifact_sha256",
        "parent_run_artifact_sha256",
        "parent_source_receipt_sha256",
        "parent_source_row_sha256",
        "question_sha256",
    ):
        require_sha256(raw.get(key), f"v2 answer plan {key}")


def _validate_repaired_plan(raw: Mapping[str, Any]) -> None:
    ordinal = int(raw["ordinal"])
    allowed = raw.get("allowed_handle_ids")
    groups = raw.get("handle_group_by_id")
    messages = raw.get("messages")
    validation = raw.get("validation_contract")
    _require(
        ordinal in REPAIRED_OPERATOR_ORDINALS
        and raw.get("mode") == SPECIALIST_MODE
        and raw.get("construction_mode") == REPAIRED_OPERATOR_MODE
        and raw.get("answer_parser_kind") == REPAIRED_OPERATOR_PARSER
        and type(allowed) is list
        and bool(allowed)
        and type(groups) is dict
        and set(groups) == set(allowed)
        and type(messages) is list
        and raw.get("messages_sha256") == identity_sha256(messages)
        and raw.get("prompt_token_proxy")
        == repaired_v4.count_chat_prompt_token_proxy(messages)
        and raw["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE
        <= HARD_COMPLETE_CHAT_TOKEN_CAP
        and type(raw.get("story_coherence")) is dict
        and type(raw.get("preservation_requirements")) is dict
        and type(validation) is dict,
        f"repaired stored plan changed at {ordinal}",
    )
    assert type(validation) is dict
    scope, _advisory, _base_contract = repaired_v4._advisory_scope(  # noqa: SLF001
        validation,
        allowed,
    )
    _require(
        scope.get("ordinal") == ordinal,
        f"repaired scope ordinal changed at {ordinal}",
    )
    transform = raw.get("adapter_prompt_transform")
    if ordinal == 42:
        _require(
            type(transform) is dict
            and transform == scope.get("adapter_prompt_transform")
            and transform.get("target_messages_sha256")
            == raw.get("messages_sha256")
            and transform.get("target_prompt_token_proxy")
            == raw.get("prompt_token_proxy"),
            "q42 full-100 prompt transform changed",
        )
    else:
        _require(
            transform is None and scope.get("adapter_prompt_transform") is None,
            f"unexpected repaired prompt transform at {ordinal}",
        )


def _validate_ordinary_typed_plan(raw: Mapping[str, Any]) -> None:
    ordinal = int(raw["ordinal"])
    allowed = raw.get("allowed_handle_ids")
    groups = raw.get("handle_group_by_id")
    messages = raw.get("messages")
    transform = raw.get("adapter_prompt_transform")
    _require(
        ordinal in ORDINARY_TYPED_ORDINALS
        and raw.get("mode") == SPECIALIST_MODE
        and raw.get("construction_mode") == SPECIALIST_MODE
        and raw.get("answer_parser_kind") == ORDINARY_TYPED_PARSER
        and type(allowed) is list
        and bool(allowed)
        and len(allowed) == len(set(allowed))
        and type(groups) is dict
        and set(groups) == set(allowed)
        and type(messages) is list
        and raw.get("messages_sha256") == identity_sha256(messages)
        and raw.get("prompt_token_proxy")
        == repaired_v4.count_chat_prompt_token_proxy(messages)
        and raw["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE
        <= HARD_COMPLETE_CHAT_TOKEN_CAP
        and type(raw.get("story_coherence")) is dict
        and type(raw.get("preservation_requirements")) is dict
        and type(raw.get("validation_contract")) is dict
        and type(transform) is dict,
        f"ordinary typed stored plan changed at {ordinal}",
    )
    assert type(transform) is dict
    transform_body = dict(transform)
    declared = transform_body.pop("receipt_sha256", None)
    _require(
        set(transform)
        == {
            "fitted_prompt_receipt_sha256",
            "format",
            "ordinal",
            "provider_input_sha256",
            "receipt_sha256",
            "source_complete_envelope_tokens",
            "source_message_renderer_format",
            "source_messages_sha256",
            "source_prompt_token_proxy",
            "source_specialist_prompt_envelope_receipt_sha256",
            "source_terminal_prompt_receipt_sha256",
            "specialist_advisories_sha256",
            "target_complete_envelope_tokens",
            "target_message_renderer_format",
            "target_messages_sha256",
            "target_prompt_token_proxy",
            "target_terminal_prompt_receipt_sha256",
            "transform",
        }
        and declared == identity_sha256(transform_body)
        and transform.get("format") == ORDINARY_TYPED_PROMPT_TRANSFORM_FORMAT
        and transform.get("ordinal") == ordinal
        and transform.get("source_message_renderer_format")
        == SPECIALIST_PROMPT_FORMAT
        and transform.get("source_terminal_prompt_receipt_sha256")
        == raw.get("terminal_prompt_receipt_sha256")
        and transform.get("target_message_renderer_format") == generic_answer.FORMAT
        and transform.get("target_messages_sha256") == raw.get("messages_sha256")
        and transform.get("target_prompt_token_proxy")
        == raw.get("prompt_token_proxy")
        and transform.get("target_complete_envelope_tokens")
        == raw["prompt_token_proxy"] + OUTPUT_TOKEN_RESERVE
        and transform.get("target_complete_envelope_tokens")
        <= HARD_COMPLETE_CHAT_TOKEN_CAP
        and transform.get("transform")
        == "rerender_identical_provider_input_as_ordinary_typed_final",
        f"ordinary typed prompt transform changed at {ordinal}",
    )
    for key in (
        "fitted_prompt_receipt_sha256",
        "provider_input_sha256",
        "source_messages_sha256",
        "source_specialist_prompt_envelope_receipt_sha256",
        "source_terminal_prompt_receipt_sha256",
        "specialist_advisories_sha256",
        "target_messages_sha256",
        "target_terminal_prompt_receipt_sha256",
    ):
        require_sha256(transform.get(key), f"ordinary typed transform {key}")


def _validate_stored_plan(raw: Mapping[str, Any]) -> dict[str, Any]:
    parser = raw.get("answer_parser_kind")
    fields = (
        _PASSTHROUGH_PLAN_FIELDS
        if parser == PASSTHROUGH_PARSER
        else _SCOPED_PLAN_FIELDS
        if parser == SPECIALIST_PARSER
        else _REPAIRED_PLAN_FIELDS
        if parser == REPAIRED_OPERATOR_PARSER
        else _REPAIRED_PLAN_FIELDS
        if parser == ORDINARY_TYPED_PARSER
        else frozenset()
    )
    _require(
        bool(fields) and set(raw) == fields,
        f"full-100 v2 plan schema changed at {raw.get('ordinal')}",
    )
    _validate_common_plan(raw)
    if parser in {SPECIALIST_PARSER, PASSTHROUGH_PARSER}:
        _BASE_VALIDATE_STORED_PLAN(_base_plan_without_v2_fields(raw))
        _require(
            raw.get("adapter_prompt_transform") is None
            and raw.get("construction_mode") == raw.get("mode"),
            f"preserved v1 plan metadata changed at {raw.get('ordinal')}",
        )
    elif parser == REPAIRED_OPERATOR_PARSER:
        _validate_repaired_plan(raw)
    else:
        _validate_ordinary_typed_plan(raw)
    assert_gold_blind(raw, path=f"loaded_locked_specialist_answer_v2_{raw.get('ordinal')}")
    return dict(raw)


def _scoped_prompt_and_scope(raw: Mapping[str, Any]):
    parser = raw.get("answer_parser_kind")
    if parser is None:
        return _BASE_SCOPED_PROMPT_AND_SCOPE(raw)
    if parser == SPECIALIST_PARSER:
        prompt, base_scope = _BASE_SCOPED_PROMPT_AND_SCOPE(raw)
        provider_input = raw.get("provider_input")
        validation_contract = raw.get("validation_contract")
        allowed = raw.get("allowed_handle_ids")
        _require(
            type(provider_input) is dict
            and type(validation_contract) is dict
            and type(allowed) is list
            and bool(allowed)
            and len(allowed) == len(set(allowed)),
            f"scoped v2 recovery input changed at {raw.get('ordinal')}",
        )
        assert type(provider_input) is dict
        assert type(validation_contract) is dict
        assert type(allowed) is list
        scope_body = {
            "allowed_handle_ids": list(allowed),
            "answer_plan_receipt_sha256": raw["answer_plan_receipt_sha256"],
            "base_scope_receipt_sha256": base_scope.receipt_sha256,
            "format": f"{MIXED_PARSER_FORMAT}-scoped-v2-recovery-scope-v1",
            "ordinal": raw["ordinal"],
            "provider_input_sha256": identity_sha256(provider_input),
            "validation_contract_sha256": identity_sha256(validation_contract),
        }
        return prompt, _ScopedSpecialistV2Scope(
            ordinal=int(raw["ordinal"]),
            base_scope=base_scope,
            allowed_handle_ids=tuple(allowed),
            provider_input=dict(provider_input),
            validation_contract=dict(validation_contract),
            answer_plan_receipt_sha256=raw["answer_plan_receipt_sha256"],
            receipt_sha256=identity_sha256(scope_body),
        )
    if parser == ORDINARY_TYPED_PARSER:
        transform = raw["adapter_prompt_transform"]
        scope_body = {
            "allowed_handle_ids": list(raw["allowed_handle_ids"]),
            "answer_plan_receipt_sha256": raw["answer_plan_receipt_sha256"],
            "format": f"{MIXED_PARSER_FORMAT}-ordinary-typed-scope-v1",
            "ordinal": raw["ordinal"],
            "prompt_transform_receipt_sha256": transform["receipt_sha256"],
            "validation_contract_sha256": identity_sha256(
                raw["validation_contract"]
            ),
        }
        scope = _OrdinaryTypedScope(
            ordinal=int(raw["ordinal"]),
            allowed_handle_ids=tuple(raw["allowed_handle_ids"]),
            handle_group_by_id=dict(raw["handle_group_by_id"]),
            story_coherence=dict(raw["story_coherence"]),
            preservation_requirements=dict(raw["preservation_requirements"]),
            validation_contract=dict(raw["validation_contract"]),
            prompt_transform_receipt_sha256=transform["receipt_sha256"],
            receipt_sha256=identity_sha256(scope_body),
        )
        prompt = SimpleNamespace(
            messages=tuple(dict(row) for row in raw["messages"]),
            prompt_token_proxy=int(raw["prompt_token_proxy"]),
        )
        return prompt, scope
    _require(
        parser == REPAIRED_OPERATOR_PARSER,
        f"unexpected provider parser at ordinal {raw.get('ordinal')}",
    )
    allowed = tuple(raw["allowed_handle_ids"])
    scope_body = {
        "allowed_handle_ids": list(allowed),
        "answer_plan_receipt_sha256": raw["answer_plan_receipt_sha256"],
        "format": f"{MIXED_PARSER_FORMAT}-repaired-scope-v1",
        "ordinal": raw["ordinal"],
        "validation_contract_sha256": identity_sha256(raw["validation_contract"]),
    }
    scope = _RepairedOperatorScope(
        ordinal=int(raw["ordinal"]),
        allowed_handle_ids=allowed,
        handle_group_by_id=dict(raw["handle_group_by_id"]),
        story_coherence=dict(raw["story_coherence"]),
        preservation_requirements=dict(raw["preservation_requirements"]),
        validation_contract=dict(raw["validation_contract"]),
        receipt_sha256=identity_sha256(scope_body),
    )
    prompt = SimpleNamespace(
        messages=tuple(dict(row) for row in raw["messages"]),
        prompt_token_proxy=int(raw["prompt_token_proxy"]),
    )
    return prompt, scope


_LOCAL_PAIR_RECOVERABLE_ERRORS = frozenset(
    {
        "specialist_temporal_order_entailment",
        "specialist_temporal_order_scope",
    }
)
_LOCAL_PAIR_VALIDATION_BASIS = "receipt_bound_local_temporal_pair_parent_agreement"


def _local_temporal_pair_resolution(
    completion: str,
    *,
    parent_prediction: str,
    scope: _ScopedSpecialistV2Scope,
    base_parsed: Any,
) -> local_pair.LocalTemporalPairResolution | None:
    if (
        base_parsed.valid
        or base_parsed.error_code not in _LOCAL_PAIR_RECOVERABLE_ERRORS
    ):
        return None
    dated_question = scope.provider_input.get("dated_question")
    if type(dated_question) is not str:
        return None
    return local_pair.resolve_parent_from_local_temporal_pair(
        dated_question=dated_question,
        parent_prediction=parent_prediction,
        provider_input=scope.provider_input,
        validation_contract=scope.validation_contract,
        allowed_handle_ids=scope.allowed_handle_ids,
        answer_plan_receipt_sha256=scope.answer_plan_receipt_sha256,
        base_scope_receipt_sha256=scope.base_scope.receipt_sha256,
        source_completion_sha256=repaired_v4.quote_sha256(completion),
    )


def _local_temporal_pair_decision(
    completion: str,
    *,
    parent_prediction: str,
    scope: _ScopedSpecialistV2Scope,
    base_parsed: Any,
    resolution: local_pair.LocalTemporalPairResolution,
):
    body = {
        "base_error_code": base_parsed.error_code,
        "base_parse_receipt_sha256": base_parsed.receipt_sha256,
        "decision": "keep_parent",
        "format": f"{MIXED_PARSER_FORMAT}-local-temporal-pair-decision-v1",
        "parent_prediction_sha256": repaired_v4.quote_sha256(parent_prediction),
        "proof_receipt_sha256": resolution.proof_receipt_sha256,
        "resolution_receipt_sha256": resolution.receipt_sha256,
        "scope_receipt_sha256": scope.receipt_sha256,
        "source_completion_sha256": repaired_v4.quote_sha256(completion),
        "validation_basis": _LOCAL_PAIR_VALIDATION_BASIS,
    }
    assert_gold_blind(body, path="locked_specialist_local_temporal_pair_decision")
    return SimpleNamespace(
        decision="keep_parent",
        error_code="none",
        prediction=parent_prediction,
        proof_kind="local_temporal_pair",
        proof_receipt_sha256=resolution.proof_receipt_sha256,
        receipt_sha256=identity_sha256(body),
        scope_receipt_sha256=scope.receipt_sha256,
        used_handle_ids=(),
        valid=True,
        validation_basis=_LOCAL_PAIR_VALIDATION_BASIS,
    )


def _parse_completion(
    completion: str,
    *,
    parent_prediction: str,
    scope: Any,
):
    if isinstance(scope, _ScopedSpecialistV2Scope):
        parsed = _BASE_PARSE_SPECIALIST(
            completion,
            parent_prediction=parent_prediction,
            scope=scope.base_scope,
        )
        resolution = _local_temporal_pair_resolution(
            completion,
            parent_prediction=parent_prediction,
            scope=scope,
            base_parsed=parsed,
        )
        if resolution is None:
            return parsed
        return _local_temporal_pair_decision(
            completion,
            parent_prediction=parent_prediction,
            scope=scope,
            base_parsed=parsed,
            resolution=resolution,
        )
    if isinstance(scope, _OrdinaryTypedScope):
        parsed = generic_answer.parse_typed_final_completion(
            completion,
            parent_prediction=parent_prediction,
            allowed_handle_ids=scope.allowed_handle_ids,
            handle_group_by_id=scope.handle_group_by_id,
            story_coherence=scope.story_coherence,
            preservation_requirements=scope.preservation_requirements,
            validation_contract=scope.validation_contract,
        )
        return SimpleNamespace(
            decision=parsed.decision,
            error_code=parsed.error_code,
            prediction=parsed.prediction,
            proof_kind=f"ordinary_typed_final_{scope.ordinal}",
            proof_receipt_sha256=scope.prompt_transform_receipt_sha256,
            receipt_sha256=parsed.receipt_sha256,
            scope_receipt_sha256=scope.receipt_sha256,
            used_handle_ids=parsed.used_handle_ids,
            valid=parsed.valid,
            validation_basis=parsed.validation_basis,
        )
    if not isinstance(scope, _RepairedOperatorScope):
        return _BASE_PARSE_SPECIALIST(
            completion,
            parent_prediction=parent_prediction,
            scope=scope,
        )
    parsed = repaired_v4.parse_v4_completion(
        completion,
        parent_prediction=parent_prediction,
        allowed_handle_ids=scope.allowed_handle_ids,
        handle_group_by_id=scope.handle_group_by_id,
        story_coherence=scope.story_coherence,
        preservation_requirements=scope.preservation_requirements,
        validation_contract=scope.validation_contract,
    )
    advisory_scope = scope.validation_contract[repaired_v4.ADVISORY_SCOPE_KEY]
    return SimpleNamespace(
        decision=parsed.decision,
        error_code=parsed.error_code,
        prediction=parsed.prediction,
        proof_kind=f"repaired_operator_{scope.ordinal}",
        proof_receipt_sha256=advisory_scope["receipt_sha256"],
        receipt_sha256=parsed.receipt_sha256,
        scope_receipt_sha256=scope.receipt_sha256,
        used_handle_ids=parsed.used_handle_ids,
        valid=parsed.valid,
        validation_basis=parsed.validation_basis,
    )


def _preflight_projection(
    construction: SealedArtifact,
    plans: tuple[dict[str, Any], ...],
    *,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> dict[str, Any]:
    payload = _BASE_PREFLIGHT_PROJECTION(
        construction,
        plans,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
    )
    provider_count = int(payload["specialist_question_count"])
    specialist_count = sum(
        row["answer_parser_kind"] == SPECIALIST_PARSER for row in plans
    )
    ordinary_typed_count = sum(
        row["answer_parser_kind"] == ORDINARY_TYPED_PARSER for row in plans
    )
    repaired_count = sum(
        row["answer_parser_kind"] == REPAIRED_OPERATOR_PARSER for row in plans
    )
    _require(
        provider_count == EXPECTED_PROVIDER_PROMPT_COUNT
        and specialist_count == EXPECTED_SCOPED_SPECIALIST_COUNT
        and ordinary_typed_count == len(ORDINARY_TYPED_ORDINALS)
        and specialist_count + ordinary_typed_count == EXPECTED_SPECIALIST_COUNT
        and repaired_count == EXPECTED_REPAIRED_OPERATOR_COUNT,
        "full-100 v2 preflight parser population changed",
    )
    payload["provider_question_count"] = provider_count
    payload["specialist_question_count"] = (
        specialist_count + ordinary_typed_count
    )
    payload["scoped_specialist_question_count"] = specialist_count
    payload["ordinary_typed_question_count"] = ordinary_typed_count
    payload["repaired_operator_question_count"] = repaired_count
    payload["answer_parser_population_sha256"] = identity_sha256(
        [
            {
                "answer_parser_kind": row["answer_parser_kind"],
                "answer_plan_receipt_sha256": row["answer_plan_receipt_sha256"],
                "ordinal": row["ordinal"],
            }
            for row in plans
        ]
    )
    assert_gold_blind(payload, path="locked_specialist_final_answer_v2_preflight")
    return payload


def _validate_preflight(artifact: SealedArtifact):
    payload = dict(artifact.payload)
    provider_count = payload.pop("provider_question_count", None)
    specialist_count = payload.get("specialist_question_count")
    scoped_count = payload.pop("scoped_specialist_question_count", None)
    ordinary_typed_count = payload.pop("ordinary_typed_question_count", None)
    repaired_count = payload.pop("repaired_operator_question_count", None)
    parser_population = payload.pop("answer_parser_population_sha256", None)
    payload["specialist_question_count"] = provider_count
    synthetic = SealedArtifact(artifact.path, artifact.sha256, payload)
    prompts, plans = _BASE_VALIDATE_PREFLIGHT(synthetic)
    observed_parser_population = identity_sha256(
        [
            {
                "answer_parser_kind": row["answer_parser_kind"],
                "answer_plan_receipt_sha256": row["answer_plan_receipt_sha256"],
                "ordinal": row["ordinal"],
            }
            for row in plans
        ]
    )
    _require(
        provider_count == EXPECTED_PROVIDER_PROMPT_COUNT
        and specialist_count == EXPECTED_SPECIALIST_COUNT
        and scoped_count == EXPECTED_SCOPED_SPECIALIST_COUNT
        and ordinary_typed_count == len(ORDINARY_TYPED_ORDINALS)
        and repaired_count == EXPECTED_REPAIRED_OPERATOR_COUNT
        and parser_population == observed_parser_population
        and sum(row["answer_parser_kind"] == SPECIALIST_PARSER for row in plans)
        == EXPECTED_SCOPED_SPECIALIST_COUNT
        and tuple(
            row["ordinal"]
            for row in plans
            if row["answer_parser_kind"] == ORDINARY_TYPED_PARSER
        )
        == ORDINARY_TYPED_ORDINALS
        and sum(
            row["answer_parser_kind"] == REPAIRED_OPERATOR_PARSER for row in plans
        )
        == EXPECTED_REPAIRED_OPERATOR_COUNT,
        "sealed full-100 v2 parser population changed",
    )
    return prompts, plans


def build_runtime(
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
        required == EXPECTED_PROVIDER_PROMPT_COUNT
        and model == DEFAULT_MODEL == artifact.payload.get("model")
        and gateway_url == artifact.payload.get("gateway_url")
        and max_concurrency == artifact.payload.get("max_concurrency")
        and len(prompts) == required,
        "full-100 v2 runtime differs from sealed preflight",
    )
    return FastCompletionRuntime(
        checkpoint_dir=Path(output_root) / CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=model,
        client=client,
        max_prompt_tokens=MAX_CHAT_PROMPT_TOKENS,
        max_new_tokens=OUTPUT_TOKEN_RESERVE,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "arm": "locked_specialist_final_terra_answer_v2",
            "authorized_unique_calls": required,
            "construction_artifact_sha256": artifact.payload[
                "construction_artifact_sha256"
            ],
            "experiment_format": FORMAT,
            "gateway_url": gateway_url,
            "gold_loaded": False,
            "preflight_artifact_sha256": artifact.sha256,
            "scoped_completion_format": MIXED_PARSER_FORMAT,
        },
    )


def _materialization_projection(
    preflight: SealedArtifact,
    plans: tuple[dict[str, Any], ...],
    batch: Any,
) -> dict[str, Any]:
    payload = _BASE_MATERIALIZATION_PROJECTION(preflight, plans, batch)
    questions = payload.get("questions")
    _require(
        type(questions) is list and len(questions) == EXPECTED_QUESTION_COUNT,
        "full-100 v2 materialized population changed",
    )
    by_ordinal = {int(row["ordinal"]): row for row in plans}
    rewritten: list[dict[str, Any]] = []
    for raw in questions:
        _require(type(raw) is dict, "full-100 v2 result row changed type")
        row = dict(raw)
        ordinal = int(row["ordinal"])
        plan = by_ordinal[ordinal]
        parser = plan["answer_parser_kind"]
        _require(
            row.get("decision") != "invalid_keep_parent"
            and row.get("solver_valid") is not False,
            f"invalid specialist completion at ordinal {ordinal}; refusing parent fallback",
        )
        if parser == REPAIRED_OPERATOR_PARSER:
            _require(
                row.get("decision") == "replace",
                f"repaired operator did not replace its parent at ordinal {ordinal}",
            )
            row["answer_mode"] = REPAIRED_OPERATOR_MODE
            row["prediction_source"] = (
                "locked_specialist_repaired_operator_validated_replacement_v2"
            )
        elif parser == SPECIALIST_PARSER:
            row["prediction_source"] = (
                "locked_specialist_scoped_validated_replacement_v2"
                if row.get("decision") == "replace"
                else "locked_specialist_scoped_validated_keep_parent_v2"
            )
        elif parser == ORDINARY_TYPED_PARSER:
            row["prediction_source"] = (
                "locked_specialist_ordinary_typed_validated_replacement_v2"
                if row.get("decision") == "replace"
                else "locked_specialist_ordinary_typed_validated_keep_parent_v2"
            )
        else:
            _require(
                parser == PASSTHROUGH_PARSER,
                f"unexpected materialized parser at ordinal {ordinal}",
            )
            row["prediction_source"] = "locked_specialist_parent_passthrough_v2"
        unsigned = dict(row)
        unsigned.pop("source_row_sha256", None)
        rewritten.append({**unsigned, "source_row_sha256": identity_sha256(unsigned)})
    payload["questions"] = rewritten
    payload["judge_rows"] = [judge_row_projection(row) for row in rewritten]
    payload["invalid_completion_parent_fallback_count"] = 0
    payload["provider_question_count"] = EXPECTED_PROVIDER_PROMPT_COUNT
    payload["specialist_question_count"] = EXPECTED_SPECIALIST_COUNT
    payload["scoped_specialist_question_count"] = EXPECTED_SCOPED_SPECIALIST_COUNT
    payload["ordinary_typed_question_count"] = len(ORDINARY_TYPED_ORDINALS)
    payload["repaired_operator_question_count"] = EXPECTED_REPAIRED_OPERATOR_COUNT
    payload["validated_keep_parent_count"] = sum(
        row["decision"] == "keep_parent" for row in rewritten
    )
    payload["validated_replacement_count"] = sum(
        row["decision"] == "replace" for row in rewritten
    )
    provider_plans = tuple(row for row in plans if row["mode"] == SPECIALIST_MODE)
    _require(
        len(provider_plans) == len(batch.logical_completions),
        "local temporal proof population changed provider alignment",
    )
    result_by_ordinal = {int(row["ordinal"]): row for row in rewritten}
    local_proofs: list[dict[str, Any]] = []
    for plan, completion in zip(
        provider_plans,
        batch.logical_completions,
        strict=True,
    ):
        if plan["answer_parser_kind"] != SPECIALIST_PARSER:
            continue
        _prompt, scope = _scoped_prompt_and_scope(plan)
        _require(
            isinstance(scope, _ScopedSpecialistV2Scope),
            f"scoped v2 proof wrapper changed at {plan['ordinal']}",
        )
        base_parsed = _BASE_PARSE_SPECIALIST(
            completion,
            parent_prediction=plan["parent_prediction"],
            scope=scope.base_scope,
        )
        resolution = _local_temporal_pair_resolution(
            completion,
            parent_prediction=plan["parent_prediction"],
            scope=scope,
            base_parsed=base_parsed,
        )
        if resolution is None:
            continue
        result = result_by_ordinal[int(plan["ordinal"])]
        _require(
            result.get("decision") == "keep_parent"
            and result.get("prediction") == plan["parent_prediction"]
            and result.get("used_handle_ids") == []
            and result.get("proof_kind") == "local_temporal_pair"
            and result.get("proof_receipt_sha256")
            == resolution.proof_receipt_sha256
            and result.get("validation_basis") == _LOCAL_PAIR_VALIDATION_BASIS
            and result.get("specialist_scope_receipt_sha256")
            == scope.receipt_sha256,
            f"local temporal pair result changed at {plan['ordinal']}",
        )
        proof_body = {
            "answer_result_source_row_sha256": result["source_row_sha256"],
            "base_error_code": base_parsed.error_code,
            "base_parse_receipt_sha256": base_parsed.receipt_sha256,
            "format": f"{MIXED_PARSER_FORMAT}-local-temporal-pair-ledger-row-v1",
            "ordinal": plan["ordinal"],
            "resolution": resolution.projection(),
            "scope_receipt_sha256": scope.receipt_sha256,
        }
        assert_gold_blind(
            proof_body,
            path=f"locked_specialist_local_temporal_pair_ledger_{plan['ordinal']}",
        )
        local_proofs.append(
            {**proof_body, "receipt_sha256": identity_sha256(proof_body)}
        )
    payload["local_temporal_pair_proof_count"] = len(local_proofs)
    payload["local_temporal_pair_proofs"] = local_proofs
    payload["local_temporal_pair_proof_population_sha256"] = identity_sha256(
        [row["receipt_sha256"] for row in local_proofs]
    )
    payload["scoped_completion_format"] = MIXED_PARSER_FORMAT
    assert_gold_blind(payload, path="locked_specialist_final_terra_answer_v2")
    return payload


_BASE_LOCK = RLock()


def _base_globals() -> dict[str, Any]:
    return {
        "CHECKPOINT_DIR_NAME": CHECKPOINT_DIR_NAME,
        "DEFAULT_CONSTRUCTION": DEFAULT_CONSTRUCTION,
        "DEFAULT_MODEL": DEFAULT_MODEL,
        "DEFAULT_OUTPUT": DEFAULT_OUTPUT,
        "FORMAT": FORMAT,
        "PREFLIGHT_FORMAT": PREFLIGHT_FORMAT,
        "PREFLIGHT_NAME": PREFLIGHT_NAME,
        "REPLAY_NAME": REPLAY_NAME,
        "RESULT_ROW_FORMAT": RESULT_ROW_FORMAT,
        "RUN_NAME": RUN_NAME,
        "SCOPED_COMPLETION_FORMAT": MIXED_PARSER_FORMAT,
        "_default_construction_loader": _default_construction_loader,
        "_materialization_projection": _materialization_projection,
        "_preflight_projection": _preflight_projection,
        "_runtime": build_runtime,
        "_scoped_prompt_and_scope": _scoped_prompt_and_scope,
        "_source_plan": _source_plan,
        "_validate_preflight": _validate_preflight,
        "_validate_stored_plan": _validate_stored_plan,
        "parse_specialist_scoped_completion": _parse_completion,
    }


@contextmanager
def _v2_base_contract() -> Iterator[None]:
    with _BASE_LOCK:
        values = _base_globals()
        previous = {name: getattr(base, name) for name in values}
        for name, value in values.items():
            setattr(base, name, value)
        try:
            yield
        finally:
            for name, value in previous.items():
                setattr(base, name, value)


def load_answer_plans(
    path: str | Path,
    expected_sha256: str = EXPECTED_CONSTRUCTION_SHA256,
):
    with _v2_base_contract():
        return base.load_answer_plans(path, expected_sha256)


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    with _v2_base_contract():
        result = base.run_preflight(args)
    return {
        **result,
        "ordinary_typed_question_count": len(ORDINARY_TYPED_ORDINALS),
        "provider_question_count": EXPECTED_PROVIDER_PROMPT_COUNT,
        "repaired_operator_question_count": EXPECTED_REPAIRED_OPERATOR_COUNT,
        "scoped_specialist_question_count": EXPECTED_SCOPED_SPECIALIST_COUNT,
    }


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    with _v2_base_contract():
        return base.run_provider(args)


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    with _v2_base_contract():
        return base.run_materialize(args)


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    with _v2_base_contract():
        return base.run_replay(args)


def validate_preflight_artifact(artifact: SealedArtifact):
    with _v2_base_contract():
        return _validate_preflight(artifact)


def build_parser() -> argparse.ArgumentParser:
    with _v2_base_contract():
        parser = base.build_parser()
    parser.description = __doc__
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = run_preflight(args)
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
    "DEFAULT_CONSTRUCTION",
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT",
    "EXPECTED_CONSTRUCTION_SHA256",
    "EXPECTED_PASSTHROUGH_COUNT",
    "EXPECTED_PROVIDER_PROMPT_COUNT",
    "EXPECTED_REPAIRED_OPERATOR_COUNT",
    "EXPECTED_SPECIALIST_COUNT",
    "FORMAT",
    "MIXED_PARSER_FORMAT",
    "PREFLIGHT_NAME",
    "REPLAY_NAME",
    "RUN_NAME",
    "LockedSpecialistFinalAnswerV2Error",
    "build_parser",
    "build_runtime",
    "load_answer_plans",
    "main",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
    "validate_preflight_artifact",
]
