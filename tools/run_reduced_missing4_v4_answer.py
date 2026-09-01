#!/usr/bin/env python3
"""Checkpoint four Terra answers from a sealed missing-four v4 construction.

This module is intentionally a version adapter.  The audited specialist-v2
answer runner remains the sole implementation of prompt sealing, checkpoint
journals, materialization, and byte-identical replay.  V4 adds only an
authenticated construction loader, a receipt-bound four-shape completion
arbiter, the four-question population contract, versioned artifact names, and
v4 runtime provenance.

Gold is never opened here.  Provider execution remains a separately authorized
phase and the complete prompt plus output reserve may not exceed 8,000 tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionRuntime,
)
from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from memory_condense.domain._tokenizer import (  # noqa: E402
    count_chat_prompt_token_proxy,
)
from tools import run_reduced_missing4_v4_construction as construction_v4  # noqa: E402
from tools import run_reduced_specialist_answer_v2 as base  # noqa: E402
from tools.matched_eval import conjunctive_event_sufficiency as event_op  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    ParsedTypedFinalDecision,
)
from tools.matched_eval import typed_memory_final_arm as typed_final  # noqa: E402
from tools.matched_eval.typed_numeric_semantics import (  # noqa: E402
    NumericDimension,
    NumericQualifier,
    numeric_mentions,
)


FORMAT = "memory-condense-reduced-missing4-terra-answer-v4"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight"

PREFLIGHT_NAME = "reduced-missing4-answer-preflight-v4.json"
RUN_NAME = "reduced-missing4-answer-v4.json"
REPLAY_NAME = "reduced-missing4-answer-replay-v4.json"
CHECKPOINT_DIR_NAME = "reduced-missing4-answer-checkpoints-v4"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONSTRUCTION = construction_v4.DEFAULT_OUTPUT_ROOT / construction_v4.CONSTRUCTION_NAME
DEFAULT_OUTPUT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/reduced-missing4-answer-v4"
)
EXPECTED_ORDINALS = tuple(construction_v4.TARGET_ORDINALS)
EXPECTED_PROVIDER_CALLS = len(EXPECTED_ORDINALS)
DEFAULT_MODEL = base.DEFAULT_MODEL
HARD_COMPLETE_CHAT_TOKEN_CAP = base.HARD_COMPLETE_CHAT_TOKEN_CAP
OUTPUT_TOKEN_RESERVE = base.OUTPUT_TOKEN_RESERVE
MAX_CHAT_PROMPT_TOKENS = base.MAX_CHAT_PROMPT_TOKENS

ADVISORY_SCOPE_FORMAT = "memory-condense-reduced-missing4-advisory-scope-v4"
ADVISORY_SCOPE_KEY = "_missing4_v4_advisory_scope"
VALIDATOR_POLICY_FORMAT = "memory-condense-reduced-missing4-validator-policy-v4"
DECISION_FORMAT = "memory-condense-reduced-missing4-validated-decision-v4"
Q42_PROMPT_TRANSFORM_FORMAT = (
    "memory-condense-reduced-missing4-q42-scoped-insufficiency-prompt-v1"
)

_BASE_VALIDATE_PREFLIGHT = base._validate_preflight  # noqa: SLF001
_BASE_MATERIALIZATION_PROJECTION = base._materialization_projection  # noqa: SLF001
_BASE_PARSE_COMPLETION = base.parse_typed_final_completion


class ReducedMissing4V4AnswerError(MatchedEvalContractError):
    """Raised when the v4 construction or answer protocol changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ReducedMissing4V4AnswerError(message)


def _q42_expected_text(advisory: Mapping[str, Any]) -> str:
    program = advisory.get("conjunctive_event_program")
    _require(
        type(program) is dict and type(program.get("obligations")) is list,
        "q42 conjunctive-event program changed",
    )
    answer_rows = [
        row
        for row in program["obligations"]
        if type(row) is dict and row.get("answer_variable") is True
    ]
    _require(len(answer_rows) == 1, "q42 answer obligation changed")
    return event_op.canonical_scoped_insufficiency_text(
        require_text(answer_rows[0].get("answer_value_type"), "q42 answer value type")
    )


def _q42_prompt_transform(
    plan: Mapping[str, Any],
    advisory: Mapping[str, Any],
    source_terminal_receipt_sha256: str,
) -> tuple[dict[str, Any], tuple[dict[str, str], ...], int]:
    allowed = tuple(plan["allowed_handle_ids"])
    expected = _q42_expected_text(advisory)
    instruction = (
        "V4 conjunctive-event scope rule: the sealed advisory establishes that "
        "the supplied evidence does not prove every requested edge on one event "
        "identity. For this supplied-evidence scope, replace the protected parent "
        f"with exactly {json.dumps(expected, ensure_ascii=False)} and cite exactly "
        f"these supplied handles: {json.dumps(list(allowed))}. This conclusion is "
        "only that the supplied evidence cannot establish the requested join; it "
        "must not assert that the fact is absent from global memory."
    )
    source_messages = tuple(dict(row) for row in plan["messages"])
    _require(
        bool(source_messages) and source_messages[0].get("role") == "system",
        "q42 base prompt lost its primary system message",
    )
    messages = (
        {
            **source_messages[0],
            "content": source_messages[0]["content"] + "\n\n" + instruction,
        },
        *source_messages[1:],
    )
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    _require(
        prompt_tokens + OUTPUT_TOKEN_RESERVE <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "q42 adapter transform escaped the hard 8k envelope",
    )
    body = {
        "format": Q42_PROMPT_TRANSFORM_FORMAT,
        "instruction": instruction,
        "instruction_sha256": quote_sha256(instruction),
        "source_messages_sha256": require_sha256(
            plan.get("messages_sha256"), "q42 source messages"
        ),
        "source_terminal_prompt_receipt_sha256": require_sha256(
            source_terminal_receipt_sha256, "q42 source terminal"
        ),
        "target_complete_envelope_tokens": prompt_tokens + OUTPUT_TOKEN_RESERVE,
        "target_messages_sha256": identity_sha256(list(messages)),
        "target_prompt_token_proxy": prompt_tokens,
        "transform": "append_to_primary_system_message",
    }
    return {**body, "receipt_sha256": identity_sha256(body)}, messages, prompt_tokens


def load_verified_construction(
    path: str | Path,
    expected_sha256: str,
) -> tuple[SealedArtifact, tuple[dict[str, Any], ...]]:
    """Authenticate v4 and normalize its rows through the audited v2 seam.

    The construction owns all retrieval-specific structure.  Every finalized
    v4 question must expose the common ``fitted_typed_prompt`` plus
    ``terminal_prompt`` source schema consumed by ``base._prompt_plan_row``.
    Retrieval-specific advisories are then copied byte-for-byte into a sealed
    validation-contract extension.  There is no deterministic-answer branch.
    """

    artifact = read_sealed_json(Path(path))
    _require(
        artifact.sha256
        == require_sha256(expected_sha256, "expected missing-four v4 construction"),
        "missing-four v4 construction digest changed",
    )
    rows = tuple(construction_v4.validate_construction(artifact))
    payload = artifact.payload
    _require(
        payload.get("format") == construction_v4.CONSTRUCTION_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("target_labels_loaded") is False
        and payload.get("target_plan_loaded") is False
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and tuple(payload.get("ordinals", ())) == EXPECTED_ORDINALS
        and payload.get("question_count") == EXPECTED_PROVIDER_CALLS
        and len(rows) == EXPECTED_PROVIDER_CALLS,
        "missing-four v4 sealed construction boundary changed",
    )
    plans_list: list[dict[str, Any]] = []
    for row, ordinal in zip(rows, EXPECTED_ORDINALS, strict=True):
        plan = base._prompt_plan_row(row, ordinal)  # noqa: SLF001
        terminal = row.get("terminal_prompt")
        _require(type(terminal) is dict, f"v4 terminal changed at {ordinal}")
        assert type(terminal) is dict
        provider_input = terminal.get("provider_input")
        _require(
            type(provider_input) is dict,
            f"v4 provider input changed at {ordinal}",
        )
        assert type(provider_input) is dict
        advisories = provider_input.get("specialist_advisories")
        _require(
            type(advisories) is list and len(advisories) == 1,
            f"v4 requires exactly one scoped advisory at {ordinal}",
        )
        _require(type(advisories[0]) is dict, f"v4 advisory changed at {ordinal}")
        source_terminal_receipt = require_sha256(
            terminal.get("terminal_prompt_receipt_sha256"),
            "v4 terminal prompt",
        )
        transform: dict[str, Any] | None = None
        if ordinal == 42:
            transform, messages, prompt_tokens = _q42_prompt_transform(
                plan,
                advisories[0],
                source_terminal_receipt,
            )
            plan = {
                **plan,
                "messages": list(messages),
                "messages_sha256": identity_sha256(list(messages)),
                "prompt_token_proxy": prompt_tokens,
            }
        scope_body = {
            "adapter_prompt_transform": transform,
            "allowed_handle_ids": list(plan["allowed_handle_ids"]),
            "format": ADVISORY_SCOPE_FORMAT,
            "ordinal": ordinal,
            "specialist_advisories": advisories,
            "specialist_advisories_sha256": identity_sha256(advisories),
            "terminal_kind": require_text(row.get("terminal_kind"), "v4 terminal kind"),
            "terminal_prompt_receipt_sha256": source_terminal_receipt,
        }
        scope = {**scope_body, "receipt_sha256": identity_sha256(scope_body)}
        validation_contract = dict(plan["validation_contract"])
        _require(
            ADVISORY_SCOPE_KEY not in validation_contract,
            "v4 advisory scope key collided with the base contract",
        )
        validation_contract[ADVISORY_SCOPE_KEY] = scope
        plan_body = dict(plan)
        plan_body.pop("prompt_row_receipt_sha256", None)
        plan_body["adapter_prompt_transform"] = transform
        plan_body["validation_contract"] = validation_contract
        plan = {
            **plan_body,
            "prompt_row_receipt_sha256": identity_sha256(plan_body),
        }
        assert_gold_blind(plan, path=f"missing4_v4_answer_plan_{ordinal}")
        plans_list.append(plan)
    plans = tuple(plans_list)
    _require(
        tuple(row.get("ordinal") for row in plans) == EXPECTED_ORDINALS
        and len({row.get("question_id") for row in plans})
        == EXPECTED_PROVIDER_CALLS
        and len({row.get("messages_sha256") for row in plans})
        == EXPECTED_PROVIDER_CALLS,
        "missing-four v4 requires four distinct ordinary provider prompts",
    )
    return artifact, plans


def _advisory_scope(
    validation_contract: Mapping[str, Any],
    allowed_handle_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = dict(validation_contract)
    raw_scope = contract.pop(ADVISORY_SCOPE_KEY, None)
    _require(type(raw_scope) is dict, "missing-four v4 advisory scope is absent")
    assert type(raw_scope) is dict
    scope = dict(raw_scope)
    declared = require_sha256(
        scope.pop("receipt_sha256", None), "missing-four v4 advisory scope"
    )
    advisories = scope.get("specialist_advisories")
    allowed = tuple(allowed_handle_ids)
    _require(
        identity_sha256(scope) == declared
        and scope.get("format") == ADVISORY_SCOPE_FORMAT
        and tuple(scope.get("allowed_handle_ids", ())) == allowed
        and scope.get("ordinal") in EXPECTED_ORDINALS
        and type(advisories) is list
        and len(advisories) == 1
        and scope.get("specialist_advisories_sha256")
        == identity_sha256(advisories),
        "missing-four v4 advisory scope receipt changed",
    )
    assert type(advisories) is list and type(advisories[0]) is dict
    return scope, dict(advisories[0]), contract


def _invalid(code: str) -> ParsedTypedFinalDecision:
    return typed_final._invalid_decision(f"missing4_v4_{code}")  # noqa: SLF001


def _valid_replace(
    prediction: str,
    used_handle_ids: Sequence[str],
    basis: str,
) -> ParsedTypedFinalDecision:
    used = tuple(used_handle_ids)
    receipt = identity_sha256(
        {
            "decision": "replace",
            "format": DECISION_FORMAT,
            "prediction_sha256": quote_sha256(prediction),
            "used_handle_ids": list(used),
            "validation_basis": basis,
            "validator_policy_format": VALIDATOR_POLICY_FORMAT,
        }
    )
    return ParsedTypedFinalDecision(
        True,
        "replace",
        prediction,
        used,
        basis,
        "none",
        receipt,
    )


def _strict_completion(
    completion: str,
    *,
    parent_prediction: str,
    allowed_handle_ids: Sequence[str],
) -> tuple[str, str, tuple[str, ...]] | ParsedTypedFinalDecision:
    try:
        raw = json.loads(
            completion,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (json.JSONDecodeError, ValueError):
        return _invalid("invalid_json")
    if type(raw) is not dict or set(raw) != {
        "decision",
        "prediction",
        "used_handle_ids",
    }:
        return _invalid("root_schema")
    decision = raw["decision"]
    prediction = raw["prediction"]
    used = raw["used_handle_ids"]
    if (
        decision not in {"keep_parent", "replace"}
        or type(prediction) is not str
        or type(used) is not list
        or any(type(value) is not str for value in used)
        or len(set(used)) != len(used)
        or not set(used) <= set(allowed_handle_ids)
    ):
        return _invalid("value_schema")
    if decision != "replace":
        return _invalid("residual_keep_parent_disallowed")
    if (
        not prediction
        or prediction.strip() != prediction
        or prediction == parent_prediction
        or not used
    ):
        return _invalid("replace_contract")
    return decision, prediction, tuple(used)


def _q42_completion(
    prediction: str,
    used: tuple[str, ...],
    allowed: tuple[str, ...],
    advisory: Mapping[str, Any],
) -> ParsedTypedFinalDecision:
    decision = advisory.get("conjunctive_event_decision_state")
    program = advisory.get("conjunctive_event_program")
    frontier = advisory.get("support_frontier")
    _require(
        advisory.get("proof_kind") == "same_event_conjunctive_obligation"
        and type(decision) is dict
        and decision.get("disposition") == "keep_parent"
        and decision.get("terminal_authorized") is False
        and str(decision.get("reason", "")).startswith("support_open_event_")
        and type(program) is dict
        and type(program.get("obligations")) is list
        and type(frontier) is dict
        and frontier.get("generic_frontier_closed") is False
        and frontier.get("semantic_absence_may_be_inferred") is False,
        "q42 advisory escaped its open conjunctive-event scope",
    )
    expected = _q42_expected_text(advisory)
    if prediction != expected:
        return _invalid("q42_scoped_insufficiency_text")
    # These handles may deliberately belong to different story groups: their
    # failure to form one proven event is the conclusion being cited.
    if set(used) != set(allowed):
        return _invalid("q42_scoped_insufficiency_provenance")
    return _valid_replace(
        prediction,
        used,
        "missing4_v4_conjunctive_scoped_insufficiency",
    )


def _q65_completion(
    prediction: str,
    used: tuple[str, ...],
    allowed: tuple[str, ...],
    advisory: Mapping[str, Any],
) -> ParsedTypedFinalDecision:
    expected_handles = advisory.get("used_handle_ids")
    facts = advisory.get("facts")
    cardinality = advisory.get("cardinality")
    _require(
        advisory.get("proof_kind") == "selected_scope_action_member_cardinality"
        and advisory.get("status") == "selected_scope_supported"
        and advisory.get("scope") == "selected_action_linked_members_only"
        and advisory.get("generic_frontier_closed") is False
        and advisory.get("upstream_truncated") is True
        and advisory.get("selected_scope_cardinality_satisfied") is True
        and type(cardinality) is int
        and cardinality >= 1
        and type(facts) is list
        and len(facts) == cardinality
        and type(expected_handles) is list
        and len(set(expected_handles)) == len(expected_handles)
        and set(expected_handles) == set(allowed),
        "q65 selected-scope advisory changed",
    )
    if prediction != advisory.get("prediction"):
        return _invalid("q65_selected_scope_prediction")
    if set(used) != set(expected_handles):
        return _invalid("q65_selected_scope_provenance")
    return _valid_replace(
        prediction,
        used,
        "missing4_v4_selected_scope_action_set_agreement",
    )


def _winner_numeric_contract(
    contract: Mapping[str, Any],
    winner_handle: str,
) -> tuple[float, str]:
    by_handle = contract.get("by_handle")
    _require(
        type(by_handle) is dict and winner_handle in by_handle,
        "q79 winner is absent from validation evidence",
    )
    winner = by_handle[winner_handle]
    _require(type(winner) is dict, "q79 winner validation row changed")
    numeric_rows = winner.get("numeric_value_rows")
    _require(
        type(numeric_rows) is list
        and len(numeric_rows) == 1
        and type(numeric_rows[0]) is dict,
        "q79 winner must bind one exact numeric answer row",
    )
    numeric = numeric_rows[0]
    value = numeric.get("numeric_value")
    unit = numeric.get("unit")
    _require(
        type(value) in {int, float}
        and numeric.get("numeric_qualifier") == NumericQualifier.EXACT.value
        and unit == "$",
        "q79 winner numeric value/currency changed",
    )
    return float(value), unit


def _q79_completion(
    prediction: str,
    used: tuple[str, ...],
    allowed: tuple[str, ...],
    advisory: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> ParsedTypedFinalDecision:
    bundle = advisory.get("temporal_bundle")
    candidate_map = advisory.get("candidate_handle_map")
    _require(
        type(bundle) is dict
        and str(bundle.get("route", "")).startswith("temporal_")
        and type(candidate_map) is dict
        and type(bundle.get("winner_candidate_id")) is str
        and type(bundle.get("winner_handle_id")) is str
        and candidate_map.get(bundle["winner_candidate_id"])
        == bundle["winner_handle_id"]
        and bundle["winner_handle_id"] in allowed,
        "q79 temporal winner advisory changed",
    )
    winner = str(bundle["winner_handle_id"])
    if used != (winner,):
        return _invalid("q79_temporal_winner_scope")
    expected_value, expected_unit = _winner_numeric_contract(contract, winner)
    mentions = numeric_mentions(
        prediction,
        expected_dimension=NumericDimension.CURRENCY,
    )
    if (
        len(mentions) != 1
        or abs(mentions[0].value - expected_value) > 1e-9
        or mentions[0].qualifier is not NumericQualifier.EXACT
        or mentions[0].unit != expected_unit
    ):
        return _invalid("q79_temporal_winner_entailment")
    return _valid_replace(
        prediction,
        used,
        "missing4_v4_temporal_winner_agreement",
    )


def _q74_resource_contract(
    preservation_requirements: Mapping[str, Any],
    lane_handle_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    by_handle = preservation_requirements.get("by_handle")
    _require(type(by_handle) is dict, "q74 preservation rows changed")
    titles: list[str] = []
    urls: list[str] = []
    answer_handles: list[str] = []
    for handle in lane_handle_ids:
        row = by_handle.get(handle)
        _require(type(row) is dict, "q74 lane preservation row changed")
        handle_titles = row.get("exact_title_anchors", ())
        handle_urls = row.get("exact_identifier_anchors", ())
        _require(
            isinstance(handle_titles, (list, tuple))
            and isinstance(handle_urls, (list, tuple))
            and all(type(value) is str and bool(value) for value in handle_titles)
            and all(type(value) is str and bool(value) for value in handle_urls),
            "q74 exact resource anchors changed",
        )
        titles.extend(handle_titles)
        urls.extend(handle_urls)
        if handle_titles and handle_urls:
            answer_handles.append(handle)
    _require(
        bool(titles) and bool(urls) and bool(answer_handles),
        "q74 lane lost its exact title/URL-bearing evidence",
    )
    return (
        tuple(dict.fromkeys(titles)),
        tuple(dict.fromkeys(urls)),
        tuple(answer_handles),
    )


def _q74_has_exact_resource(
    prediction: str,
    exact_titles: Sequence[str],
    exact_urls: Sequence[str],
) -> bool:
    folded = prediction.casefold()
    return all(title.casefold() in folded for title in exact_titles) and set(
        exact_urls
    ) <= set(typed_final._exact_urls(prediction))  # noqa: SLF001


def parse_v4_completion(
    completion: str,
    *,
    parent_prediction: str,
    allowed_handle_ids: Sequence[str],
    handle_group_by_id: Mapping[str, str],
    story_coherence: Mapping[str, Any],
    preservation_requirements: Mapping[str, Any],
    validation_contract: Mapping[str, Any],
) -> ParsedTypedFinalDecision:
    """Validate one generic prompt response against its sealed v4 advisory."""

    scope, advisory, base_contract = _advisory_scope(
        validation_contract, allowed_handle_ids
    )
    allowed = tuple(allowed_handle_ids)
    ordinal = scope["ordinal"]
    terminal_kind = scope["terminal_kind"]
    expected_kind = {
        42: "conjunctive_event_synthesis",
        65: "selected_scope_action_set_synthesis",
        74: "semantic_residual_synthesis",
        79: "temporal_specialist_synthesis",
    }[ordinal]
    _require(terminal_kind == expected_kind, f"v4 terminal kind changed at {ordinal}")
    if ordinal == 74:
        # The sealed parent already contains the exact Mayo title+URL.  Both a
        # validated keep and a validated replacement are acceptable here.
        generic = _BASE_PARSE_COMPLETION(
            completion,
            parent_prediction=parent_prediction,
            allowed_handle_ids=allowed,
            handle_group_by_id=handle_group_by_id,
            story_coherence=story_coherence,
            preservation_requirements=preservation_requirements,
            validation_contract=base_contract,
        )
        lane = advisory.get("lane_handle_ids")
        _require(
            advisory.get("proof_kind") == "sealed_semantic_residual_lane"
            and advisory.get("sealed_lane_complete") is True
            and advisory.get("global_exhaustiveness_claimed") is False
            and advisory.get("truncated") is False
            and type(lane) is list
            and set(lane) == set(allowed),
            "q74 sealed residual advisory changed",
        )
        if not generic.valid:
            return generic
        exact_titles, exact_urls, answer_handles = _q74_resource_contract(
            preservation_requirements,
            lane,
        )
        if generic.decision == "keep_parent":
            _require(
                _q74_has_exact_resource(
                    parent_prediction,
                    exact_titles,
                    exact_urls,
                ),
                "q74 protected parent lost its exact title or URL",
            )
            return generic
        if (
            not set(generic.used_handle_ids) <= set(lane)
            or not set(answer_handles) <= set(generic.used_handle_ids)
        ):
            return _invalid("q74_residual_lane_scope")
        if not _q74_has_exact_resource(
            generic.prediction,
            exact_titles,
            exact_urls,
        ):
            return _invalid("q74_exact_resource_anchor_loss")
        return _valid_replace(
            generic.prediction,
            generic.used_handle_ids,
            "missing4_v4_semantic_residual_entailment",
        )

    parsed = _strict_completion(
        completion,
        parent_prediction=parent_prediction,
        allowed_handle_ids=allowed,
    )
    if type(parsed) is ParsedTypedFinalDecision:
        return parsed
    _decision, prediction, used = parsed
    if ordinal == 42:
        return _q42_completion(prediction, used, allowed, advisory)
    if ordinal == 65:
        return _q65_completion(prediction, used, allowed, advisory)
    if ordinal == 79:
        return _q79_completion(prediction, used, allowed, advisory, base_contract)
    raise AssertionError("unreachable v4 ordinal")


def validate_preflight_artifact(
    artifact: SealedArtifact,
) -> tuple[tuple[tuple[dict[str, str], ...], ...], tuple[dict[str, Any], ...]]:
    construction_sha = require_sha256(
        artifact.payload.get("construction_artifact_sha256"),
        "v4 preflight construction",
    )
    previous = base.EXPECTED_CONSTRUCTION_SHA256
    base.EXPECTED_CONSTRUCTION_SHA256 = construction_sha
    try:
        prompts, rows = _BASE_VALIDATE_PREFLIGHT(artifact)
    finally:
        base.EXPECTED_CONSTRUCTION_SHA256 = previous
    observed = max(
        int(row["prompt_token_proxy"]) + OUTPUT_TOKEN_RESERVE for row in rows
    )
    _require(
        artifact.payload.get("hard_complete_chat_token_cap")
        == HARD_COMPLETE_CHAT_TOKEN_CAP
        and artifact.payload.get("observed_max_complete_envelope_tokens") == observed
        and observed <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "v4 preflight complete-envelope budget changed",
    )
    return prompts, rows


def materialization_projection(
    preflight: SealedArtifact,
    prompt_rows: tuple[dict[str, Any], ...],
    batch: Any,
) -> dict[str, Any]:
    construction_sha = require_sha256(
        preflight.payload.get("construction_artifact_sha256"),
        "v4 materialization construction",
    )
    previous = base.EXPECTED_CONSTRUCTION_SHA256
    base.EXPECTED_CONSTRUCTION_SHA256 = construction_sha
    try:
        payload = _BASE_MATERIALIZATION_PROJECTION(preflight, prompt_rows, batch)
    finally:
        base.EXPECTED_CONSTRUCTION_SHA256 = previous
    questions = payload.get("questions")
    _require(
        type(questions) is list
        and len(questions) == EXPECTED_PROVIDER_CALLS
        and all(
            row.get("solver_valid") is True
            and (
                row.get("decision") == "replace"
                or row.get("ordinal") == 74
                and row.get("decision") == "keep_parent"
            )
            and row.get("prediction_source")
            != "typed_final_invalid_keep_parent_v1"
            for row in questions
            if type(row) is dict
        )
        and len(questions) == sum(type(row) is dict for row in questions),
        "v4 residual materialization refuses a silent parent fallback",
    )
    return payload


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
    """Build the shared checkpoint runtime with v4-only provenance."""

    _require(
        model == DEFAULT_MODEL == artifact.payload.get("model")
        and gateway_url == artifact.payload.get("gateway_url")
        and max_concurrency == artifact.payload.get("max_concurrency")
        and len(prompts) == EXPECTED_PROVIDER_CALLS,
        "missing-four v4 runtime differs from its sealed preflight",
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
            "arm": "reduced_missing4_scoped_terra_answer_v4",
            "authorized_unique_calls": EXPECTED_PROVIDER_CALLS,
            "construction_artifact_sha256": artifact.payload[
                "construction_artifact_sha256"
            ],
            "experiment_format": FORMAT,
            "gateway_url": gateway_url,
            "gold_loaded": False,
            "preflight_artifact_sha256": artifact.sha256,
            "validator_policy_format": VALIDATOR_POLICY_FORMAT,
        },
    )


_BASE_LOCK = RLock()


def _base_globals() -> dict[str, Any]:
    return {
        "CHECKPOINT_DIR_NAME": CHECKPOINT_DIR_NAME,
        "DEFAULT_CONSTRUCTION": DEFAULT_CONSTRUCTION,
        "DEFAULT_MODEL": DEFAULT_MODEL,
        "DEFAULT_OUTPUT": DEFAULT_OUTPUT,
        "EXPECTED_CONSTRUCTION_SHA256": "0" * 64,
        "EXPECTED_ORDINALS": EXPECTED_ORDINALS,
        "EXPECTED_PROVIDER_CALLS": EXPECTED_PROVIDER_CALLS,
        "FORMAT": FORMAT,
        "PREFLIGHT_FORMAT": PREFLIGHT_FORMAT,
        "PREFLIGHT_NAME": PREFLIGHT_NAME,
        "REPLAY_NAME": REPLAY_NAME,
        "RUN_NAME": RUN_NAME,
        "VALIDATOR_POLICY_FORMAT": VALIDATOR_POLICY_FORMAT,
        "_materialization_projection": materialization_projection,
        "_read_construction": load_verified_construction,
        "_runtime": build_runtime,
        "_validate_preflight": validate_preflight_artifact,
        "parse_typed_final_completion": parse_v4_completion,
    }


@contextmanager
def _v4_base_contract() -> Iterator[None]:
    """Temporarily version the existing answer engine; always restore it."""

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


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    with _v4_base_contract():
        return base.run_preflight(args)


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    with _v4_base_contract():
        return base.run_provider(args)


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    with _v4_base_contract():
        return base.run_materialize(args)


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    with _v4_base_contract():
        return base.run_replay(args)


def _add_runtime_settings(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gateway-url", default=base.live.DEFAULT_GATEWAY_URL)
    parser.add_argument("--max-concurrency", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="seal four scoped prompts")
    _add_runtime_settings(preflight)
    preflight.add_argument("--construction", type=Path, default=DEFAULT_CONSTRUCTION)
    preflight.add_argument("--expected-construction-sha256", required=True)

    provider = commands.add_parser("provider-run", help="execute four Terra prompts")
    _add_runtime_settings(provider)
    provider.add_argument("--expected-preflight-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, default=0)
    provider.add_argument("--api-key-env", default=base.live.DEFAULT_API_KEY_ENV)

    materialize = commands.add_parser(
        "materialize", help="materialize checkpoint-only scoped predictions"
    )
    _add_runtime_settings(materialize)
    materialize.add_argument("--expected-preflight-sha256", required=True)

    replay = commands.add_parser("replay", help="prove byte-identical v4 replay")
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
    "EXPECTED_ORDINALS",
    "EXPECTED_PROVIDER_CALLS",
    "FORMAT",
    "PREFLIGHT_NAME",
    "REPLAY_NAME",
    "RUN_NAME",
    "ReducedMissing4V4AnswerError",
    "build_parser",
    "build_runtime",
    "load_verified_construction",
    "main",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
]
