#!/usr/bin/env python3
"""Build and replay the sealed exact-11 cumulative P/R/L/G terminal assay."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.domain._tokenizer import (  # noqa: E402
    count_chat_prompt_token_proxy,
)
from memory_condense.domain.discourse import quote_sha256  # noqa: E402
from tools import run_locked_semantic_residual_construction_v4 as r7_cli  # noqa: E402
from tools import run_reduced_semantic_global_completion_assay as v7_cli  # noqa: E402
from tools import run_reduced_source_group_reinjection_assay as v6_cli  # noqa: E402
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
from tools.matched_eval.semantic_global_terminal_adapter import (  # noqa: E402
    EXACT_SPAN_SUPPORT_FORMAT,
    EXACT_SPAN_SUPPORT_POPULATION_FORMAT,
    EXACT_SPAN_SUPPORT_RANKING_POLICY,
    FORMAT as TERMINAL_COMPILATION_FORMAT,
    HARD_PROMPT_TOKEN_CAP,
    OUTPUT_TOKEN_RESERVE,
    PLANE_ORDER,
    ExactSpanSupportAuthority,
    ExactSpanSupportPopulationReceipt,
    SemanticGlobalTerminalPolicy,
    TerminalSealedSources,
    compile_semantic_global_terminal,
    load_selected_protected_owner_evidence,
    replay_semantic_global_terminal,
)
from tools.matched_eval.typed_memory_final_arm import (  # noqa: E402
    render_final_messages,
)


FORMAT = "memory-condense-reduced-semantic-global-terminal-assay-v2"
ANSWER_PLAN_FORMAT = f"{FORMAT}-answer-plan-v2"
CONSTRUCTION_NAME = "reduced-semantic-global-terminal-assay-v2.json"
REPLAY_NAME = "reduced-semantic-global-terminal-assay-replay-v2.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-semantic-global-terminal-v2-r1"
)
DEFAULT_V7_SOURCE_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-semantic-global-completion-v7-r2"
)
DEFAULT_V7_ASSAY_SHA256 = (
    "acdae80a84ef94f0ceaaea3f9d9d58d69d15023edd68b45758b3aa38bdd93f95"
)
EXACT_ORDINALS = v7_cli.DEFAULT_ORDINALS
ROUTE_ID = "semantic-global-terminal-terra-answer-v2"
TERMINAL_TOP_LEVEL_KEYS = frozenset(
    {
        "base_v7_assay_sha256",
        "base_v7_format",
        "base_v7_replay_sha256",
        "exact_terminal_ordinals",
        "terminal_answer_plan_count",
        "terminal_policy",
        "terminal_source_artifact_bindings",
        "terminalized_resident_pass_identity_sha256",
    }
)

ANSWER_PLAN_KEYS = frozenset(
    {
        "allowed_handle_ids",
        "answer_plan_receipt_sha256",
        "dated_question",
        "dated_question_sha256",
        "format",
        "handle_group_by_id",
        "hard_prompt_token_cap",
        "messages_sha256",
        "ordinal",
        "output_token_reserve",
        "parent_prediction",
        "parent_prediction_sha256",
        "preservation_requirements",
        "prompt_token_proxy",
        "provider_input",
        "provider_input_sha256",
        "question_id",
        "question_sha256",
        "route_id",
        "source_artifact_bindings",
        "story_coherence",
        "terminal_compilation",
        "terminal_compilation_receipt_sha256",
        "validation_contract",
    }
)

EXACT_SPAN_SUPPORT_AUTHORITY_KEYS = frozenset(
    {
        "authority_candidate_receipt_sha256s",
        "authority_source_planes",
        "exact_relation_support",
        "exact_span_identity_sha256",
        "format",
        "matched_query_actions",
        "past_event_witness",
        "policy",
        "priority_prefix",
        "query_temporal_support",
        "receipt_sha256",
        "role",
        "source_group_supported_kinds",
        "source_group_supported_obligation_ids",
        "supported_obligation_ids",
    }
)
EXACT_SPAN_SUPPORT_POPULATION_KEYS = frozenset(
    {
        "authorities",
        "format",
        "plane_candidate_receipt_sha256s",
        "plane_selection_receipt_sha256s",
        "policy",
        "receipt_sha256",
    }
)


class ReducedSemanticGlobalTerminalAssayError(MatchedEvalContractError):
    """A terminal compilation, exact-11 binding, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ReducedSemanticGlobalTerminalAssayError(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _exact_int(value: object, label: str) -> int:
    _require(type(value) is int, f"{label} must be an exact integer")
    return value  # type: ignore[return-value]


def _with_receipt(
    body: Mapping[str, Any], key: str = "receipt_sha256"
) -> dict[str, Any]:
    return {**dict(body), key: identity_sha256(body)}


def _compile_answer_plan_core(
    *,
    dated_question: str,
    parent_prediction: str,
    residual_index: Any,
    query: Any,
    protected_owner_universe_bindings: Sequence[Any],
    selected_protected_owner_evidence_rows: Sequence[Mapping[str, Any]],
    residual_result: Any,
    local_result: Any,
    global_result: Any,
    sealed_sources: TerminalSealedSources,
    policy: SemanticGlobalTerminalPolicy,
) -> dict[str, Any]:
    """Compile only gold-blind mechanism inputs; identity wrapping is external."""

    upstream_projection_sha256s = {
        "global": identity_sha256(global_result.projection()),
        "local": identity_sha256(local_result.projection()),
        "residual": identity_sha256(residual_result.projection()),
    }
    selected_owners = load_selected_protected_owner_evidence(
        selected_protected_owner_evidence_rows
    )
    compilation = compile_semantic_global_terminal(
        dated_question=dated_question,
        parent_prediction=parent_prediction,
        residual_index=residual_index,
        query=query,
        protected_owner_universe_bindings=protected_owner_universe_bindings,
        selected_protected_owner_evidence=selected_owners,
        residual_result=residual_result,
        local_result=local_result,
        global_result=global_result,
        sealed_sources=sealed_sources,
        policy=policy,
    )
    replayed = replay_semantic_global_terminal(
        dated_question=dated_question,
        parent_prediction=parent_prediction,
        residual_index=residual_index,
        query=query,
        protected_owner_universe_bindings=protected_owner_universe_bindings,
        selected_protected_owner_evidence=selected_owners,
        residual_result=residual_result,
        local_result=local_result,
        global_result=global_result,
        sealed_sources=sealed_sources,
        sealed_compilation=compilation,
        policy=policy,
    )
    _require(
        replayed.projection(include_local=True)
        == compilation.projection(include_local=True),
        "terminal compilation changed during immediate resident replay",
    )
    provider_input = compilation.provider_projection()
    messages = render_final_messages(provider_input)
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    _require(
        tuple(messages) == compilation.fitted.messages
        and identity_sha256(list(messages))
        == compilation.fitted.projection()["messages_sha256"]
        and prompt_tokens == compilation.fitted.prompt_token_proxy
        and prompt_tokens + OUTPUT_TOKEN_RESERVE <= HARD_PROMPT_TOKEN_CAP,
        "terminal answer-plan messages or hard token envelope changed",
    )
    body = {
        "allowed_handle_ids": list(compilation.fitted.allowed_handle_ids),
        "dated_question": dated_question,
        "format": ANSWER_PLAN_FORMAT,
        "handle_group_by_id": dict(compilation.fitted.handle_group_by_id),
        "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
        "messages_sha256": identity_sha256(list(messages)),
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "parent_prediction": parent_prediction,
        "parent_prediction_sha256": quote_sha256(parent_prediction),
        "preservation_requirements": dict(
            compilation.fitted.preservation_requirements
        ),
        "prompt_token_proxy": prompt_tokens,
        "provider_input": provider_input,
        "provider_input_sha256": identity_sha256(provider_input),
        "route_id": ROUTE_ID,
        "source_artifact_bindings": sealed_sources.projection(),
        "story_coherence": dict(compilation.fitted.story_coherence),
        "terminal_compilation": compilation.projection(include_local=True),
        "terminal_compilation_receipt_sha256": compilation.receipt_sha256,
        "validation_contract": dict(compilation.fitted.validation_contract),
    }
    _require(
        upstream_projection_sha256s
        == {
            "global": identity_sha256(global_result.projection()),
            "local": identity_sha256(local_result.projection()),
            "residual": identity_sha256(residual_result.projection()),
        },
        "terminal compiler mutated an authenticated upstream retrieval result",
    )
    assert_gold_blind(body, path="semantic_global_terminal_answer_plan_core")
    return body


def _project_base_v7(terminalized: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only terminal additions and reconstruct exact frozen V7 receipts."""

    question_rows: list[dict[str, Any]] = []
    receipt_by_ordinal: dict[int, str] = {}
    for raw in _exact_list(terminalized.get("questions"), "terminalized questions"):
        question = _exact_dict(raw, "terminalized question")
        body = {
            key: value
            for key, value in question.items()
            if key not in {"terminal_answer_plan", "question_assay_receipt_sha256"}
        }
        rebuilt = _with_receipt(body, "question_assay_receipt_sha256")
        ordinal = rebuilt.get("ordinal")
        _require(type(ordinal) is int and ordinal not in receipt_by_ordinal, "V7 ordinal changed")
        receipt_by_ordinal[ordinal] = rebuilt["question_assay_receipt_sha256"]
        question_rows.append(rebuilt)

    namespace_rows: list[dict[str, Any]] = []
    ordinals_by_namespace: dict[str, list[int]] = {}
    for question in question_rows:
        ordinals_by_namespace.setdefault(str(question["namespace_id"]), []).append(
            int(question["ordinal"])
        )
    for raw in _exact_list(
        terminalized.get("namespace_receipts"), "terminalized namespaces"
    ):
        row = _exact_dict(raw, "terminalized namespace")
        namespace_id = str(row.get("namespace_id"))
        body = {
            key: value
            for key, value in row.items()
            if key != "namespace_assay_receipt_sha256"
        }
        body["question_assay_receipt_sha256s"] = [
            receipt_by_ordinal[ordinal]
            for ordinal in ordinals_by_namespace.get(namespace_id, [])
        ]
        namespace_rows.append(_with_receipt(body, "namespace_assay_receipt_sha256"))

    body = {
        key: value
        for key, value in terminalized.items()
        if key != "construction_identity_sha256"
        and key not in TERMINAL_TOP_LEVEL_KEYS
    }
    body["format"] = v7_cli.FORMAT
    body["questions"] = question_rows
    body["namespace_receipts"] = namespace_rows
    return {**body, "construction_identity_sha256": identity_sha256(body)}


def _first_projection_difference(
    left: object,
    right: object,
    *,
    path: str = "$",
) -> Mapping[str, str] | None:
    """Return a content-blind first-difference proof for a failed parent check."""

    if type(left) is not type(right):
        return {
            "left_type": type(left).__name__,
            "path": path,
            "right_type": type(right).__name__,
        }
    if type(left) is dict:
        left_mapping = left
        right_mapping = right
        left_keys = set(left_mapping)
        right_keys = set(right_mapping)
        if left_keys != right_keys:
            return {
                "left_keys_sha256": identity_sha256(sorted(left_keys)),
                "path": path,
                "right_keys_sha256": identity_sha256(sorted(right_keys)),
            }
        for key in sorted(
            left_keys,
            key=lambda value: (str(value).endswith("_sha256"), str(value)),
        ):
            difference = _first_projection_difference(
                left_mapping[key],
                right_mapping[key],
                path=f"{path}.{key}",
            )
            if difference is not None:
                return difference
        return None
    if type(left) is list:
        left_values = left
        right_values = right
        if len(left_values) != len(right_values):
            return {
                "left_length": str(len(left_values)),
                "path": path,
                "right_length": str(len(right_values)),
            }
        for index, (left_value, right_value) in enumerate(
            zip(left_values, right_values, strict=True)
        ):
            difference = _first_projection_difference(
                left_value,
                right_value,
                path=f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if left != right:
        return {
            "left_sha256": identity_sha256(left),
            "path": path,
            "right_sha256": identity_sha256(right),
        }
    return None


def _verified_v7_sources(
    args: argparse.Namespace,
) -> tuple[SealedArtifact, SealedArtifact]:
    construction = read_sealed_json(Path(args.v7_construction))
    replay = read_sealed_json(Path(args.v7_replay))
    _require(
        construction.sha256
        == require_sha256(str(args.expected_v7_assay_sha256), "V7 construction")
        and replay.sha256
        == require_sha256(str(args.expected_v7_replay_sha256), "V7 replay")
        and construction.sha256 == replay.sha256
        and construction.payload == replay.payload
        and construction.payload.get("format") == v7_cli.FORMAT
        and tuple(
            row.get("ordinal")
            for row in _exact_list(
                construction.payload.get("questions"), "V7 source questions"
            )
            if type(row) is dict
        )
        == EXACT_ORDINALS,
        "frozen V7 construction/replay binding changed",
    )
    return construction, replay


def _require_frozen_episode_resolution(
    frozen_v7_payload: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    """Bind the resident rebuild to the frozen V7 episode-resolution mode.

    The exact-11 V7 parent resolved one authenticated fixed-interval artifact
    independently inside every namespace.  Inheriting the generic V6 default
    (episodes disabled) changes L, and can consequently change whether a G row
    is novel or an exact protected duplicate.  Reject that configuration
    before rebuilding any namespace rather than discovering it at the final
    parent-projection gate.
    """

    namespace_rows = _exact_list(
        frozen_v7_payload.get("namespace_receipts"),
        "frozen V7 namespaces",
    )
    modes = {
        require_text(
            _exact_dict(
                _exact_dict(row, "frozen V7 namespace").get(
                    "episode_artifact_binding"
                ),
                "frozen V7 episode binding",
            ).get("resolution_mode"),
            "frozen V7 episode resolution mode",
        )
        for row in namespace_rows
    }
    _require(
        modes == {"authenticated_namespace_fixed_interval_auto"}
        and bool(getattr(args, "auto_resolve_episode_artifact", False))
        and getattr(args, "episode_artifact_id", None) is None,
        "terminal episode resolution must reproduce frozen V7 authenticated auto bindings",
    )


def build_assay(args: argparse.Namespace) -> dict[str, Any]:
    v7_construction, v7_replay = _verified_v7_sources(args)
    _require_frozen_episode_resolution(v7_construction.payload, args)
    r7_artifact = v6_cli._verified_r7_construction(args)  # noqa: SLF001
    gate, _sources = r7_cli._load_verified_gate(args)  # noqa: SLF001
    sealed_sources = TerminalSealedSources(
        protected_owner_artifact_sha256=r7_artifact.sha256,
        residual_artifact_sha256=r7_artifact.sha256,
        parent_artifact_sha256=gate.sha256,
    )
    policy = SemanticGlobalTerminalPolicy()
    terminalized = v7_cli.build_assay(
        args,
        terminal_compiler=partial(
            _compile_answer_plan_core,
            sealed_sources=sealed_sources,
            policy=policy,
        ),
    )
    projected_v7 = _project_base_v7(terminalized)
    projected_questions = {
        int(row["ordinal"]): row
        for row in _exact_list(projected_v7.get("questions"), "projected V7 questions")
    }
    frozen_questions = {
        int(row["ordinal"]): row
        for row in _exact_list(
            v7_construction.payload.get("questions"), "frozen V7 questions"
        )
    }
    _require(
        projected_questions.keys() == frozen_questions.keys(),
        "terminal-stripped V7 question population changed",
    )
    for ordinal in EXACT_ORDINALS:
        question_difference = _first_projection_difference(
            projected_questions[ordinal],
            frozen_questions[ordinal],
        )
        _require(
            question_difference is None,
            f"terminal-stripped V7 question {ordinal} changed; "
            f"first_difference={json.dumps(question_difference, sort_keys=True)}",
        )
    projected_namespaces = {
        str(row["namespace_id"]): row
        for row in _exact_list(
            projected_v7.get("namespace_receipts"), "projected V7 namespaces"
        )
    }
    frozen_namespaces = {
        str(row["namespace_id"]): row
        for row in _exact_list(
            v7_construction.payload.get("namespace_receipts"),
            "frozen V7 namespaces",
        )
    }
    _require(
        projected_namespaces.keys() == frozen_namespaces.keys(),
        "terminal-stripped V7 namespace population changed",
    )
    for namespace_id in sorted(projected_namespaces):
        namespace_difference = _first_projection_difference(
            projected_namespaces[namespace_id],
            frozen_namespaces[namespace_id],
        )
        _require(
            namespace_difference is None,
            "terminal-stripped V7 namespace changed; "
            f"first_difference={json.dumps(namespace_difference, sort_keys=True)}",
        )
    projected_top = {
        key: value
        for key, value in projected_v7.items()
        if key not in {"construction_identity_sha256", "namespace_receipts", "questions"}
    }
    frozen_top = {
        key: value
        for key, value in v7_construction.payload.items()
        if key not in {"construction_identity_sha256", "namespace_receipts", "questions"}
    }
    top_difference = _first_projection_difference(projected_top, frozen_top)
    _require(
        top_difference is None,
        "terminal-stripped V7 top-level body changed; "
        f"first_difference={json.dumps(top_difference, sort_keys=True)}",
    )
    parent_difference = _first_projection_difference(
        projected_v7,
        v7_construction.payload,
    )
    _require(
        parent_difference is None,
        "terminalized resident pass does not reduce to the frozen V7 assay; "
        f"first_difference={json.dumps(parent_difference, sort_keys=True)}",
    )
    questions = _exact_list(terminalized.get("questions"), "terminal questions")
    _require(
        len(questions) == len(EXACT_ORDINALS)
        and tuple(
            _exact_dict(row, "terminal question").get("ordinal")
            for row in questions
        )
        == EXACT_ORDINALS
        and all(
            type(_exact_dict(row, "terminal question").get("terminal_answer_plan"))
            is dict
            for row in questions
        ),
        "terminal answer-plan population changed",
    )
    body = {
        key: value
        for key, value in terminalized.items()
        if key not in {"construction_identity_sha256", "format"}
    }
    body.update(
        {
            "base_v7_assay_sha256": v7_construction.sha256,
            "base_v7_format": v7_cli.FORMAT,
            "base_v7_replay_sha256": v7_replay.sha256,
            "exact_terminal_ordinals": list(EXACT_ORDINALS),
            "format": FORMAT,
            "terminal_answer_plan_count": len(questions),
            "terminal_policy": policy.projection(),
            "terminal_source_artifact_bindings": sealed_sources.projection(),
            "terminalized_resident_pass_identity_sha256": terminalized[
                "construction_identity_sha256"
            ],
        }
    )
    assert_gold_blind(body, path="reduced_semantic_global_terminal_assay")
    return {**body, "construction_identity_sha256": identity_sha256(body)}


def _validate_exact_span_support_population(
    value: object,
) -> ExactSpanSupportPopulationReceipt:
    population = _exact_dict(value, "terminal exact-span support population")
    _require(
        set(population) == EXACT_SPAN_SUPPORT_POPULATION_KEYS
        and population.get("format") == EXACT_SPAN_SUPPORT_POPULATION_FORMAT
        and population.get("policy") == EXACT_SPAN_SUPPORT_RANKING_POLICY,
        "terminal exact-span support population schema changed",
    )
    raw_authorities = _exact_list(
        population.get("authorities"),
        "terminal exact-span support authorities",
    )
    authorities: list[ExactSpanSupportAuthority] = []
    for raw in raw_authorities:
        authority = _exact_dict(raw, "terminal exact-span support authority")
        _require(
            set(authority) == EXACT_SPAN_SUPPORT_AUTHORITY_KEYS
            and authority.get("format") == EXACT_SPAN_SUPPORT_FORMAT
            and authority.get("policy") == EXACT_SPAN_SUPPORT_RANKING_POLICY,
            "terminal exact-span support authority schema changed",
        )
        parsed = ExactSpanSupportAuthority(
            exact_span_identity_sha256=require_sha256(
                str(authority.get("exact_span_identity_sha256")),
                "terminal exact-span support identity",
            ),
            authority_candidate_receipt_sha256s=tuple(
                _exact_list(
                    authority.get("authority_candidate_receipt_sha256s"),
                    "terminal exact-span support candidates",
                )
            ),
            authority_source_planes=tuple(
                _exact_list(
                    authority.get("authority_source_planes"),
                    "terminal exact-span support source planes",
                )
            ),
            supported_obligation_ids=tuple(
                _exact_list(
                    authority.get("supported_obligation_ids"),
                    "terminal exact-span direct obligations",
                )
            ),
            source_group_supported_obligation_ids=tuple(
                _exact_list(
                    authority.get("source_group_supported_obligation_ids"),
                    "terminal exact-span source-group obligations",
                )
            ),
            source_group_supported_kinds=tuple(
                _exact_list(
                    authority.get("source_group_supported_kinds"),
                    "terminal exact-span support kinds",
                )
            ),
            matched_query_actions=tuple(
                _exact_list(
                    authority.get("matched_query_actions"),
                    "terminal exact-span query actions",
                )
            ),
            exact_relation_support=authority.get("exact_relation_support"),
            query_temporal_support=authority.get("query_temporal_support"),
            past_event_witness=authority.get("past_event_witness"),
            role=require_text(authority.get("role"), "terminal exact-span role"),
            priority_prefix=tuple(
                _exact_list(
                    authority.get("priority_prefix"),
                    "terminal exact-span priority prefix",
                )
            ),
            receipt_sha256=require_sha256(
                str(authority.get("receipt_sha256")),
                "terminal exact-span support authority",
            ),
        )
        _require(
            parsed.projection() == authority,
            "terminal exact-span support authority projection changed",
        )
        authorities.append(parsed)

    raw_candidates = _exact_dict(
        population.get("plane_candidate_receipt_sha256s"),
        "terminal exact-span support plane candidates",
    )
    _require(
        set(raw_candidates) == set(PLANE_ORDER),
        "terminal exact-span support candidate planes changed",
    )
    parsed_population = ExactSpanSupportPopulationReceipt(
        plane_candidate_receipt_sha256s=tuple(
            (
                plane,
                tuple(
                    _exact_list(
                        raw_candidates.get(plane),
                        f"terminal exact-span support {plane} candidates",
                    )
                ),
            )
            for plane in PLANE_ORDER
        ),
        plane_selection_receipt_sha256s=tuple(
            _exact_list(
                population.get("plane_selection_receipt_sha256s"),
                "terminal exact-span support plane selections",
            )
        ),
        authorities=tuple(authorities),
        receipt_sha256=require_sha256(
            str(population.get("receipt_sha256")),
            "terminal exact-span support population",
        ),
    )
    _require(
        parsed_population.projection() == population,
        "terminal exact-span support population projection changed",
    )
    return parsed_population


def _validate_answer_plan(
    row: Mapping[str, Any],
    question: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _exact_dict(row, "terminal answer plan")
    _require(set(plan) == ANSWER_PLAN_KEYS, "terminal answer-plan schema changed")
    receipt = require_sha256(
        str(plan.get("answer_plan_receipt_sha256")), "terminal answer plan"
    )
    body = {
        key: value
        for key, value in plan.items()
        if key != "answer_plan_receipt_sha256"
    }
    compilation = _exact_dict(
        plan.get("terminal_compilation"), "terminal compilation"
    )
    compilation_body = {
        key: value
        for key, value in compilation.items()
        if key not in {"local_audit", "receipt_sha256"}
    }
    terminal_prompt = _exact_dict(
        compilation.get("terminal_prompt"), "terminal prompt"
    )
    local_audit = _exact_dict(compilation.get("local_audit"), "terminal local audit")
    exact_span_support_population = _validate_exact_span_support_population(
        local_audit.get("exact_span_support_population")
    )
    plane_selections = _exact_list(
        compilation.get("plane_selections"), "terminal plane selections"
    )
    plane_selection_receipts: list[str] = []
    plane_candidate_receipts: list[tuple[str, tuple[str, ...]]] = []
    _require(
        len(plane_selections) == len(PLANE_ORDER),
        "terminal plane selection population changed",
    )
    for expected_plane, raw_selection in zip(
        PLANE_ORDER, plane_selections, strict=True
    ):
        selection = _exact_dict(raw_selection, "terminal plane selection")
        selection_receipt = require_sha256(
            str(selection.get("receipt_sha256")), "terminal plane selection"
        )
        selection_body = {
            key: value
            for key, value in selection.items()
            if key != "receipt_sha256"
        }
        candidate_receipts = tuple(
            require_sha256(str(value), "terminal plane candidate")
            for value in _exact_list(
                selection.get("candidate_receipt_sha256s"),
                "terminal plane candidates",
            )
        )
        _require(
            selection.get("plane") == expected_plane
            and selection_receipt == identity_sha256(selection_body),
            "terminal plane selection receipt/order changed",
        )
        plane_selection_receipts.append(selection_receipt)
        plane_candidate_receipts.append((expected_plane, candidate_receipts))
    local_prompt = _exact_dict(
        local_audit.get("terminal_prompt"), "terminal local prompt"
    )
    global_completion = _exact_dict(
        question.get("global_completion"), "terminal inherited global completion"
    )
    provider_input = _exact_dict(plan.get("provider_input"), "terminal provider input")
    allowed = _exact_list(plan.get("allowed_handle_ids"), "terminal allowed handles")
    handle_groups = _exact_dict(
        plan.get("handle_group_by_id"), "terminal handle groups"
    )
    prompt_tokens = _exact_int(plan.get("prompt_token_proxy"), "terminal prompt tokens")
    output_reserve = _exact_int(
        plan.get("output_token_reserve"), "terminal output reserve"
    )
    hard_cap = _exact_int(plan.get("hard_prompt_token_cap"), "terminal hard cap")
    messages = render_final_messages(provider_input)
    _require(
        receipt == identity_sha256(body)
        and plan.get("format") == ANSWER_PLAN_FORMAT
        and plan.get("route_id") == ROUTE_ID
        and all(
            plan.get(key) == question.get(key)
            for key in (
                "ordinal",
                "question_id",
                "question_sha256",
                "dated_question_sha256",
            )
        )
        and quote_sha256(require_text(plan.get("dated_question"), "dated question"))
        == plan.get("dated_question_sha256")
        and quote_sha256(
            require_text(plan.get("parent_prediction"), "parent prediction")
        )
        == plan.get("parent_prediction_sha256")
        and provider_input.get("dated_question") == plan.get("dated_question")
        and _exact_dict(
            provider_input.get("protected_parent_fallback"),
            "terminal parent fallback",
        ).get("prediction")
        == plan.get("parent_prediction")
        and identity_sha256(provider_input) == plan.get("provider_input_sha256")
        and identity_sha256(list(messages)) == plan.get("messages_sha256")
        and count_chat_prompt_token_proxy(messages) == prompt_tokens
        and prompt_tokens + output_reserve <= hard_cap
        and output_reserve == OUTPUT_TOKEN_RESERVE
        and hard_cap == HARD_PROMPT_TOKEN_CAP
        and len(set(allowed)) == len(allowed)
        and all(type(value) is str and bool(value) for value in allowed)
        and set(handle_groups) == set(allowed)
        and compilation.get("receipt_sha256")
        == plan.get("terminal_compilation_receipt_sha256")
        and identity_sha256(compilation_body)
        == compilation.get("receipt_sha256")
        and compilation.get("format") == TERMINAL_COMPILATION_FORMAT
        and question.get("new_provider_calls") == 0
        and question.get("retained_transformer_token_state_bytes") == 0
        and compilation.get("new_provider_calls") == 0
        and compilation.get("retained_transformer_token_state_bytes") == 0
        and global_completion.get("new_provider_calls") == 0
        and global_completion.get("retained_transformer_token_state_bytes") == 0
        and compilation.get("local_result_receipt_sha256")
        == question.get("v6_result_receipt_sha256")
        and compilation.get("global_result_receipt_sha256")
        == global_completion.get("receipt_sha256")
        and compilation.get("query_receipt_sha256")
        == global_completion.get("query_receipt_sha256")
        and compilation.get("residual_index_receipt_sha256")
        == global_completion.get("residual_index_receipt_sha256")
        and compilation.get("exact_span_support_population_receipt_sha256")
        == exact_span_support_population.receipt_sha256
        and exact_span_support_population.plane_selection_receipt_sha256s
        == tuple(plane_selection_receipts)
        and exact_span_support_population.plane_candidate_receipt_sha256s
        == tuple(plane_candidate_receipts)
        and identity_sha256(
            {
                "format": f"{TERMINAL_COMPILATION_FORMAT}-local-audit-v1",
                "exact_span_support_population": (
                    exact_span_support_population.projection()
                ),
                "local_rows": _exact_list(
                    local_audit.get("local_rows"), "terminal local rows"
                ),
                "mechanism_by_handle": _exact_dict(
                    local_audit.get("mechanism_by_handle"),
                    "terminal local mechanisms",
                ),
            }
        )
        == compilation.get("local_audit_receipt_sha256")
        and terminal_prompt.get("provider_input") == provider_input
        and terminal_prompt.get("messages_sha256") == plan.get("messages_sha256")
        and terminal_prompt.get("prompt_token_proxy") == prompt_tokens
        and terminal_prompt.get("allowed_handle_ids")
        == plan.get("allowed_handle_ids")
        and local_prompt.get("provider_input") == provider_input
        and local_prompt.get("messages_sha256") == plan.get("messages_sha256")
        and local_prompt.get("prompt_token_proxy") == prompt_tokens
        and local_prompt.get("allowed_handle_ids") == plan.get("allowed_handle_ids")
        and local_prompt.get("handle_group_by_id") == plan.get("handle_group_by_id")
        and terminal_prompt.get("story_coherence") == plan.get("story_coherence")
        and local_prompt.get("story_coherence") == plan.get("story_coherence")
        and terminal_prompt.get("preservation_requirements")
        == plan.get("preservation_requirements")
        and local_prompt.get("preservation_requirements")
        == plan.get("preservation_requirements")
        and terminal_prompt.get("validation_contract")
        == plan.get("validation_contract")
        and local_prompt.get("validation_contract")
        == plan.get("validation_contract")
        and compilation.get("sealed_sources")
        == plan.get("source_artifact_bindings"),
        "terminal answer plan failed strict self-authentication",
    )
    return plan


def load_verified_terminal_assay(
    output_root: str | Path,
    expected_construct_sha256: str,
    expected_replay_sha256: str,
    *,
    v7_source_root: str | Path = DEFAULT_V7_SOURCE_ROOT,
) -> tuple[SealedArtifact, SealedArtifact, tuple[dict[str, Any], ...]]:
    root = Path(output_root)
    construction = read_sealed_json(root / CONSTRUCTION_NAME)
    replay = read_sealed_json(root / REPLAY_NAME)
    frozen_v7_root = Path(v7_source_root)
    frozen_v7_construction = read_sealed_json(
        frozen_v7_root / v7_cli.CONSTRUCTION_NAME
    )
    frozen_v7_replay = read_sealed_json(frozen_v7_root / v7_cli.REPLAY_NAME)
    _require(
        construction.sha256
        == require_sha256(expected_construct_sha256, "terminal construction")
        and replay.sha256
        == require_sha256(expected_replay_sha256, "terminal replay")
        and construction.sha256 == replay.sha256
        and construction.payload == replay.payload
        and construction.payload.get("format") == FORMAT,
        "terminal construction/replay artifact binding changed",
    )
    payload = construction.payload
    _require(
        frozen_v7_construction.sha256 == frozen_v7_replay.sha256
        and frozen_v7_construction.payload == frozen_v7_replay.payload
        and frozen_v7_construction.payload.get("format") == v7_cli.FORMAT
        and payload.get("base_v7_assay_sha256")
        == frozen_v7_construction.sha256
        and payload.get("base_v7_replay_sha256") == frozen_v7_replay.sha256
        and _project_base_v7(payload) == frozen_v7_construction.payload,
        "terminal assay differs from its independently sealed V7 ancestor",
    )
    identity = require_sha256(
        str(payload.get("construction_identity_sha256")),
        "terminal construction identity",
    )
    body = {
        key: value
        for key, value in payload.items()
        if key != "construction_identity_sha256"
    }
    questions = _exact_list(payload.get("questions"), "terminal assay questions")
    terminal_policy = _exact_dict(payload.get("terminal_policy"), "terminal policy")
    terminal_policy_receipt = require_sha256(
        str(terminal_policy.get("receipt_sha256")), "terminal policy"
    )
    terminal_policy_body = {
        key: value
        for key, value in terminal_policy.items()
        if key != "receipt_sha256"
    }
    terminal_sources = _exact_dict(
        payload.get("terminal_source_artifact_bindings"),
        "terminal source artifact bindings",
    )
    terminal_sources_receipt = require_sha256(
        str(terminal_sources.get("receipt_sha256")), "terminal source bindings"
    )
    terminal_sources_body = {
        key: value
        for key, value in terminal_sources.items()
        if key != "receipt_sha256"
    }
    r7_bindings = _exact_dict(payload.get("r7_bindings"), "terminal R7 bindings")
    r7_construction_sha256 = require_sha256(
        str(r7_bindings.get("construction_artifact_sha256")),
        "terminal R7 construction",
    )
    r7_gate_sha256 = require_sha256(
        str(r7_bindings.get("gate_artifact_sha256")), "terminal R7 gate"
    )
    query_vector_sha256 = require_sha256(
        str(r7_bindings.get("query_vector_artifact_sha256")),
        "terminal R7 query vectors",
    )
    query_vector_replay_sha256 = require_sha256(
        str(r7_bindings.get("query_vector_replay_artifact_sha256")),
        "terminal R7 query-vector replay",
    )
    _require(
        identity == identity_sha256(body)
        and payload.get("base_v7_format") == v7_cli.FORMAT
        and payload.get("exact_terminal_ordinals") == list(EXACT_ORDINALS)
        and payload.get("new_provider_calls") == 0
        and payload.get("retained_transformer_token_state_bytes") == 0
        and payload.get("question_count") == len(EXACT_ORDINALS)
        and payload.get("terminal_answer_plan_count") == len(EXACT_ORDINALS)
        and terminal_policy_receipt == identity_sha256(terminal_policy_body)
        and terminal_policy == SemanticGlobalTerminalPolicy().projection()
        and terminal_sources_receipt == identity_sha256(terminal_sources_body)
        and terminal_sources.get("protected_owner_artifact_sha256")
        == r7_construction_sha256
        and terminal_sources.get("residual_artifact_sha256")
        == r7_construction_sha256
        and terminal_sources.get("parent_artifact_sha256")
        == r7_gate_sha256
        and query_vector_sha256 == query_vector_replay_sha256
        and tuple(
            _exact_dict(row, "terminal assay question").get("ordinal")
            for row in questions
        )
        == EXACT_ORDINALS,
        "terminal construction identity/population changed",
    )
    plans_list: list[dict[str, Any]] = []
    question_receipt_by_ordinal: dict[int, str] = {}
    question_ordinals_by_namespace: dict[str, list[int]] = {}
    for raw in questions:
        question = _exact_dict(raw, "terminal assay question")
        question_receipt = require_sha256(
            str(question.get("question_assay_receipt_sha256")),
            "terminal question assay",
        )
        question_body = {
            key: value
            for key, value in question.items()
            if key != "question_assay_receipt_sha256"
        }
        _require(
            question_receipt == identity_sha256(question_body),
            "terminal question assay receipt changed",
        )
        ordinal = _exact_int(question.get("ordinal"), "terminal question ordinal")
        namespace_id = require_text(
            question.get("namespace_id"), "terminal question namespace"
        )
        question_receipt_by_ordinal[ordinal] = question_receipt
        question_ordinals_by_namespace.setdefault(namespace_id, []).append(ordinal)
        plan = _validate_answer_plan(question.get("terminal_answer_plan"), question)
        compilation = _exact_dict(
            plan.get("terminal_compilation"), "terminal plan compilation"
        )
        _require(
            plan.get("source_artifact_bindings") == terminal_sources
            and compilation.get("sealed_sources") == terminal_sources
            and compilation.get("policy") == terminal_policy,
            "terminal plan escaped top-level policy/source bindings",
        )
        plans_list.append(plan)
    namespace_rows = _exact_list(
        payload.get("namespace_receipts"), "terminal namespace receipts"
    )
    _require(
        len(namespace_rows) == len(question_ordinals_by_namespace),
        "terminal namespace population changed",
    )
    seen_namespaces: set[str] = set()
    for raw in namespace_rows:
        namespace = _exact_dict(raw, "terminal namespace receipt")
        namespace_id = require_text(
            namespace.get("namespace_id"), "terminal namespace ID"
        )
        namespace_receipt = require_sha256(
            str(namespace.get("namespace_assay_receipt_sha256")),
            "terminal namespace receipt",
        )
        namespace_body = {
            key: value
            for key, value in namespace.items()
            if key != "namespace_assay_receipt_sha256"
        }
        _require(
            namespace_id not in seen_namespaces
            and namespace_receipt == identity_sha256(namespace_body)
            and namespace.get("question_assay_receipt_sha256s")
            == [
                question_receipt_by_ordinal[ordinal]
                for ordinal in question_ordinals_by_namespace.get(namespace_id, [])
            ],
            "terminal namespace receipt/question binding changed",
        )
        seen_namespaces.add(namespace_id)
    _require(
        seen_namespaces == set(question_ordinals_by_namespace),
        "terminal namespace set differs from question population",
    )
    plans = tuple(plans_list)
    assert_gold_blind(payload, path="verified_semantic_global_terminal_assay")
    return construction, replay, plans


def run_construct(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_assay(args)
    artifact, created = publish_sealed_json(
        Path(args.output_root) / CONSTRUCTION_NAME, payload
    )
    return {
        "assay_sha256": artifact.sha256,
        "created": created,
        "new_provider_calls": 0,
        "question_count": payload["question_count"],
        "retained_transformer_token_state_bytes": 0,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    rebuilt = build_assay(args)
    artifact = read_sealed_json(Path(args.output_root) / CONSTRUCTION_NAME)
    replay_difference = _first_projection_difference(rebuilt, artifact.payload)
    _require(
        artifact.sha256
        == require_sha256(str(args.expected_assay_sha256), "terminal assay")
        and replay_difference is None,
        "terminal assay differs from exact resident-index replay; "
        f"first_difference={json.dumps(replay_difference, sort_keys=True)}",
    )
    replay, _created = publish_sealed_json(
        Path(args.output_root) / REPLAY_NAME, rebuilt
    )
    _require(replay.sha256 == artifact.sha256, "terminal assay replay changed bytes")
    return {
        "assay_sha256": artifact.sha256,
        "byte_identical": True,
        "new_provider_calls": 0,
        "replay_sha256": replay.sha256,
        "retained_transformer_token_state_bytes": 0,
    }


def _add_args(parser: argparse.ArgumentParser) -> None:
    v7_cli._add_args(parser)  # noqa: SLF001
    parser.set_defaults(
        auto_resolve_episode_artifact=True,
        output_root=DEFAULT_OUTPUT_ROOT,
        ordinals=EXACT_ORDINALS,
    )
    parser.add_argument(
        "--v7-construction",
        type=Path,
        default=DEFAULT_V7_SOURCE_ROOT / v7_cli.CONSTRUCTION_NAME,
    )
    parser.add_argument(
        "--expected-v7-assay-sha256", default=DEFAULT_V7_ASSAY_SHA256
    )
    parser.add_argument(
        "--v7-replay",
        type=Path,
        default=DEFAULT_V7_SOURCE_ROOT / v7_cli.REPLAY_NAME,
    )
    parser.add_argument(
        "--expected-v7-replay-sha256", default=DEFAULT_V7_ASSAY_SHA256
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    construct = commands.add_parser("construct")
    _add_args(construct)
    replay = commands.add_parser("replay")
    _add_args(replay)
    replay.add_argument("--expected-assay-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_construct(args) if args.command == "construct" else run_replay(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANSWER_PLAN_FORMAT",
    "CONSTRUCTION_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_V7_ASSAY_SHA256",
    "DEFAULT_V7_SOURCE_ROOT",
    "EXACT_ORDINALS",
    "FORMAT",
    "REPLAY_NAME",
    "ROUTE_ID",
    "ReducedSemanticGlobalTerminalAssayError",
    "build_assay",
    "build_parser",
    "load_verified_terminal_assay",
    "main",
    "run_construct",
    "run_replay",
]
