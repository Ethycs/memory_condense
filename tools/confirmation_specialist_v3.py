#!/usr/bin/env python3
"""Arbitrary-population confirmation specialist V2 and deterministic V3.

This module ports the promoted specialist path without importing any
validation coordinates.  A question-local route selects numeric, preference,
and temporal specialists against the immutable namespace store.  Provider
work is then sealed through the common native Terra journal lifecycle.  The
answer plane is parsed against advisory-local proofs, with the historical
ordinary typed renderer used only for recognized legacy proof shapes.  A
provider-free V3 finally composes temporal, numeric, cross-plane authority,
and V2 fallback in the frozen order.

No function in this module opens labels.  Only ``run_*_provider`` can create a
client, all retry counts are zero, and every API is population-size neutral.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from memory_condense.domain._tokenizer import count_chat_prompt_token_proxy
from memory_condense.domain.discourse import quote_sha256
from memory_condense.persistence.db import Database
from tools import confirmation_terra_completion_lifecycle as terra
from tools.confirmation_contracts import publish_sealed_json
from tools.matched_eval.artifacts import SealedArtifact as MatchedArtifact
from tools.matched_eval import confirmation_specialist_core as specialist
from tools.matched_eval import (
    confirmation_specialist_reconciliation as reconcile_v3,
)
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
    require_text,
)
from tools.matched_eval.full_store_slot_closure import build_full_store_window_index
from tools.matched_eval.query_guided_scan import cache_namespace_partitions
from tools.matched_eval.specialist_scoped_completion import (
    PROMPT_FORMAT as SPECIALIST_PROMPT_FORMAT,
    SpecialistScopedCompletionError,
    SpecialistValidationScope,
    compile_specialist_validation_scope,
    parse_specialist_scoped_completion,
    render_specialist_scoped_prompt,
)
from tools.matched_eval.typed_memory_final_arm import (
    VALIDATOR_POLICY_FORMAT,
    parse_typed_final_completion,
    render_final_messages,
)
from tools.matched_eval.prediction_row_projection import (
    prediction_row_projection,
)
from tools.matched_eval.typed_operator_spec import (
    TemporalMode,
    compile_typed_operator_spec,
)
from tools.confirmation_canonical import canonical_sha256


FORMAT = "memory-condense-confirmation-specialist-v3"
CONSTRUCTION_FORMAT = f"{FORMAT}-construction-v2"
CONSTRUCTION_REPLAY_FORMAT = f"{FORMAT}-construction-replay-v1"
ANSWER_PLAN_FORMAT = f"{FORMAT}-answer-plan-v2"
PROMPT_STAGE_ID = "confirmation-specialist-v2-terra-answer"
V2_RUN_FORMAT = f"{FORMAT}-answer-run-v2"
V3_RUN_FORMAT = f"{FORMAT}-reconciliation-run-v3"
V3_STATUS_FORMAT = f"{FORMAT}-final-reconciliation-v3-status-v1"
V3_POLICY_FORMAT = f"{FORMAT}-reconciliation-policy-v3"
ORDINARY_TYPED_TRANSFORM_FORMAT = f"{FORMAT}-ordinary-typed-transform-v1"

CONSTRUCTION_NAME = "confirmation-specialist-construction-v2.json"
CONSTRUCTION_REPLAY_NAME = "confirmation-specialist-construction-replay-v2.json"
PROMPT_NAME = "confirmation-specialist-prompt-v2.json"
V2_RUN_NAME = "confirmation-specialist-answer-v2.json"
V2_REPLAY_NAME = "confirmation-specialist-answer-replay-v2.json"
V3_RUN_NAME = "confirmation-specialist-reconciliation-v3.json"
V3_REPLAY_NAME = "confirmation-specialist-reconciliation-replay-v3.json"
TERMINAL_PARENT_NAME = "confirmation-terminal-parent-population-v1.json"

SCOPED_PARSER = "specialist_scoped_v2"
ORDINARY_TYPED_PARSER = "ordinary_typed_final_v2"
PASSTHROUGH_PARSER = "parent_passthrough"

HARD_COMPLETE_CHAT_TOKEN_CAP = 8_000
OUTPUT_TOKEN_RESERVE = 768
MAX_CHAT_PROMPT_TOKENS = HARD_COMPLETE_CHAT_TOKEN_CAP - OUTPUT_TOKEN_RESERVE

# These are the exact, semantic failures observed in the promoted V2 adapter.
# They describe unsupported proof topology, not questions, IDs, or positions.
_RECOGNIZED_LEGACY_PROOF_ERRORS = frozenset(
    {
        "numeric group candidates escaped or overlap",
        "numeric operation mode changed",
        "specialist candidate handle map is empty",
    }
)


class ConfirmationSpecialistV3Error(MatchedEvalContractError):
    """A specialist source, prompt, checkpoint, proof, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationSpecialistV3Error(message)


def _exact_dict(value: object, label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an exact object")
    return value  # type: ignore[return-value]


def _exact_list(value: object, label: str) -> list[Any]:
    _require(type(value) is list, f"{label} must be an exact array")
    return value  # type: ignore[return-value]


def _self_hashed(
    row: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> dict[str, Any]:
    body = dict(row)
    declared = require_sha256(body.pop(key, None), label)
    _require(identity_sha256(body) == declared, f"{label} receipt changed")
    return dict(row)


def _sealed(body: Mapping[str, Any], key: str = "receipt_sha256") -> dict[str, Any]:
    return {**dict(body), key: identity_sha256(body)}


def _artifact_payload(value: object, label: str) -> dict[str, Any]:
    payload = getattr(value, "payload", None)
    return _exact_dict(payload, label)


def _artifact_sha(value: object, label: str) -> str:
    return require_sha256(getattr(value, "sha256", None), label)


def _question_text(dated_question: str) -> str:
    first, separator, rest = dated_question.partition("\n")
    return rest if separator and first.startswith("[Question asked at ") else dated_question


@dataclass(frozen=True, slots=True)
class OrdinaryTypedScope:
    allowed_handle_ids: tuple[str, ...]
    handle_group_by_id: Mapping[str, str]
    story_coherence: Mapping[str, Any]
    preservation_requirements: Mapping[str, Any]
    validation_contract: Mapping[str, Any]
    transform_receipt_sha256: str
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class ConfirmationSpecialistAnswerPlan:
    ordinal: int
    question_id: str
    parser_kind: str
    parent_prediction: str
    provider_input: Mapping[str, Any] | None
    messages: tuple[dict[str, str], ...]
    scope: SpecialistValidationScope | OrdinaryTypedScope | None
    projection: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require(
            self.parser_kind in {
                SCOPED_PARSER,
                ORDINARY_TYPED_PARSER,
                PASSTHROUGH_PARSER,
            },
            "specialist answer plan parser changed",
        )
        _require(
            (self.parser_kind == PASSTHROUGH_PARSER)
            == (self.provider_input is None and not self.messages and self.scope is None),
            "specialist passthrough/provider disposition changed",
        )
        _self_hashed(
            self.projection,
            "answer_plan_receipt_sha256",
            label="specialist answer plan",
        )

    @property
    def receipt_sha256(self) -> str:
        return str(self.projection["answer_plan_receipt_sha256"])


@dataclass(frozen=True, slots=True)
class ConfirmationSpecialistConstruction:
    artifact: terra.SealedArtifact
    questions: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ConfirmationSpecialistPreflight:
    construction: ConfirmationSpecialistConstruction
    prompt_artifact: terra.SealedArtifact
    lifecycle_preflight_artifact: terra.SealedArtifact
    plans: tuple[ConfirmationSpecialistAnswerPlan, ...]

    @property
    def submitted_plans(self) -> tuple[ConfirmationSpecialistAnswerPlan, ...]:
        return tuple(row for row in self.plans if row.parser_kind != PASSTHROUGH_PARSER)

    @property
    def required_provider_calls(self) -> int:
        return int(
            self.lifecycle_preflight_artifact.payload["population"][
                "unique_prompt_count"
            ]
        )


@dataclass(frozen=True, slots=True)
class ConfirmationSpecialistV2Materialization:
    preflight: ConfirmationSpecialistPreflight
    release_artifact: terra.SealedArtifact
    completion_artifact: terra.SealedArtifact
    run_artifact: terra.SealedArtifact
    completion_batch: Any
    predictions: tuple[str, ...]
    result_rows: tuple[dict[str, Any], ...]
    judge_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class VerifiedConfirmationSpecialistV2Plane:
    construction_artifact: terra.SealedArtifact
    prompt_artifact: terra.SealedArtifact
    lifecycle_preflight_artifact: terra.SealedArtifact
    release_artifact: terra.SealedArtifact
    completion_artifact: terra.SealedArtifact
    run_artifact: terra.SealedArtifact
    replay_artifact: terra.SealedArtifact
    completion_batch: Any
    plans: tuple[ConfirmationSpecialistAnswerPlan, ...]
    predictions: tuple[str, ...]
    result_rows: tuple[dict[str, Any], ...]
    judge_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ConfirmationSpecialistV3Audit:
    source_bundle: Any
    lane_audits: Any

    @property
    def status_population_sha256s(self) -> dict[str, str]:
        return {
            "temporal": self.lane_audits.temporal.status_population_sha256,
            "numeric": self.lane_audits.numeric.status_population_sha256,
            "authority": self.lane_audits.authority.status_population_sha256,
        }


@dataclass(frozen=True, slots=True)
class VerifiedConfirmationSpecialistV3Plane:
    v2_plane: VerifiedConfirmationSpecialistV2Plane
    run_artifact: terra.SealedArtifact
    replay_artifact: terra.SealedArtifact
    predictions: tuple[str, ...]
    result_rows: tuple[dict[str, Any], ...]
    judge_rows: tuple[dict[str, Any], ...]
    status_rows: tuple[dict[str, Any], ...]
    lane_status_population_sha256s: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ConfirmationTerminalParentSource:
    """Question-local V3 ancestry needed by the terminal policy boundary."""

    question_id: str
    namespace_id: str
    namespace_receipt_sha256: str
    question: str
    dated_question: str
    source_row_receipt_sha256: str
    answer_row: Mapping[str, Any]
    construction_row: Mapping[str, Any]
    prior_answer_row: Mapping[str, Any] | None
    reconciliation_row: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        require_text(self.question_id, "terminal source question ID")
        require_text(self.namespace_id, "terminal source namespace ID")
        require_text(self.question, "terminal source question")
        require_text(self.dated_question, "terminal source dated question")
        require_sha256(
            self.namespace_receipt_sha256, "terminal source namespace receipt"
        )
        require_sha256(self.source_row_receipt_sha256, "terminal source row receipt")
        _require(
            isinstance(self.answer_row, Mapping)
            and isinstance(self.construction_row, Mapping)
            and (
                self.prior_answer_row is None
                or isinstance(self.prior_answer_row, Mapping)
            )
            and (
                self.reconciliation_row is None
                or isinstance(self.reconciliation_row, Mapping)
            ),
            "terminal source eligibility ancestry changed type",
        )


def question_local_specialist_route(dated_question: str) -> tuple[str, ...]:
    """Expose the production question-only route without population metadata."""

    require_text(dated_question, "specialist dated question")
    result = specialist.applicable_specialist_ids(dated_question)
    _require(len(result) == len(set(result)), "specialist route repeats a mechanism")
    return result


def _is_recognized_legacy_proof_error(exc: BaseException) -> bool:
    return type(exc) is SpecialistScopedCompletionError and str(exc) in (
        _RECOGNIZED_LEGACY_PROOF_ERRORS
    )


def _require_typed_plane(value: object) -> Any:
    """Import the upstream carrier lazily so this stage cannot fork its work."""

    from tools.confirmation_typed_final import (  # noqa: PLC0415
        VerifiedConfirmationTypedFinalPlane,
    )

    _require(
        type(value) is VerifiedConfirmationTypedFinalPlane,
        "specialist construction requires the exact verified typed-final plane",
    )
    rows = tuple(value.result_rows)
    judges = tuple(value.judge_rows)
    _require(
        bool(rows)
        and len(rows) == len(judges) == len(value.predictions)
        and tuple(row.get("ordinal") for row in rows) == tuple(range(len(rows))),
        "typed-final parent population changed",
    )
    for ordinal, (row, judge, prediction) in enumerate(
        zip(rows, judges, value.predictions, strict=True)
    ):
        _require(
            type(row) is dict
            and type(judge) is dict
            and row.get("prediction") == prediction
            and row.get("prediction_sha256") == quote_sha256(prediction)
            and judge == prediction_row_projection(row),
            f"typed-final answer/judge seam changed at ordinal {ordinal}",
        )
    assert_gold_blind(
        _artifact_payload(value.run_artifact, "typed-final run"),
        path="confirmation_specialist_typed_parent",
    )
    return value


def _parent_source(
    typed_plane: Any,
    *,
    ordinal: int,
    source_row: Mapping[str, Any],
    judge_row: Mapping[str, Any],
) -> dict[str, Any]:
    prediction = require_text(source_row.get("prediction"), "typed parent prediction")
    prediction_sha = require_sha256(
        source_row.get("prediction_sha256"), "typed parent prediction"
    )
    source_receipt = require_sha256(
        source_row.get("source_row_sha256"), "typed parent source row"
    )
    _require(
        source_row.get("ordinal") == judge_row.get("ordinal") == ordinal
        and prediction_sha == quote_sha256(prediction)
        and judge_row == prediction_row_projection(source_row),
        f"typed parent row changed at ordinal {ordinal}",
    )
    body = {
        "parent_judge_row": dict(judge_row),
        "parent_judge_row_sha256": identity_sha256(judge_row),
        "prediction": prediction,
        "prediction_sha256": prediction_sha,
        "replay_artifact_sha256": _artifact_sha(
            typed_plane.replay_artifact, "typed-final replay"
        ),
        "run_artifact_sha256": _artifact_sha(
            typed_plane.run_artifact, "typed-final run"
        ),
        "source_row_sha256": source_receipt,
    }
    return _sealed(body)


def _passthrough_question(
    *,
    ordinal: int,
    namespace_id: str,
    composition_row: Mapping[str, Any],
    parent_source: Mapping[str, Any],
    dated_question: str,
) -> dict[str, Any]:
    body = {
        "applicable_specialist_ids": list(
            question_local_specialist_route(dated_question)
        ),
        "dated_question_sha256": require_sha256(
            composition_row.get("dated_question_sha256"), "dated question"
        ),
        "methods": [],
        "mode": "parent_passthrough",
        "namespace_id": require_sha256(namespace_id, "specialist namespace"),
        "new_provider_calls": 0,
        "ordinal": ordinal,
        "parent_source": dict(parent_source),
        "question_id": require_text(
            composition_row.get("question_id"), "specialist question ID"
        ),
        "question_sha256": require_sha256(
            composition_row.get("question_sha256"), "specialist question"
        ),
        "retained_transformer_token_state_bytes": 0,
        "route": specialist.route_question(dated_question).identity_payload(),
        "terminal_prompt": None,
    }
    assert_gold_blind(body, path="confirmation_specialist_passthrough")
    return _sealed(body, "question_receipt_sha256")


def _specialist_question(
    *,
    ordinal: int,
    namespace_id: str,
    index: Any,
    composition_row: Mapping[str, Any],
    composition_sha256: str,
    parent_source: Mapping[str, Any],
) -> dict[str, Any]:
    dated_question, _old_prediction, question_id = specialist._question_inputs(  # noqa: SLF001
        composition_row
    )
    routes = question_local_specialist_route(dated_question)
    if not routes:
        return _passthrough_question(
            ordinal=ordinal,
            namespace_id=namespace_id,
            composition_row=composition_row,
            parent_source=parent_source,
            dated_question=dated_question,
        )
    generic = specialist._composed_question(  # noqa: SLF001
        ordinal=ordinal,
        index=index,
        composition_row=composition_row,
        parent_composition_artifact_sha256=composition_sha256,
        frozen_row={
            "namespace_id": namespace_id,
            "ordinal": ordinal,
            "question_id": question_id,
        },
        parent_prediction_override=require_text(
            parent_source.get("prediction"), "specialist protected parent"
        ),
        terminal_message_renderer_format=SPECIALIST_PROMPT_FORMAT,
        terminal_prompt_envelope_renderer=render_specialist_scoped_prompt,
    )
    terminal = _exact_dict(generic.get("terminal_prompt"), "specialist terminal")
    provider = _exact_dict(terminal.get("provider_input"), "specialist provider input")
    advisories = _exact_list(
        provider.get("specialist_advisories"), "specialist advisories"
    )
    if not advisories:
        return _passthrough_question(
            ordinal=ordinal,
            namespace_id=namespace_id,
            composition_row=composition_row,
            parent_source=parent_source,
            dated_question=dated_question,
        )
    spec = compile_typed_operator_spec(dated_question)
    body = dict(generic)
    body.pop("question_receipt_sha256", None)
    body.update(
        {
            "mode": "specialist",
            "parent_source": dict(parent_source),
            "successor_policy": _sealed(
                {
                    "applicable_specialist_ids": list(routes),
                    "latest_state_routed_from_typed_spec": (
                        spec.temporal_mode is TemporalMode.LATEST_STATE
                    ),
                    "question_sha256": spec.question_sha256,
                    "route_basis": "dated_question_and_receipt_bound_operator_state",
                }
            ),
        }
    )
    _require(
        terminal.get("full_chat_plus_output_tokens", HARD_COMPLETE_CHAT_TOKEN_CAP + 1)
        <= HARD_COMPLETE_CHAT_TOKEN_CAP,
        "specialist prompt escaped the hard complete-chat cap",
    )
    assert_gold_blind(body, path="confirmation_specialist_question")
    return _sealed(body, "question_receipt_sha256")


def _cache_receipts_by_namespace(closure_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = _exact_list(closure_payload.get("cache_receipts"), "typed closure cache receipts")
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = _exact_dict(raw, "typed closure cache receipt")
        namespace_id = require_sha256(row.get("namespace_id"), "closure namespace")
        _require(namespace_id not in result, "closure namespace repeats")
        result[namespace_id] = row
    return result


def _validate_cache_receipt(sealed: Mapping[str, Any], cache: Any, index: Any) -> None:
    for key, observed in (
        ("cache_receipt_sha256", cache.cache_receipt_sha256),
        ("window_index_receipt_sha256", index.receipt_sha256),
        ("content_row_count", cache.content_row_count),
        ("physical_store_row_count", cache.physical_store_row_count),
    ):
        _require(sealed.get(key) == observed, f"specialist closure {key} changed")


def build_confirmation_specialist_construction_payload(
    typed_plane: object,
    context: object,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Build the provider-free specialist construction over arbitrary N."""

    parent = _require_typed_plane(typed_plane)
    from tools.confirmation_query_expansion_adapter import (  # noqa: PLC0415
        ConfirmationQueryExpansionContext,
    )

    _require(
        type(context) is ConfirmationQueryExpansionContext,
        "specialist construction requires the exact confirmation query context",
    )
    context.revalidate_store_bytes()
    composition_artifact = parent.composition_artifact
    closure_artifact = parent.closure_input_artifact
    composition_payload = _artifact_payload(composition_artifact, "typed composition")
    closure_payload = _artifact_payload(closure_artifact, "typed closure")
    composition_rows = tuple(
        _exact_dict(row, "typed composition row")
        for row in _exact_list(composition_payload.get("questions"), "typed composition questions")
    )
    count = len(parent.result_rows)
    _require(
        count > 0
        and len(composition_rows) == context.question_count == count,
        "specialist parent populations differ",
    )
    population_by_question = {
        row.source.packet.question_id: row for row in context.population.rows
    }
    _require(len(population_by_question) == count, "specialist query population repeats")
    grouped: dict[str, list[int]] = defaultdict(list)
    parent_sources: list[dict[str, Any]] = []
    dated_questions: list[str] = []
    for ordinal, (composition, source, judge) in enumerate(
        zip(composition_rows, parent.result_rows, parent.judge_rows, strict=True)
    ):
        dated_question, _old_prediction, question_id = specialist._question_inputs(  # noqa: SLF001
            composition
        )
        population_row = population_by_question.get(question_id)
        _require(population_row is not None, "specialist question left its store namespace")
        packet = population_row.source.packet
        _require(
            composition.get("ordinal") == source.get("ordinal") == ordinal
            and composition.get("question_id") == source.get("question_id") == question_id
            and composition.get("question_sha256") == source.get("question_sha256") == packet.question_sha256
            and composition.get("dated_question_sha256")
            == source.get("dated_question_sha256")
            == packet.dated_question_sha256
            == quote_sha256(dated_question),
            f"specialist parent question binding changed at ordinal {ordinal}",
        )
        parent_sources.append(
            _parent_source(parent, ordinal=ordinal, source_row=source, judge_row=judge)
        )
        dated_questions.append(dated_question)
        if question_local_specialist_route(dated_question):
            grouped[population_row.namespace.namespace_id].append(ordinal)

    sealed_cache = _cache_receipts_by_namespace(closure_payload)
    questions: list[dict[str, Any] | None] = [None] * count
    lifecycle: list[dict[str, Any]] = []
    composition_sha = _artifact_sha(composition_artifact, "typed composition")
    for ordinal, dated_question in enumerate(dated_questions):
        if not question_local_specialist_route(dated_question):
            population_row = population_by_question[composition_rows[ordinal]["question_id"]]
            questions[ordinal] = _passthrough_question(
                ordinal=ordinal,
                namespace_id=population_row.namespace.namespace_id,
                composition_row=composition_rows[ordinal],
                parent_source=parent_sources[ordinal],
                dated_question=dated_question,
            )

    namespaces = {
        row.namespace.namespace_id: row.namespace for row in context.population.rows
    }
    for namespace_id in sorted(grouped):
        namespace = namespaces[namespace_id]
        database_path = context.store_dirs_by_namespace[namespace_id] / "memory.db"
        with Database(database_path, read_only=True) as database:
            cache = cache_namespace_partitions(
                database,
                namespace,
                source_database_sha256=context.database_sha256_by_namespace[namespace_id],
                source_store_receipt_sha256=namespace.combined_store_receipt_sha256,
            )
        index = build_full_store_window_index(cache)
        sealed_cache_row = sealed_cache.get(namespace_id)
        _require(sealed_cache_row is not None, "specialist closure namespace is missing")
        _validate_cache_receipt(sealed_cache_row, cache, index)
        for ordinal in grouped[namespace_id]:
            questions[ordinal] = _specialist_question(
                ordinal=ordinal,
                namespace_id=namespace_id,
                index=index,
                composition_row=composition_rows[ordinal],
                composition_sha256=composition_sha,
                parent_source=parent_sources[ordinal],
            )
        lifecycle.append(
            {
                "cache_receipt_sha256": cache.cache_receipt_sha256,
                "content_row_count": cache.content_row_count,
                "database_read_passes": 1,
                "namespace_id": namespace_id,
                "physical_content_token_count": index.physical_content_tokens_indexed,
                "physical_store_row_count": cache.physical_store_row_count,
                "window_index_receipt_sha256": index.receipt_sha256,
            }
        )

    _require(all(type(row) is dict for row in questions), "specialist construction lost a row")
    typed_questions = tuple(row for row in questions if type(row) is dict)
    modes = Counter(row["mode"] for row in typed_questions)
    terminal_tokens = [
        int(row["terminal_prompt"]["full_chat_plus_output_tokens"])
        for row in typed_questions
        if row["mode"] == "specialist"
    ]
    body: dict[str, Any] = {
        "bindings": {
            "parent_composition_artifact_sha256": composition_sha,
            "parent_full_store_input_artifact_sha256": _artifact_sha(
                closure_artifact, "typed closure"
            ),
            "parent_replay_artifact_sha256": _artifact_sha(
                parent.replay_artifact, "typed-final replay"
            ),
            "parent_run_artifact_sha256": _artifact_sha(
                parent.run_artifact, "typed-final run"
            ),
        },
        "construction_is_posthoc_outcome_conditioned": False,
        "format": CONSTRUCTION_FORMAT,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "max_terminal_complete_envelope_tokens": max(terminal_tokens, default=0),
        "new_provider_calls": 0,
        "ordinals": list(range(count)),
        "parent_passthrough_count": modes["parent_passthrough"],
        "question_count": count,
        "questions": list(typed_questions),
        "resident_index_lifecycle": {
            "database_read_passes_per_used_namespace": 1,
            "maximum_simultaneous_namespace_indexes": 1,
            "receipts": lifecycle,
            "total_database_read_passes": len(lifecycle),
            "unique_namespace_count": len(lifecycle),
        },
        "retained_transformer_token_state_bytes": 0,
        "routing_basis": "question_text_and_receipt_bound_local_proof_only",
        "selection_and_routing_frozen_before_target_plan_load": True,
        "specialist_provider_prompt_count": modes["specialist"],
        "target_labels_loaded": False,
        "target_plan_loaded": False,
    }
    assert_gold_blind(body, path="confirmation_specialist_construction")
    return (
        {**body, "construction_identity_sha256": identity_sha256(body)},
        typed_questions,
    )


def publish_confirmation_specialist_construction(
    typed_plane: object,
    context: object,
    *,
    output_root: str | Path,
) -> ConfirmationSpecialistConstruction:
    payload, rows = build_confirmation_specialist_construction_payload(
        typed_plane, context
    )
    artifact, _created = terra.publish_sealed_artifact(
        Path(output_root) / CONSTRUCTION_NAME, payload
    )
    return ConfirmationSpecialistConstruction(artifact, rows)


def replay_confirmation_specialist_construction(
    typed_plane: object,
    context: object,
    *,
    output_root: str | Path,
    expected_construction_sha256: str,
) -> ConfirmationSpecialistConstruction:
    expected = require_sha256(
        expected_construction_sha256, "expected specialist construction"
    )
    source = terra.read_sealed_artifact(
        Path(output_root) / CONSTRUCTION_NAME,
        expected_sha256=expected,
        label="specialist construction",
    )
    payload, rows = build_confirmation_specialist_construction_payload(
        typed_plane, context
    )
    _require(source.payload == payload, "specialist construction replay changed")
    replay, _created = terra.publish_sealed_artifact(
        Path(output_root) / CONSTRUCTION_REPLAY_NAME, payload
    )
    _require(replay.sha256 == source.sha256, "specialist construction replay is not byte-identical")
    return ConfirmationSpecialistConstruction(source, rows)


def _parent_plan_fields(raw: Mapping[str, Any]) -> dict[str, Any]:
    parent = _exact_dict(raw.get("parent_source"), "specialist parent source")
    _self_hashed(parent, "receipt_sha256", label="specialist parent source")
    prediction = require_text(parent.get("prediction"), "specialist parent prediction")
    _require(
        parent.get("prediction_sha256") == quote_sha256(prediction),
        "specialist parent prediction changed",
    )
    route = _exact_dict(raw.get("route"), "specialist route")
    judge = _exact_dict(parent.get("parent_judge_row"), "specialist parent judge")
    legacy = route.get("legacy_route")
    route_id = (
        "temporal_timeline"
        if route.get("temporal_mode") == TemporalMode.LATEST_STATE.value
        else judge.get("route_id")
        or route.get("style")
        or (legacy.get("style") if type(legacy) is dict else None)
    )
    route_id = require_text(route_id, "specialist route style")
    return {
        "construction_question_receipt_sha256": require_sha256(
            raw.get("question_receipt_sha256"), "specialist construction row"
        ),
        "dated_question_sha256": require_sha256(
            raw.get("dated_question_sha256"), "specialist dated question"
        ),
        "ordinal": raw.get("ordinal"),
        "parent_judge_row_sha256": require_sha256(
            parent.get("parent_judge_row_sha256"), "specialist parent judge row"
        ),
        "parent_prediction": prediction,
        "parent_prediction_sha256": parent["prediction_sha256"],
        "parent_replay_artifact_sha256": require_sha256(
            parent.get("replay_artifact_sha256"), "specialist parent replay"
        ),
        "parent_run_artifact_sha256": require_sha256(
            parent.get("run_artifact_sha256"), "specialist parent run"
        ),
        "parent_source_receipt_sha256": require_sha256(
            parent.get("receipt_sha256"), "specialist parent source"
        ),
        "parent_source_row_sha256": require_sha256(
            parent.get("source_row_sha256"), "specialist parent source row"
        ),
        "question_id": require_text(raw.get("question_id"), "specialist question ID"),
        "question_sha256": require_sha256(
            raw.get("question_sha256"), "specialist question"
        ),
        "route_id": route_id,
    }


def _ordinary_typed_plan(
    *,
    raw: Mapping[str, Any],
    common: Mapping[str, Any],
    provider_input: Mapping[str, Any],
    allowed: tuple[str, ...],
    groups: Mapping[str, str],
    fitted: Mapping[str, Any],
    source_prompt: Any,
    failure: SpecialistScopedCompletionError,
) -> ConfirmationSpecialistAnswerPlan:
    _require(
        _is_recognized_legacy_proof_error(failure),
        "unrecognized specialist proof failure cannot enter the typed fallback",
    )
    messages = tuple(dict(row) for row in render_final_messages(provider_input))
    prompt_tokens = count_chat_prompt_token_proxy(messages)
    _require(
        prompt_tokens <= MAX_CHAT_PROMPT_TOKENS,
        "ordinary typed fallback escaped the hard prompt cap",
    )
    terminal = _exact_dict(raw.get("terminal_prompt"), "specialist terminal")
    transform_body = {
        "format": ORDINARY_TYPED_TRANSFORM_FORMAT,
        "legacy_proof_shape": str(failure),
        "provider_input_sha256": identity_sha256(provider_input),
        "source_message_renderer_format": SPECIALIST_PROMPT_FORMAT,
        "source_messages_sha256": identity_sha256(list(source_prompt.messages)),
        "source_prompt_envelope_receipt_sha256": source_prompt.receipt_sha256,
        "source_terminal_prompt_receipt_sha256": require_sha256(
            terminal.get("terminal_prompt_receipt_sha256"), "specialist terminal"
        ),
        "target_message_renderer_format": specialist.TYPED_ANSWER_FORMAT,
        "target_messages_sha256": identity_sha256(list(messages)),
        "target_prompt_token_proxy": prompt_tokens,
        "transform": "rerender_identical_provider_input_as_ordinary_typed_final",
    }
    transform = _sealed(transform_body)
    validation = _exact_dict(
        fitted.get("validation_contract"), "ordinary typed validation contract"
    )
    story = _exact_dict(fitted.get("story_coherence"), "ordinary typed story")
    preservation = _exact_dict(
        fitted.get("preservation_requirements"), "ordinary typed preservation"
    )
    scope_body = {
        "allowed_handle_ids": list(allowed),
        "format": f"{ORDINARY_TYPED_TRANSFORM_FORMAT}-scope-v1",
        "handle_group_by_id": dict(groups),
        "preservation_requirements_sha256": identity_sha256(preservation),
        "prompt_transform_receipt_sha256": transform["receipt_sha256"],
        "story_coherence_sha256": identity_sha256(story),
        "validation_contract_sha256": identity_sha256(validation),
    }
    scope_receipt = identity_sha256(scope_body)
    scope = OrdinaryTypedScope(
        allowed,
        dict(groups),
        story,
        preservation,
        validation,
        transform["receipt_sha256"],
        scope_receipt,
    )
    body = {
        **dict(common),
        "adapter_prompt_transform": transform,
        "allowed_handle_ids": list(allowed),
        "answer_parser_kind": ORDINARY_TYPED_PARSER,
        "format": ANSWER_PLAN_FORMAT,
        "handle_group_by_id": dict(groups),
        "messages": list(messages),
        "messages_sha256": identity_sha256(list(messages)),
        "mode": "specialist",
        "preservation_requirements": preservation,
        "prompt_token_proxy": prompt_tokens,
        "provider_input": dict(provider_input),
        "scope_projection": scope_body,
        "scope_receipt_sha256": scope_receipt,
        "story_coherence": story,
        "terminal_prompt_receipt_sha256": transform[
            "source_terminal_prompt_receipt_sha256"
        ],
        "validation_contract": validation,
    }
    projection = {
        **body,
        "answer_plan_receipt_sha256": identity_sha256(body),
    }
    assert_gold_blind(projection, path="confirmation_specialist_ordinary_typed_plan")
    return ConfirmationSpecialistAnswerPlan(
        int(common["ordinal"]),
        str(common["question_id"]),
        ORDINARY_TYPED_PARSER,
        str(common["parent_prediction"]),
        dict(provider_input),
        messages,
        scope,
        projection,
    )


def _answer_plan(raw: Mapping[str, Any], ordinal: int) -> ConfirmationSpecialistAnswerPlan:
    _self_hashed(raw, "question_receipt_sha256", label="specialist construction row")
    common = _parent_plan_fields(raw)
    _require(
        common["ordinal"] == ordinal,
        f"specialist construction order changed at ordinal {ordinal}",
    )
    if raw.get("mode") == "parent_passthrough":
        _require(
            raw.get("terminal_prompt") is None and raw.get("methods") == [],
            f"specialist passthrough exposed work at ordinal {ordinal}",
        )
        body = {
            **common,
            "adapter_prompt_transform": None,
            "answer_parser_kind": PASSTHROUGH_PARSER,
            "format": ANSWER_PLAN_FORMAT,
            "messages": [],
            "messages_sha256": None,
            "mode": "parent_passthrough",
            "prompt_token_proxy": None,
            "provider_input": None,
            "scope_projection": None,
            "scope_receipt_sha256": None,
        }
        projection = {**body, "answer_plan_receipt_sha256": identity_sha256(body)}
        return ConfirmationSpecialistAnswerPlan(
            ordinal,
            str(common["question_id"]),
            PASSTHROUGH_PARSER,
            str(common["parent_prediction"]),
            None,
            (),
            None,
            projection,
        )

    _require(raw.get("mode") == "specialist", "specialist construction mode changed")
    terminal = _exact_dict(raw.get("terminal_prompt"), "specialist terminal")
    fitted = _exact_dict(raw.get("fitted_typed_prompt"), "specialist fitted prompt")
    provider = _exact_dict(terminal.get("provider_input"), "specialist provider input")
    fitted_provider = _exact_dict(
        fitted.get("provider_input"), "specialist fitted provider input"
    )
    advisories = _exact_list(provider.get("specialist_advisories"), "specialist advisories")
    _require(
        bool(advisories)
        and provider == {**dict(fitted_provider), "specialist_advisories": advisories},
        "specialist terminal escaped its fitted provider input",
    )
    allowed_raw = _exact_list(fitted.get("allowed_handle_ids"), "specialist allowed handles")
    allowed = tuple(require_text(value, "specialist handle") for value in allowed_raw)
    groups = specialist.handle_groups(provider, allowed)
    source_prompt = render_specialist_scoped_prompt(provider)
    _require(
        terminal.get("message_renderer_format") == SPECIALIST_PROMPT_FORMAT
        and terminal.get("messages_sha256")
        == identity_sha256(list(source_prompt.messages))
        and terminal.get("specialist_prompt_envelope_receipt_sha256")
        == source_prompt.receipt_sha256
        and terminal.get("prompt_token_proxy") == source_prompt.prompt_token_proxy
        and source_prompt.prompt_token_proxy <= MAX_CHAT_PROMPT_TOKENS,
        "specialist scoped prompt changed",
    )
    validation = _exact_dict(fitted.get("validation_contract"), "specialist validation")
    try:
        scope = compile_specialist_validation_scope(
            specialist_advisories=advisories,
            declared_specialist_advisories_sha256=require_sha256(
                terminal.get("specialist_advisories_sha256"),
                "specialist advisories",
            ),
            sealed_source_receipt_sha256=require_sha256(
                terminal.get("terminal_prompt_receipt_sha256"),
                "specialist terminal",
            ),
            terminal_allowed_handle_ids=allowed,
            handle_group_by_id=groups,
            validation_contract=validation,
            prompt_envelope=source_prompt,
        )
    except SpecialistScopedCompletionError as exc:
        return _ordinary_typed_plan(
            raw=raw,
            common=common,
            provider_input=provider,
            allowed=allowed,
            groups=groups,
            fitted=fitted,
            source_prompt=source_prompt,
            failure=exc,
        )
    messages = tuple(dict(row) for row in source_prompt.messages)
    body = {
        **common,
        "adapter_prompt_transform": None,
        "allowed_handle_ids": list(allowed),
        "answer_parser_kind": SCOPED_PARSER,
        "format": ANSWER_PLAN_FORMAT,
        "handle_group_by_id": dict(groups),
        "messages": list(messages),
        "messages_sha256": identity_sha256(list(messages)),
        "mode": "specialist",
        "prompt_token_proxy": source_prompt.prompt_token_proxy,
        "provider_input": dict(provider),
        "scope_projection": scope.projection(),
        "scope_receipt_sha256": scope.receipt_sha256,
        "specialist_advisories_sha256": source_prompt.specialist_advisories_sha256,
        "terminal_prompt_receipt_sha256": require_sha256(
            terminal.get("terminal_prompt_receipt_sha256"), "specialist terminal"
        ),
        "validation_contract": validation,
    }
    projection = {**body, "answer_plan_receipt_sha256": identity_sha256(body)}
    assert_gold_blind(projection, path="confirmation_specialist_scoped_plan")
    return ConfirmationSpecialistAnswerPlan(
        ordinal,
        str(common["question_id"]),
        SCOPED_PARSER,
        str(common["parent_prediction"]),
        dict(provider),
        messages,
        scope,
        projection,
    )


def compile_confirmation_specialist_answer_plans(
    construction: ConfirmationSpecialistConstruction,
) -> tuple[ConfirmationSpecialistAnswerPlan, ...]:
    _require(
        type(construction) is ConfirmationSpecialistConstruction,
        "specialist plan compiler requires an exact construction",
    )
    payload = construction.artifact.payload
    unsigned = dict(payload)
    declared = require_sha256(
        unsigned.pop("construction_identity_sha256", None),
        "specialist construction identity",
    )
    _require(
        payload.get("format") == CONSTRUCTION_FORMAT
        and payload.get("gold_loaded") is False
        and payload.get("target_labels_loaded") is False
        and payload.get("target_plan_loaded") is False
        and identity_sha256(unsigned) == declared
        and payload.get("question_count") == len(construction.questions)
        and list(construction.questions) == payload.get("questions"),
        "specialist construction boundary changed",
    )
    plans = tuple(_answer_plan(raw, ordinal) for ordinal, raw in enumerate(construction.questions))
    _require(
        bool(plans)
        and tuple(row.ordinal for row in plans) == tuple(range(len(plans)))
        and len({row.question_id for row in plans}) == len(plans),
        "specialist answer plan population changed",
    )
    return plans


def publish_confirmation_specialist_preflight(
    construction: ConfirmationSpecialistConstruction,
    *,
    output_root: str | Path,
    model: str = terra.TERRA_MODEL,
    gateway_url: str = terra.TERRA_GATEWAY_URL,
    max_concurrency: int = 4,
) -> ConfirmationSpecialistPreflight:
    plans = compile_confirmation_specialist_answer_plans(construction)
    submitted = tuple(row for row in plans if row.parser_kind != PASSTHROUGH_PARSER)
    _require(bool(submitted), "specialist provider population is empty")
    runtime = {
        "gateway_url": gateway_url,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "input_token_cap": MAX_CHAT_PROMPT_TOKENS,
        "max_concurrency": max_concurrency,
        "model": model,
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "retry_count": 0,
    }
    prompt_payload = terra.compile_intermediate_prompt_artifact(
        stage_id=PROMPT_STAGE_ID,
        ordered_question_ids=[row.question_id for row in submitted],
        source_row_receipts=[row.receipt_sha256 for row in submitted],
        messages=[list(row.messages) for row in submitted],
        runtime=runtime,
        stage_bindings={
            "answer_plan_population_sha256": identity_sha256(
                [row.receipt_sha256 for row in plans]
            ),
            "construction_artifact_sha256": construction.artifact.sha256,
            "ordinary_typed_plan_count": sum(
                row.parser_kind == ORDINARY_TYPED_PARSER for row in plans
            ),
            "parent_passthrough_count": sum(
                row.parser_kind == PASSTHROUGH_PARSER for row in plans
            ),
            "submitted_plan_count": len(submitted),
            "submitted_plan_population_sha256": identity_sha256(
                [row.receipt_sha256 for row in submitted]
            ),
        },
    )
    root = Path(output_root)
    prompt_artifact, _created = terra.publish_sealed_artifact(
        root / PROMPT_NAME, prompt_payload
    )
    lifecycle, _created = terra.publish_lifecycle_preflight(
        prompt_artifact_path=prompt_artifact.path,
        expected_prompt_artifact_sha256=prompt_artifact.sha256,
        output_root=root,
    )
    return ConfirmationSpecialistPreflight(
        construction, prompt_artifact, lifecycle, plans
    )


def approve_confirmation_specialist_release(
    preflight: ConfirmationSpecialistPreflight,
    *,
    output_root: str | Path,
    approve_provider_release: bool,
    authorized_provider_calls: int,
) -> terra.SealedArtifact:
    _require(
        type(preflight) is ConfirmationSpecialistPreflight,
        "specialist release requires the exact preflight",
    )
    artifact, _created = terra.approve_provider_release(
        prompt_artifact_path=preflight.prompt_artifact.path,
        expected_prompt_artifact_sha256=preflight.prompt_artifact.sha256,
        output_root=output_root,
        expected_lifecycle_preflight_sha256=(
            preflight.lifecycle_preflight_artifact.sha256
        ),
        approve_provider_release=approve_provider_release,
        authorized_provider_calls=authorized_provider_calls,
    )
    return artifact


def run_confirmation_specialist_provider(
    preflight: ConfirmationSpecialistPreflight,
    *,
    output_root: str | Path,
    expected_release_sha256: str,
    enable_provider: bool,
    authorized_provider_calls: int,
    api_key_env: str = terra.DEFAULT_API_KEY_ENV,
    client_factory: terra.ClientFactory | None = None,
) -> dict[str, Any]:
    _require(
        type(preflight) is ConfirmationSpecialistPreflight,
        "specialist provider requires the exact preflight",
    )
    kwargs: dict[str, Any] = {}
    if client_factory is not None:
        kwargs["client_factory"] = client_factory
    return terra.run_provider_completion(
        prompt_artifact_path=preflight.prompt_artifact.path,
        expected_prompt_artifact_sha256=preflight.prompt_artifact.sha256,
        output_root=output_root,
        expected_lifecycle_preflight_sha256=(
            preflight.lifecycle_preflight_artifact.sha256
        ),
        expected_release_sha256=expected_release_sha256,
        enable_provider=enable_provider,
        authorized_provider_calls=authorized_provider_calls,
        api_key_env=api_key_env,
        **kwargs,
    )


def _parse_answer(
    plan: ConfirmationSpecialistAnswerPlan,
    completion: str,
) -> Any:
    if plan.parser_kind == SCOPED_PARSER:
        _require(
            type(plan.scope) is SpecialistValidationScope,
            "scoped specialist plan lost its proof scope",
        )
        return parse_specialist_scoped_completion(
            completion,
            parent_prediction=plan.parent_prediction,
            scope=plan.scope,
        )
    _require(
        plan.parser_kind == ORDINARY_TYPED_PARSER
        and type(plan.scope) is OrdinaryTypedScope,
        "ordinary typed plan lost its validation scope",
    )
    scope = plan.scope
    return parse_typed_final_completion(
        completion,
        parent_prediction=plan.parent_prediction,
        allowed_handle_ids=scope.allowed_handle_ids,
        handle_group_by_id=scope.handle_group_by_id,
        story_coherence=scope.story_coherence,
        preservation_requirements=scope.preservation_requirements,
        validation_contract=scope.validation_contract,
    )


def _v2_result_rows(
    preflight: ConfirmationSpecialistPreflight,
    batch: Any,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    submitted = preflight.submitted_plans
    _require(
        len(batch.logical_completions) == len(submitted),
        "specialist completion population changed",
    )
    records = {row.messages_sha256: row for row in batch.unique_records}
    _require(
        len(records) == preflight.required_provider_calls,
        "specialist checkpoint identities repeat",
    )
    completions = iter(batch.logical_completions)
    results: list[dict[str, Any]] = []
    for plan in preflight.plans:
        projection = dict(plan.projection)
        common = {
            "answer_plan_receipt_sha256": plan.receipt_sha256,
            "construction_question_receipt_sha256": projection[
                "construction_question_receipt_sha256"
            ],
            "dated_question_sha256": projection["dated_question_sha256"],
            "format": f"{V2_RUN_FORMAT}-result-row-v1",
            "gold_loaded": False,
            "ordinal": plan.ordinal,
            "parent_prediction_sha256": projection["parent_prediction_sha256"],
            "physical_provider_calls": 0,
            "question_id": plan.question_id,
            "question_sha256": projection["question_sha256"],
            "retained_transformer_token_state_bytes": 0,
            "route_id": projection["route_id"],
        }
        if plan.parser_kind == PASSTHROUGH_PARSER:
            body = {
                **common,
                "call_key_sha256": None,
                "changed_from_parent": False,
                "completion_receipt_sha256": None,
                "decision": "parent_passthrough",
                "parse_error_code": "none",
                "parse_receipt_sha256": None,
                "prediction": plan.parent_prediction,
                "prediction_sha256": quote_sha256(plan.parent_prediction),
                "prediction_source": "confirmation_specialist_parent_passthrough_v2",
                "request_journal_sha256": None,
                "response_journal_sha256": None,
                "solver_valid": True,
                "specialist_scope_receipt_sha256": None,
                "used_handle_ids": [],
                "validation_basis": "parent_passthrough",
                "validator_policy_format": VALIDATOR_POLICY_FORMAT,
            }
        else:
            completion = next(completions)
            messages_sha = str(projection["messages_sha256"])
            record = records.get(messages_sha)
            _require(
                record is not None
                and record.completion == completion
                and record.checkpoint_hit is True
                and record.physical_call is False,
                f"specialist checkpoint changed at ordinal {plan.ordinal}",
            )
            parsed = _parse_answer(plan, completion)
            _require(
                parsed.valid and parsed.decision in {"keep_parent", "replace"},
                f"invalid specialist completion at ordinal {plan.ordinal}; refusing unproved fallback",
            )
            prediction = (
                parsed.prediction
                if parsed.decision == "replace"
                else plan.parent_prediction
            )
            proof_kind = (
                parsed.proof_kind
                if plan.parser_kind == SCOPED_PARSER
                else f"ordinary_typed_{parsed.validation_basis}"
            )
            proof_receipt = (
                parsed.proof_receipt_sha256
                if plan.parser_kind == SCOPED_PARSER
                else plan.scope.transform_receipt_sha256  # type: ignore[union-attr]
            )
            body = {
                **common,
                "call_key_sha256": record.call_key_sha256,
                "changed_from_parent": prediction != plan.parent_prediction,
                "completion_receipt_sha256": record.completion_sha256,
                "decision": parsed.decision,
                "parse_error_code": parsed.error_code,
                "parse_receipt_sha256": parsed.receipt_sha256,
                "prediction": prediction,
                "prediction_sha256": quote_sha256(prediction),
                "prediction_source": (
                    f"confirmation_specialist_{plan.parser_kind}_validated_"
                    f"{parsed.decision}_v2"
                ),
                "proof_kind": proof_kind,
                "proof_receipt_sha256": proof_receipt,
                "request_journal_sha256": record.request_journal_sha256,
                "response_journal_sha256": record.response_journal_sha256,
                "solver_valid": True,
                "specialist_scope_receipt_sha256": projection[
                    "scope_receipt_sha256"
                ],
                "used_handle_ids": list(parsed.used_handle_ids),
                "validation_basis": parsed.validation_basis,
                "validator_policy_format": VALIDATOR_POLICY_FORMAT,
            }
        assert_gold_blind(body, path="confirmation_specialist_v2_result")
        results.append({**body, "source_row_sha256": identity_sha256(body)})
    _require(
        tuple(row["ordinal"] for row in results) == tuple(range(len(results))),
        "specialist V2 result order changed",
    )
    judges = tuple(prediction_row_projection(row) for row in results)
    return tuple(results), judges


def _v2_run_payload(
    preflight: ConfirmationSpecialistPreflight,
    release: terra.SealedArtifact,
    completion: terra.SealedArtifact,
    batch: Any,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    rows, judges = _v2_result_rows(preflight, batch)
    parser_counts = Counter(row.parser_kind for row in preflight.plans)
    body: dict[str, Any] = {
        "answer_plan_population_sha256": identity_sha256(
            [row.receipt_sha256 for row in preflight.plans]
        ),
        "changed_prediction_count": sum(row["changed_from_parent"] for row in rows),
        "completion_artifact_sha256": completion.sha256,
        "format": V2_RUN_FORMAT,
        "gold_loaded": False,
        "judge_rows": list(judges),
        "lifecycle_preflight_sha256": preflight.lifecycle_preflight_artifact.sha256,
        "ordinary_typed_question_count": parser_counts[ORDINARY_TYPED_PARSER],
        "parent_passthrough_count": parser_counts[PASSTHROUGH_PARSER],
        "physical_provider_calls_during_materialization": 0,
        "prompt_artifact_sha256": preflight.prompt_artifact.sha256,
        "provider_question_count": len(preflight.submitted_plans),
        "question_count": len(rows),
        "questions": list(rows),
        "release_artifact_sha256": release.sha256,
        "retained_transformer_token_state_bytes": 0,
        "scoped_specialist_question_count": parser_counts[SCOPED_PARSER],
        "source_construction_artifact_sha256": preflight.construction.artifact.sha256,
    }
    assert_gold_blind(body, path="confirmation_specialist_v2_run")
    return ({**body, "artifact_identity_sha256": identity_sha256(body)}, rows, judges)


def materialize_confirmation_specialist_v2(
    preflight: ConfirmationSpecialistPreflight,
    *,
    output_root: str | Path,
    expected_release_sha256: str,
) -> ConfirmationSpecialistV2Materialization:
    completion, _created = terra.materialize_completions(
        prompt_artifact_path=preflight.prompt_artifact.path,
        expected_prompt_artifact_sha256=preflight.prompt_artifact.sha256,
        output_root=output_root,
        expected_lifecycle_preflight_sha256=(
            preflight.lifecycle_preflight_artifact.sha256
        ),
        expected_release_sha256=expected_release_sha256,
    )
    batch = terra.load_completed_batch(
        prompt_artifact_path=preflight.prompt_artifact.path,
        expected_prompt_artifact_sha256=preflight.prompt_artifact.sha256,
        output_root=output_root,
        expected_lifecycle_preflight_sha256=(
            preflight.lifecycle_preflight_artifact.sha256
        ),
        expected_release_sha256=expected_release_sha256,
        expected_completion_sha256=completion.sha256,
    )
    release = terra.read_sealed_artifact(
        Path(output_root) / terra.RELEASE_NAME,
        expected_sha256=expected_release_sha256,
        label="specialist provider release",
    )
    payload, rows, judges = _v2_run_payload(
        preflight, release, completion, batch
    )
    run, _created = terra.publish_sealed_artifact(
        Path(output_root) / V2_RUN_NAME, payload
    )
    return ConfirmationSpecialistV2Materialization(
        preflight,
        release,
        completion,
        run,
        batch,
        tuple(row["prediction"] for row in rows),
        rows,
        judges,
    )


def replay_confirmation_specialist_v2(
    preflight: ConfirmationSpecialistPreflight,
    *,
    output_root: str | Path,
    expected_release_sha256: str,
    expected_completion_sha256: str,
    expected_run_sha256: str,
) -> VerifiedConfirmationSpecialistV2Plane:
    completion_replay, _created = terra.replay_completions(
        prompt_artifact_path=preflight.prompt_artifact.path,
        expected_prompt_artifact_sha256=preflight.prompt_artifact.sha256,
        output_root=output_root,
        expected_lifecycle_preflight_sha256=(
            preflight.lifecycle_preflight_artifact.sha256
        ),
        expected_release_sha256=expected_release_sha256,
        expected_completion_sha256=expected_completion_sha256,
    )
    batch = terra.load_completed_batch(
        prompt_artifact_path=preflight.prompt_artifact.path,
        expected_prompt_artifact_sha256=preflight.prompt_artifact.sha256,
        output_root=output_root,
        expected_lifecycle_preflight_sha256=(
            preflight.lifecycle_preflight_artifact.sha256
        ),
        expected_release_sha256=expected_release_sha256,
        expected_completion_sha256=expected_completion_sha256,
    )
    release = terra.read_sealed_artifact(
        Path(output_root) / terra.RELEASE_NAME,
        expected_sha256=expected_release_sha256,
        label="specialist provider release",
    )
    completion = terra.read_sealed_artifact(
        Path(output_root) / terra.COMPLETION_NAME,
        expected_sha256=expected_completion_sha256,
        label="specialist completions",
    )
    payload, rows, judges = _v2_run_payload(
        preflight, release, completion, batch
    )
    run = terra.read_sealed_artifact(
        Path(output_root) / V2_RUN_NAME,
        expected_sha256=expected_run_sha256,
        label="specialist V2 run",
    )
    _require(run.payload == payload, "specialist V2 replay changed result bytes")
    replay, _created = terra.publish_sealed_artifact(
        Path(output_root) / V2_REPLAY_NAME, payload
    )
    _require(replay.sha256 == run.sha256, "specialist V2 replay is not byte-identical")
    _require(
        completion_replay.sha256 == completion.sha256,
        "specialist completion replay changed",
    )
    return VerifiedConfirmationSpecialistV2Plane(
        preflight.construction.artifact,
        preflight.prompt_artifact,
        preflight.lifecycle_preflight_artifact,
        release,
        completion,
        run,
        replay,
        batch,
        preflight.plans,
        tuple(row["prediction"] for row in rows),
        rows,
        judges,
    )


def _v3_source_bundle(v2_plane: VerifiedConfirmationSpecialistV2Plane) -> Any:
    _require(
        type(v2_plane) is VerifiedConfirmationSpecialistV2Plane,
        "V3 requires the exact verified specialist V2 plane",
    )
    _require(
        v2_plane.run_artifact.sha256 == v2_plane.replay_artifact.sha256
        and v2_plane.run_artifact.payload == v2_plane.replay_artifact.payload,
        "specialist V2 run/replay are not byte-identical",
    )
    raw_plans = tuple(dict(row.projection) for row in v2_plane.plans)
    providers = {
        row.ordinal: dict(row.provider_input)
        for row in v2_plane.plans
        if row.provider_input is not None
    }
    _require(
        tuple(row["ordinal"] for row in raw_plans)
        == tuple(range(len(raw_plans)))
        == tuple(row["ordinal"] for row in v2_plane.result_rows),
        "specialist V3 source order changed",
    )
    return reconcile_v3.SourceBundle(
        v2_plane.lifecycle_preflight_artifact,
        v2_plane.run_artifact,
        v2_plane.replay_artifact,
        raw_plans,
        v2_plane.result_rows,
        providers,
    )


def audit_confirmation_specialist_v3(
    v2_plane: VerifiedConfirmationSpecialistV2Plane,
) -> ConfirmationSpecialistV3Audit:
    """Run the three deterministic lanes and expose their freeze receipts."""

    bundle = _v3_source_bundle(v2_plane)
    audits = reconcile_v3.build_lane_audits(bundle)
    return ConfirmationSpecialistV3Audit(bundle, audits)


def _v3_status_rows(
    questions: Sequence[Mapping[str, Any]],
    v2_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    statuses: list[dict[str, Any]] = []
    for question, prior in zip(questions, v2_rows, strict=True):
        reconciliation = question.get("reconciliation")
        body = {
            "combined_decision": question.get("decision_lane"),
            "decision_lane": question.get("decision_lane"),
            "format": V3_STATUS_FORMAT,
            "gold_loaded": False,
            "prediction_sha256": question.get("prediction_sha256"),
            "provider_calls": 0,
            "question_id": question.get("question_id"),
            "resolved": question.get("decision_lane") != "v2_fallback",
            "retained_transformer_token_state_bytes": 0,
            "selected_reconciliation_receipt_sha256": (
                None
                if type(reconciliation) is not dict
                else reconciliation.get("receipt_sha256")
            ),
            "source_v2_row_sha256": prior.get("source_row_sha256"),
            "source_v3_row_sha256": question.get("source_row_sha256"),
            "status": (
                "resolved"
                if question.get("decision_lane") != "v2_fallback"
                else "unresolved"
            ),
        }
        statuses.append(_sealed(body))
    return tuple(statuses)


def _v3_policy_projection(
    audit: ConfirmationSpecialistV3Audit,
    *,
    expected_status_population_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Seal the live arbitrary-N lane receipts, never validation targets."""

    expected = dict(expected_status_population_sha256s)
    observed = audit.status_population_sha256s
    _require(
        set(expected) == {"temporal", "numeric", "authority"},
        "specialist V3 lane freeze is incomplete",
    )
    for lane, digest in expected.items():
        _require(
            require_sha256(digest, f"expected {lane} specialist lane")
            == observed[lane],
            f"specialist {lane} lane differs from its current sealed audit",
        )
    bundle = audit.source_bundle
    audits = audit.lane_audits
    body = {
        "authority_composition_gate": {
            "bounded_composite_minimum_operand_count": 2,
            "distinct_identity_requires_explicit_dedup_proof": True,
            "recurring_frequency_requires_closure": True,
        },
        "composition_order": list(reconcile_v3.COMPOSITION_ORDER),
        "format": V3_POLICY_FORMAT,
        "gold_loaded": False,
        "hard_complete_chat_token_cap": HARD_COMPLETE_CHAT_TOKEN_CAP,
        "local_lane_audit_receipts": {
            lane: getattr(audits, lane).projection()["receipt_sha256"]
            for lane in ("authority", "numeric", "temporal")
        },
        "local_lane_resolved_population_sha256s": {
            lane: getattr(audits, lane).resolved_population_sha256
            for lane in ("authority", "numeric", "temporal")
        },
        "sealed_lane_status_population_sha256s": expected,
        "physical_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
        "v2_preflight_artifact_sha256": bundle.preflight.sha256,
        "v2_replay_artifact_sha256": bundle.replay.sha256,
        "v2_run_artifact_sha256": bundle.run.sha256,
    }
    assert_gold_blind(body, path="confirmation_specialist_v3_policy")
    return _sealed(body)


def _v3_payload(
    audit: ConfirmationSpecialistV3Audit,
    *,
    expected_status_population_sha256s: Mapping[str, str],
) -> tuple[
    dict[str, Any],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    _require(
        type(audit) is ConfirmationSpecialistV3Audit,
        "specialist V3 materialization requires an exact audit",
    )
    bundle = audit.source_bundle
    audits = audit.lane_audits
    policy = _v3_policy_projection(
        audit,
        expected_status_population_sha256s=expected_status_population_sha256s,
    )
    questions = reconcile_v3.compose_rows(bundle, audits)
    judges = tuple(prediction_row_projection(row) for row in questions)
    statuses = _v3_status_rows(questions, bundle.rows)
    prompt_runtime = bundle.preflight.payload["runtime"]
    prompt_rows = [
        row for row in bundle.plans if row.get("mode") == "specialist"
    ]
    body: dict[str, Any] = {
        "changed_from_v2_count": sum(row["changed_from_v2"] for row in questions),
        "composition_policy": policy,
        "format": V3_RUN_FORMAT,
        "gold_loaded": False,
        "judge_rows": list(judges),
        "lane_audits": {
            "authority": audits.authority.projection(),
            "numeric": audits.numeric.projection(),
            "temporal": audits.temporal.projection(),
        },
        "max_chat_prompt_tokens": prompt_runtime["input_token_cap"],
        "observed_max_complete_envelope_tokens": max(
            (
                int(row["prompt_token_proxy"]) + OUTPUT_TOKEN_RESERVE
                for row in prompt_rows
            ),
            default=0,
        ),
        "ordered_status_rows": list(statuses),
        "ordered_status_population_sha256": identity_sha256(
            [row["receipt_sha256"] for row in statuses]
        ),
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "physical_provider_calls_during_materialization": 0,
        "question_count": len(questions),
        "questions": list(questions),
        "retained_transformer_token_state_bytes": 0,
        "v2_preflight_artifact_sha256": bundle.preflight.sha256,
        "v2_replay_artifact_sha256": bundle.replay.sha256,
        "v2_run_artifact_sha256": bundle.run.sha256,
    }
    assert_gold_blind(body, path="confirmation_specialist_v3_run")
    return (
        {**body, "artifact_identity_sha256": identity_sha256(body)},
        tuple(questions),
        judges,
        statuses,
    )


def materialize_confirmation_specialist_v3(
    audit: ConfirmationSpecialistV3Audit,
    *,
    output_root: str | Path,
    expected_status_population_sha256s: Mapping[str, str],
) -> terra.SealedArtifact:
    payload, _rows, _judges, _statuses = _v3_payload(
        audit,
        expected_status_population_sha256s=expected_status_population_sha256s,
    )
    artifact, _created = terra.publish_sealed_artifact(
        Path(output_root) / V3_RUN_NAME, payload
    )
    return artifact


def replay_confirmation_specialist_v3(
    v2_plane: VerifiedConfirmationSpecialistV2Plane,
    *,
    output_root: str | Path,
    expected_status_population_sha256s: Mapping[str, str],
    expected_run_sha256: str,
) -> VerifiedConfirmationSpecialistV3Plane:
    audit = audit_confirmation_specialist_v3(v2_plane)
    payload, rows, judges, statuses = _v3_payload(
        audit,
        expected_status_population_sha256s=expected_status_population_sha256s,
    )
    run = terra.read_sealed_artifact(
        Path(output_root) / V3_RUN_NAME,
        expected_sha256=expected_run_sha256,
        label="specialist V3 run",
    )
    _require(run.payload == payload, "specialist V3 replay changed result bytes")
    replay, _created = terra.publish_sealed_artifact(
        Path(output_root) / V3_REPLAY_NAME, payload
    )
    _require(replay.sha256 == run.sha256, "specialist V3 replay is not byte-identical")
    return VerifiedConfirmationSpecialistV3Plane(
        v2_plane,
        run,
        replay,
        tuple(row["prediction"] for row in rows),
        rows,
        judges,
        statuses,
        dict(audit.status_population_sha256s),
    )


_TERMINAL_ROUTING_KEYS = frozenset(
    {
        "allowlist",
        "eligible_ordinals",
        "eligible_question_ids",
        "miss_ordinals",
        "ordinal",
        "ordinals",
        "target_ordinals",
        "target_question_ids",
        "validation_ordinals",
        "validation_question_ids",
        "whitelist",
    }
)


def _routing_neutral(value: object) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _routing_neutral(child)
            for key, child in value.items()
            if str(key).casefold() not in _TERMINAL_ROUTING_KEYS
        }
    if isinstance(value, (tuple, list)):
        return [_routing_neutral(child) for child in value]
    return value


def _answer_eligibility_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only fields consumed by the semantic residual gate.

    The upstream rows carry population positions and large proof payloads.  A
    terminal eligibility decision needs neither: its answer semantics are the
    final lane, validity, disposition, cited-handle count, and prediction.
    """

    keys = (
        "format",
        "prediction",
        "decision",
        "used_handle_ids",
        "solver_valid",
        "parse_error_code",
        "decision_lane",
        "prediction_source",
        "route_id",
    )
    result = {key: _routing_neutral(row[key]) for key in keys if key in row}
    _require("prediction" in result, "terminal eligibility answer lost prediction")
    assert_gold_blind(result, path="confirmation_specialist_terminal_answer")
    return result


def _frontier_eligibility_projection(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    keys = (
        "format",
        "mode",
        "frontier_mode",
        "truncated",
        "selection_truncated",
        "closed",
        "sufficient",
        "unresolved_slot_ids",
        "missing_slot_ids",
    )
    body = {key: _routing_neutral(value[key]) for key in keys if key in value}
    return _sealed(body) if body else {}


def _route_eligibility_projection(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    keys = (
        "format",
        "style",
        "temporal_mode",
        "applicable_specialist_ids",
        "applicable_mechanism_ids",
    )
    route = {key: _routing_neutral(value[key]) for key in keys if key in value}
    legacy = value.get("legacy_route")
    if isinstance(legacy, Mapping):
        route["legacy_route"] = {
            key: _routing_neutral(legacy[key])
            for key in ("format", "style")
            if key in legacy
        }
    return route


def _construction_eligibility_projection(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    projection: dict[str, Any] = {
        key: _routing_neutral(row[key])
        for key in ("format", "mode", "applicable_specialist_ids")
        if key in row
    }
    projection["route"] = _route_eligibility_projection(row.get("route"))

    raw_methods = row.get("methods")
    methods: list[dict[str, Any]] = []
    if isinstance(raw_methods, Sequence) and not isinstance(
        raw_methods, (str, bytes, bytearray)
    ):
        for raw_method in raw_methods:
            if not isinstance(raw_method, Mapping):
                continue
            method = {
                key: _routing_neutral(raw_method[key])
                for key in ("format", "mechanism", "mechanism_id")
                if key in raw_method
            }
            contributions = raw_method.get("typed_contributions")
            if isinstance(contributions, Sequence) and not isinstance(
                contributions, (str, bytes, bytearray)
            ):
                method["typed_contributions"] = [
                    projected
                    for raw in contributions
                    if (projected := _frontier_eligibility_projection(raw))
                ]
            methods.append(method)
    projection["methods"] = methods

    terminal = row.get("terminal_prompt")
    if isinstance(terminal, Mapping):
        provider = terminal.get("provider_input")
        if isinstance(provider, Mapping):
            typed = provider.get("typed_evidence")
            if isinstance(typed, Mapping):
                typed_projection: dict[str, Any] = {}
                operator = typed.get("operator_spec")
                if isinstance(operator, Mapping):
                    typed_projection["operator_spec"] = {
                        key: _routing_neutral(operator[key])
                        for key in ("format", "style")
                        if key in operator
                    }
                frontier = _frontier_eligibility_projection(typed.get("frontier"))
                if frontier:
                    typed_projection["frontier"] = frontier
                if typed_projection:
                    projection["terminal_prompt"] = {
                        "provider_input": {"typed_evidence": typed_projection}
                    }
    assert_gold_blind(
        projection, path="confirmation_specialist_terminal_construction"
    )
    return projection


def _reconciliation_eligibility_projection(
    row: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if row is None:
        return None
    keys = (
        "format",
        "decision_scope",
        "reconciliation_scope",
        "scope",
        "combined_decision",
        "final_decision",
        "resolved",
        "sufficient",
        "solver_valid",
        "decision",
        "disposition",
        "outcome",
        "resolution_status",
        "status",
    )
    result = {key: _routing_neutral(row[key]) for key in keys if key in row}
    for key in ("frontier", "classified_frontier", "sufficiency", "sufficiency_gate"):
        projection = _frontier_eligibility_projection(row.get(key))
        if projection:
            result[key] = projection
    assert_gold_blind(
        result, path="confirmation_specialist_terminal_reconciliation"
    )
    return result


def compile_confirmation_terminal_parent_payload(
    sources: Sequence[ConfirmationTerminalParentSource],
    *,
    policy_manifest_sha256: str,
    treatment_file_sha256: str,
    treatment_preflight_sha256: str,
    ordered_question_ids_sha256: str,
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """Compile and replay the terminal gate for an arbitrary V3 population."""

    from tools.confirmation_terminal_policy_boundary import (  # noqa: PLC0415
        ELIGIBILITY_INPUT_FORMAT,
        PARENT_POPULATION_FORMAT,
        PARENT_ROW_FORMAT,
    )
    from tools.matched_eval.semantic_residual_eligibility import (  # noqa: PLC0415
        SemanticResidualEligibilityPolicy,
        evaluate_semantic_residual_eligibility,
        replay_semantic_residual_eligibility,
    )

    frozen = tuple(sources)
    _require(
        bool(frozen)
        and all(type(source) is ConfirmationTerminalParentSource for source in frozen),
        "terminal parent sources must be a non-empty exact population",
    )
    question_ids = tuple(source.question_id for source in frozen)
    _require(
        len(question_ids) == len(set(question_ids)),
        "terminal parent source questions repeat",
    )
    ordered_sha = require_sha256(
        ordered_question_ids_sha256, "terminal parent ordered question IDs"
    )
    _require(
        ordered_sha == canonical_sha256(list(question_ids)),
        "terminal parent source order changed",
    )
    bindings = {
        "policy_manifest_sha256": require_sha256(
            policy_manifest_sha256, "terminal parent policy manifest"
        ),
        "treatment_file_sha256": require_sha256(
            treatment_file_sha256, "terminal parent treatment"
        ),
        "treatment_preflight_sha256": require_sha256(
            treatment_preflight_sha256, "terminal parent treatment preflight"
        ),
    }
    policy = SemanticResidualEligibilityPolicy()
    decisions: list[Any] = []
    rows: list[dict[str, Any]] = []
    for source in frozen:
        answer = _answer_eligibility_projection(source.answer_row)
        construction = _construction_eligibility_projection(source.construction_row)
        prior = (
            None
            if source.prior_answer_row is None
            else _answer_eligibility_projection(source.prior_answer_row)
        )
        reconciliation = _reconciliation_eligibility_projection(
            source.reconciliation_row
        )
        decision = evaluate_semantic_residual_eligibility(
            answer,
            construction,
            prior_answer_row=prior,
            reconciliation_row=reconciliation,
            policy=policy,
        )
        replayed = replay_semantic_residual_eligibility(
            answer,
            construction,
            decision,
            prior_answer_row=prior,
            reconciliation_row=reconciliation,
            policy=policy,
        )
        _require(
            replayed.projection() == decision.projection(),
            "terminal eligibility replay changed",
        )
        eligibility = _sealed(
            {
                "answer_row": answer,
                "construction_row": construction,
                "format": ELIGIBILITY_INPUT_FORMAT,
                "prior_answer_row": prior,
                "reconciliation_row": reconciliation,
            }
        )
        prediction = require_text(answer.get("prediction"), "terminal parent prediction")
        row_body = {
            "dated_question": source.dated_question,
            "dated_question_sha256": hashlib.sha256(
                source.dated_question.encode("utf-8")
            ).hexdigest(),
            "eligibility_input": eligibility,
            "format": PARENT_ROW_FORMAT,
            "namespace_id": source.namespace_id,
            "namespace_receipt_sha256": source.namespace_receipt_sha256,
            "parent_prediction": prediction,
            "parent_prediction_sha256": quote_sha256(prediction),
            "question": source.question,
            "question_id": source.question_id,
            "question_sha256": hashlib.sha256(
                source.question.encode("utf-8")
            ).hexdigest(),
            "source_row_receipt_sha256": source.source_row_receipt_sha256,
        }
        rows.append(_sealed(row_body, "row_receipt_sha256"))
        decisions.append(decision)
    parent_body: dict[str, Any] = {
        "format": PARENT_POPULATION_FORMAT,
        "gold_loaded": False,
        "ordered_question_ids_sha256": ordered_sha,
        "physical_provider_calls": 0,
        **bindings,
        "question_count": len(frozen),
        "rows": rows,
        "status": "complete",
    }
    payload = _sealed(parent_body, "artifact_identity_sha256")
    assert_gold_blind(payload, path="confirmation_specialist_terminal_parent")
    return payload, tuple(decisions)


def publish_confirmation_terminal_parent_sources(
    sources: Sequence[ConfirmationTerminalParentSource],
    *,
    policy_manifest_sha256: str,
    treatment_file_sha256: str,
    treatment_preflight_sha256: str,
    ordered_question_ids_sha256: str,
    output_path: str | Path,
) -> tuple[Any, tuple[Any, ...]]:
    """Publish a compiled parent and return its replayed gate decisions."""

    payload, decisions = compile_confirmation_terminal_parent_payload(
        sources,
        policy_manifest_sha256=policy_manifest_sha256,
        treatment_file_sha256=treatment_file_sha256,
        treatment_preflight_sha256=treatment_preflight_sha256,
        ordered_question_ids_sha256=ordered_question_ids_sha256,
    )
    artifact, _created = publish_sealed_json(output_path, payload)
    return artifact, decisions


def publish_confirmation_terminal_parent_population(
    v3_plane: VerifiedConfirmationSpecialistV3Plane,
    typed_plane: object,
    context: object,
    *,
    treatment_preflight_artifact: object,
    output_path: str | Path,
) -> Any:
    """Adapt exact V3 rows to the terminal P/R/L/G parent contract.

    Eligibility rows are reduced to the exact gate-consumed semantics, while
    the exact upstream row receipts remain bound outside that question-local
    projection.  The terminal gate can therefore replay its decision without
    population coordinates or proof payload clutter.
    """

    _require(
        type(v3_plane) is VerifiedConfirmationSpecialistV3Plane
        and type(v3_plane.v2_plane) is VerifiedConfirmationSpecialistV2Plane,
        "terminal parent adapter requires the exact specialist V3 plane",
    )
    parent = _require_typed_plane(typed_plane)
    from tools.confirmation_query_expansion_adapter import (  # noqa: PLC0415
        ConfirmationQueryExpansionContext,
    )
    _require(
        type(context) is ConfirmationQueryExpansionContext,
        "terminal parent adapter requires the exact confirmation context",
    )
    preflight_payload = _artifact_payload(
        treatment_preflight_artifact, "confirmation treatment preflight"
    )
    preflight_sha = _artifact_sha(
        treatment_preflight_artifact, "confirmation treatment preflight"
    )
    bindings = context.protected_plane.payload["bindings"]
    _require(
        preflight_sha == bindings["treatment_preflight_sha256"],
        "terminal parent adapter received another treatment preflight",
    )
    membership: dict[str, tuple[str, str]] = {}
    for raw in _exact_list(preflight_payload.get("namespaces"), "treatment namespaces"):
        namespace = _exact_dict(raw, "treatment namespace")
        namespace_id = require_text(namespace.get("namespace_id"), "treatment namespace ID")
        receipt = require_sha256(
            namespace.get("namespace_receipt_sha256"), "treatment namespace receipt"
        )
        for question_id in _exact_list(namespace.get("question_ids"), "namespace questions"):
            key = require_text(question_id, "namespace question ID")
            _require(key not in membership, "treatment namespaces overlap")
            membership[key] = (namespace_id, receipt)

    construction_rows = v3_plane.v2_plane.construction_artifact.payload["questions"]
    population_rows = context.population.rows
    count = len(v3_plane.result_rows)
    _require(
        len(parent.result_rows)
        == len(construction_rows)
        == len(population_rows)
        == len(v3_plane.v2_plane.result_rows)
        == len(v3_plane.status_rows)
        == count
        and set(membership)
        == {row.source.packet.question_id for row in population_rows},
        "terminal parent adapter population changed",
    )
    sources: list[ConfirmationTerminalParentSource] = []
    for index, (population, answer, construction, prior, reconciliation) in enumerate(
        zip(
            population_rows,
            v3_plane.result_rows,
            construction_rows,
            v3_plane.v2_plane.result_rows,
            v3_plane.status_rows,
            strict=True,
        )
    ):
        packet = population.source.packet
        question_id = packet.question_id
        namespace_id, namespace_receipt = membership[question_id]
        _require(
            answer.get("ordinal") == prior.get("ordinal") == construction.get("ordinal") == index
            and answer.get("question_id") == prior.get("question_id") == construction.get("question_id") == question_id
            and namespace_id == population.namespace.namespace_id,
            f"terminal parent source binding changed at row {index}",
        )
        question = _question_text(packet.dated_question)
        _require(
            packet.dated_question_sha256 == quote_sha256(packet.dated_question),
            f"terminal parent dated question changed at row {index}",
        )
        sources.append(
            ConfirmationTerminalParentSource(
                question_id=question_id,
                namespace_id=namespace_id,
                namespace_receipt_sha256=namespace_receipt,
                question=question,
                dated_question=packet.dated_question,
                source_row_receipt_sha256=require_sha256(
                    answer.get("source_row_sha256"), "terminal parent source row"
                ),
                answer_row=answer,
                construction_row=construction,
                prior_answer_row=prior,
                reconciliation_row=reconciliation,
            )
        )
    artifact, decisions = publish_confirmation_terminal_parent_sources(
        tuple(sources),
        policy_manifest_sha256=bindings["policy_manifest_sha256"],
        treatment_file_sha256=bindings["treatment_file_sha256"],
        treatment_preflight_sha256=preflight_sha,
        ordered_question_ids_sha256=canonical_sha256(
            [row.source.packet.question_id for row in population_rows]
        ),
        output_path=output_path,
    )
    _require(
        len(decisions) == count,
        "terminal eligibility receipt population changed",
    )
    return artifact


__all__ = [
    "ANSWER_PLAN_FORMAT",
    "CONSTRUCTION_FORMAT",
    "CONSTRUCTION_NAME",
    "CONSTRUCTION_REPLAY_NAME",
    "ConfirmationSpecialistAnswerPlan",
    "ConfirmationSpecialistConstruction",
    "ConfirmationSpecialistPreflight",
    "ConfirmationTerminalParentSource",
    "ConfirmationSpecialistV2Materialization",
    "ConfirmationSpecialistV3Audit",
    "ConfirmationSpecialistV3Error",
    "ORDINARY_TYPED_PARSER",
    "PASSTHROUGH_PARSER",
    "PROMPT_NAME",
    "SCOPED_PARSER",
    "TERMINAL_PARENT_NAME",
    "V2_REPLAY_NAME",
    "V2_RUN_NAME",
    "V3_POLICY_FORMAT",
    "V3_REPLAY_NAME",
    "V3_RUN_NAME",
    "VerifiedConfirmationSpecialistV2Plane",
    "VerifiedConfirmationSpecialistV3Plane",
    "approve_confirmation_specialist_release",
    "audit_confirmation_specialist_v3",
    "build_confirmation_specialist_construction_payload",
    "compile_confirmation_specialist_answer_plans",
    "compile_confirmation_terminal_parent_payload",
    "materialize_confirmation_specialist_v2",
    "materialize_confirmation_specialist_v3",
    "publish_confirmation_specialist_construction",
    "publish_confirmation_specialist_preflight",
    "publish_confirmation_terminal_parent_population",
    "publish_confirmation_terminal_parent_sources",
    "question_local_specialist_route",
    "replay_confirmation_specialist_construction",
    "replay_confirmation_specialist_v2",
    "replay_confirmation_specialist_v3",
    "run_confirmation_specialist_provider",
]
