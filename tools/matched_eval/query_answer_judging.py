"""Reusable changed-only Sol judging for verified query answer arms.

The answer arm is completely replayed and structurally validated before this
module opens benchmark gold or parent verdicts.  That validation fixes the
changed-prediction population.  The network phase then accepts only a sealed
population of standard binary-judge prompts and can write only immutable Sol
journals.  A separate client-free materializer joins those new verdicts with
the sealed S0-v2 parent verdicts and publishes the score artifacts.

The direct payload, routed fact, structured-operator, and V2 evidence-map
solver arms implement the same small verified-plane surface. Arm-specific
validation is isolated in ``_AnswerArmAdapter``; judging, scoring, and provider
execution are shared.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.domain._tokenizer import tokenizer_proxy_identity
from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval._binary_judge_protocol import (
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import (
    build_judge_prompt,
    exact_match,
    f1_score,
)
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from tools._routed_repair_routing import route_question

from . import adaptive_evidence_solver_judge_adapter
from . import closure_judging, judging, live, query_fact_answer_live
from . import query_evidence_map_solver_v2_live
from . import query_operator_refinement_live, query_payload_live
from .artifacts import SealedArtifact, publish_sealed_json, read_sealed_json
from .contracts import (
    MatchedEvalContractError,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from .ledger import (
    ScoreLedgerEntry,
    _validated_runtime_ledger,
    build_score_ledger,
)
from .population import EXPECTED_QUESTION_COUNT


JUDGE_PREFLIGHT_FORMAT = (
    "memory-condense-matched-query-answer-changed-only-sol-preflight-v1"
)
JUDGE_FORMAT = "memory-condense-matched-query-answer-changed-only-sol-judge-v1"
CHANGE_PROJECTION_FORMAT = (
    "memory-condense-matched-query-answer-change-projection-v1"
)
EMPTY_PROMPT_POPULATION_FORMAT = (
    "memory-condense-matched-query-answer-empty-prompts-v1"
)
JUDGE_PLAN_ID = "matched_query_answer_changed_only_sol_judge_v1"

JUDGE_PREFLIGHT_NAME = judging.JUDGE_PREFLIGHT_NAME
JUDGE_NAME = judging.JUDGE_NAME
JUDGE_REPLAY_NAME = judging.JUDGE_REPLAY_NAME
SCORE_LEDGER_NAME = judging.SCORE_LEDGER_NAME
SCORE_LEDGER_REPLAY_NAME = judging.SCORE_LEDGER_REPLAY_NAME
JUDGE_CHECKPOINT_DIR_NAME = "sol-query-answer-judge-calls-v1"


def _require(condition: object, message: str) -> None:
    if not condition:
        raise MatchedEvalContractError(message)


def _runtime_json(value: Mapping[str, Any]) -> dict[str, Any]:
    result = live._thaw_json(value)
    _require(type(result) is dict, "query answer runtime must be an object")
    return result


@dataclass(frozen=True, slots=True)
class _AnswerArmAdapter:
    kind: str
    plane_type: type
    row_type: type
    arm_label: str
    parent_arm_label: str
    arm_plan_id: str
    answer_plan_id: str
    answer_stage_id: str
    parent_stage_id: str
    renderer_id: str
    terra_prediction_source: str
    terra_mechanism_id: str
    fallback_prediction_source: str
    fallback_mechanism_id: str
    fallback_provider_calls: tuple[int, ...]
    inherited_verdict_source: str
    parent_answer_source_role: str
    parent_runtime_source_role: str
    source_roles: tuple[str, ...]
    keep_parent_prediction_source: str | None = None
    keep_parent_mechanism_id: str | None = None

    @property
    def judge_plan_id(self) -> str:
        return f"{JUDGE_PLAN_ID}:{self.kind}"


_PAYLOAD_ADAPTER = _AnswerArmAdapter(
    kind="query_payload",
    plane_type=query_payload_live.VerifiedQueryPayloadAnswerPlane,
    row_type=query_payload_live.VerifiedQueryPayloadAnswerRow,
    arm_label=query_payload_live.ARM_LABEL,
    parent_arm_label=query_payload_live.PARENT_ARM_LABEL,
    arm_plan_id=query_payload_live.ARM_PLAN_ID,
    answer_plan_id=query_payload_live.ANSWER_PLAN_ID,
    answer_stage_id=query_payload_live.ANSWER_STAGE_ID,
    parent_stage_id=query_payload_live.PAYLOAD_STAGE_ID,
    renderer_id=query_payload_live.RENDERER_ID,
    terra_prediction_source="terra_query_payload",
    terra_mechanism_id="terra_query_payload_responder",
    fallback_prediction_source="sealed_parent_fallback",
    fallback_mechanism_id="sealed_parent_prediction_reuse",
    fallback_provider_calls=(0,),
    inherited_verdict_source="sealed_parent_s0_v2_judge",
    parent_answer_source_role="parent_answer_run",
    parent_runtime_source_role="parent_runtime_ledger",
    source_roles=(
        "sealed_retrieval",
        "query_preflight",
        "query_run",
        "query_adapter",
        "parent_answer_run",
        "parent_runtime_ledger",
        "answer_preflight",
        "answer_run",
    ),
)

_FACT_ADAPTER = _AnswerArmAdapter(
    kind="query_fact",
    plane_type=query_fact_answer_live.VerifiedQueryFactAnswerPlane,
    row_type=query_fact_answer_live.VerifiedQueryFactAnswerRow,
    arm_label=query_fact_answer_live.ARM_LABEL,
    parent_arm_label=query_fact_answer_live.PARENT_ARM_LABEL,
    arm_plan_id=query_fact_answer_live.ARM_PLAN_ID,
    answer_plan_id=query_fact_answer_live.ANSWER_PLAN_ID,
    answer_stage_id=query_fact_answer_live.ANSWER_STAGE_ID,
    parent_stage_id=query_fact_answer_live.FACT_STAGE_ID,
    renderer_id=query_fact_answer_live.RENDERER_ID,
    terra_prediction_source="terra_query_fact_answer",
    terra_mechanism_id="terra_query_fact_responder",
    fallback_prediction_source="sealed_parent_fallback",
    fallback_mechanism_id="sealed_parent_prediction_reuse",
    fallback_provider_calls=(0,),
    inherited_verdict_source="sealed_parent_s0_v2_judge",
    parent_answer_source_role="parent_answer_run",
    parent_runtime_source_role="parent_runtime_ledger",
    source_roles=(
        "sealed_retrieval",
        "query_preflight",
        "query_run",
        "query_adapter",
        "query_fact_compression",
        "query_fact_compression_runtime_ledger",
        "parent_answer_run",
        "parent_runtime_ledger",
        "answer_preflight",
        "answer_run",
    ),
)

_OPERATOR_ADAPTER = _AnswerArmAdapter(
    kind="query_operator_refinement",
    plane_type=(
        query_operator_refinement_live.VerifiedQueryOperatorRefinementPlane
    ),
    row_type=query_operator_refinement_live.VerifiedQueryOperatorRefinementRow,
    arm_label=query_operator_refinement_live.ARM_LABEL,
    parent_arm_label=query_operator_refinement_live.PARENT_ARM_LABEL,
    arm_plan_id=query_operator_refinement_live.ARM_PLAN_ID,
    answer_plan_id=query_operator_refinement_live.ANSWER_PLAN_ID,
    answer_stage_id=query_operator_refinement_live.ANSWER_STAGE_ID,
    parent_stage_id=query_operator_refinement_live.OPERATOR_STAGE_ID,
    renderer_id=query_operator_refinement_live.RENDERER_ID,
    terra_prediction_source="terra_query_operator_refinement",
    terra_mechanism_id="terra_structured_operator_prediction",
    fallback_prediction_source="sealed_direct_query_fallback",
    fallback_mechanism_id="sealed_direct_query_prediction_reuse",
    fallback_provider_calls=(0, 1),
    inherited_verdict_source="sealed_parent_query_payload_judge",
    parent_answer_source_role="direct_answer_run",
    parent_runtime_source_role="direct_runtime_ledger",
    source_roles=(
        "sealed_retrieval",
        "query_preflight",
        "query_run",
        "query_adapter",
        "direct_answer_run",
        "direct_runtime_ledger",
        "answer_preflight",
        "answer_run",
    ),
)

_EVIDENCE_SOLVER_V2_ADAPTER = _AnswerArmAdapter(
    kind="query_evidence_map_solver_v2",
    plane_type=query_evidence_map_solver_v2_live.VerifiedEvidenceSolverPlane,
    row_type=query_evidence_map_solver_v2_live.VerifiedEvidenceSolverRow,
    arm_label=query_evidence_map_solver_v2_live.ARM_LABEL,
    parent_arm_label=query_evidence_map_solver_v2_live.PARENT_ARM_LABEL,
    arm_plan_id=query_evidence_map_solver_v2_live.ARM_PLAN_ID,
    answer_plan_id=query_evidence_map_solver_v2_live.ANSWER_PLAN_ID,
    answer_stage_id=query_evidence_map_solver_v2_live.ANSWER_STAGE_ID,
    parent_stage_id=query_evidence_map_solver_v2_live.SOLVER_STAGE_ID,
    renderer_id=query_evidence_map_solver_v2_live.SOLVER_RENDERER_ID,
    terra_prediction_source="terra_query_evidence_solver_v2",
    terra_mechanism_id="terra_validated_map_replacement_v2",
    fallback_prediction_source="sealed_direct_query_fallback",
    fallback_mechanism_id="sealed_direct_query_prediction_reuse",
    fallback_provider_calls=(0, 1),
    inherited_verdict_source="sealed_parent_query_payload_judge",
    parent_answer_source_role="direct_answer_run",
    parent_runtime_source_role="direct_runtime_ledger",
    source_roles=(
        "sealed_retrieval",
        "query_preflight",
        "query_run",
        "query_adapter",
        "direct_answer_run",
        "direct_runtime_ledger",
        "map_run",
        "map_runtime_ledger",
        "solver_preflight",
        "answer_run",
    ),
    keep_parent_prediction_source="terra_query_evidence_solver_v2_keep_parent",
    keep_parent_mechanism_id="terra_validated_map_keep_parent_v2",
)


def _adaptive_adapter(
    profile: adaptive_evidence_solver_judge_adapter.AdaptiveJudgeProfile,
    plane_type: type,
) -> _AnswerArmAdapter:
    return _AnswerArmAdapter(
        kind=profile.kind,
        plane_type=plane_type,
        row_type=(
            adaptive_evidence_solver_judge_adapter.VerifiedAdaptiveEvidenceSolverJudgeRow
        ),
        arm_label=profile.arm_label,
        parent_arm_label=query_payload_live.ARM_LABEL,
        arm_plan_id=profile.arm_plan_id,
        answer_plan_id=profile.answer_plan_id,
        answer_stage_id=adaptive_evidence_solver_judge_adapter.ANSWER_STAGE_ID,
        parent_stage_id=adaptive_evidence_solver_judge_adapter.SOLVER_STAGE_ID,
        renderer_id=adaptive_evidence_solver_judge_adapter.RENDERER_ID,
        terra_prediction_source=(
            "adaptive_validated_evidence_replacement_v3"
        ),
        terra_mechanism_id=(
            "terra_adaptive_validated_evidence_replacement_v3"
        ),
        fallback_prediction_source="sealed_direct_query_fallback",
        fallback_mechanism_id="sealed_direct_query_prediction_reuse",
        fallback_provider_calls=(0, 1),
        inherited_verdict_source="sealed_parent_query_payload_judge",
        parent_answer_source_role="direct_answer_run",
        parent_runtime_source_role="direct_runtime_ledger",
        source_roles=adaptive_evidence_solver_judge_adapter.SOURCE_ROLES,
        keep_parent_prediction_source=(
            "adaptive_validated_evidence_keep_parent_v3"
        ),
        keep_parent_mechanism_id=(
            "terra_adaptive_validated_evidence_keep_parent_v3"
        ),
    )


_ADAPTIVE_D_ADAPTER = _adaptive_adapter(
    adaptive_evidence_solver_judge_adapter.DIRECT_PROFILE,
    adaptive_evidence_solver_judge_adapter.VerifiedAdaptiveEvidenceSolverDirectJudgePlane,
)
_ADAPTIVE_P_ADAPTER = _adaptive_adapter(
    adaptive_evidence_solver_judge_adapter.PARTITION_PROFILE,
    adaptive_evidence_solver_judge_adapter.VerifiedAdaptiveEvidenceSolverPartitionJudgePlane,
)
_ADAPTIVE_G_ADAPTER = _adaptive_adapter(
    adaptive_evidence_solver_judge_adapter.GUIDED_PROFILE,
    adaptive_evidence_solver_judge_adapter.VerifiedAdaptiveEvidenceSolverGuidedJudgePlane,
)
_ADAPTIVE_DG_ADAPTER = _adaptive_adapter(
    adaptive_evidence_solver_judge_adapter.DIRECT_GUIDED_PROFILE,
    adaptive_evidence_solver_judge_adapter.VerifiedAdaptiveEvidenceSolverDirectGuidedJudgePlane,
)
_ADAPTIVE_ADAPTERS = (
    _ADAPTIVE_D_ADAPTER,
    _ADAPTIVE_P_ADAPTER,
    _ADAPTIVE_G_ADAPTER,
    _ADAPTIVE_DG_ADAPTER,
)

_ADAPTERS = (
    _PAYLOAD_ADAPTER,
    _FACT_ADAPTER,
    _OPERATOR_ADAPTER,
    _EVIDENCE_SOLVER_V2_ADAPTER,
    *_ADAPTIVE_ADAPTERS,
)


def _adapter_for_plane(answer_plane: object) -> _AnswerArmAdapter:
    for adapter in _ADAPTERS:
        if type(answer_plane) is adapter.plane_type:
            return adapter
    raise TypeError(
        "answer_plane must be an exact VerifiedQueryPayloadAnswerPlane or "
        "VerifiedQueryFactAnswerPlane, VerifiedQueryOperatorRefinementPlane, "
        "VerifiedEvidenceSolverPlane, or adaptive V3 judge plane"
    )


def _adapter_for_label(arm_label: str) -> _AnswerArmAdapter:
    for adapter in _ADAPTERS:
        if arm_label == adapter.arm_label:
            return adapter
    raise MatchedEvalContractError(f"unknown query-answer arm: {arm_label!r}")


@dataclass(frozen=True, slots=True)
class _QueryAnswerJudgePlan:
    answer_plane: Any
    adapter: _AnswerArmAdapter
    gold_rows: tuple[judging._GoldRow, ...]
    gold_population_sha256: str
    change_projection: Mapping[str, Any]
    prompt_rows: tuple[judging._JudgePromptRow, ...]
    prompt_population: FastPromptPopulation | None
    parent_judge: closure_judging._VerifiedParentJudge

    @property
    def changed_count(self) -> int:
        return len(self.prompt_rows)

    @property
    def required_calls(self) -> int:
        if self.prompt_population is None:
            return 0
        return self.prompt_population.unique_prompt_count


@dataclass(frozen=True, slots=True)
class SealedQueryAnswerJudgeProviderPopulation:
    """The only data accepted by the network-enabled judge phase."""

    preflight_artifact: SealedArtifact
    output_root: Path
    adapter: _AnswerArmAdapter
    prompts: tuple[tuple[dict[str, str], ...], ...]
    prompt_population: FastPromptPopulation | None
    required_calls: int


@dataclass(frozen=True, slots=True)
class QueryAnswerJudgeProviderResult:
    preflight_artifact: SealedArtifact
    batch: FastCompletionBatch | None
    physical_provider_calls: int
    checkpoint_hits: int


@dataclass(frozen=True, slots=True)
class QueryAnswerJudgeRunResult:
    judge_artifact: SealedArtifact
    score_ledger_artifact: SealedArtifact
    correct: int
    physical_provider_calls: int
    checkpoint_hits: int


def _expected_source_bindings(
    answer_plane: Any,
    adapter: _AnswerArmAdapter,
    runtime: Mapping[str, Any],
) -> dict[str, str]:
    raw = runtime.get("source_artifacts")
    _require(type(raw) is list, "query answer runtime sources must be an array")
    roles: list[str] = []
    bindings: dict[str, str] = {}
    prefix = f"{adapter.arm_label}:"
    for item in raw:
        _require(
            type(item) is dict and set(item) == {"role", "sha256"},
            "query answer runtime source schema changed",
        )
        role = require_text(item["role"], "query answer runtime source role")
        digest = require_sha256(
            item["sha256"], f"query answer runtime source {role}"
        )
        _require(role.startswith(prefix), "query answer runtime source arm changed")
        short = role[len(prefix) :]
        roles.append(short)
        bindings[short] = digest
    _require(
        tuple(roles) == adapter.source_roles and len(bindings) == len(roles),
        "query answer runtime source-artifact envelope changed",
    )
    expected = {
        "sealed_retrieval": answer_plane.retrieval_sha256,
        "query_adapter": answer_plane.adapter_population_id,
        adapter.parent_answer_source_role: answer_plane.parent_plane.run_sha256,
        adapter.parent_runtime_source_role: (
            answer_plane.parent_plane.runtime_ledger_sha256
        ),
        "answer_run": answer_plane.run_sha256,
    }
    if adapter is _FACT_ADAPTER:
        expected.update(
            {
                "query_fact_compression": answer_plane.compression_sha256,
                "query_fact_compression_runtime_ledger": (
                    answer_plane.compression_runtime_ledger_sha256
                ),
            }
        )
    if adapter is _EVIDENCE_SOLVER_V2_ADAPTER:
        expected.update(
            {
                "map_run": answer_plane.map_plane.run_sha256,
                "map_runtime_ledger": (
                    answer_plane.map_plane.runtime_ledger_sha256
                ),
            }
        )
    if adapter in _ADAPTIVE_ADAPTERS:
        expected.update(
            adaptive_evidence_solver_judge_adapter.expected_source_bindings(
                answer_plane
            )
        )
    for role, expected_sha in expected.items():
        _require(
            bindings.get(role) == expected_sha,
            f"query answer runtime source binding changed: {role}",
        )
    return bindings


def _validate_answer_plane(
    answer_plane: object,
    *,
    expected_question_count: int,
) -> tuple[_AnswerArmAdapter, dict[str, Any]]:
    """Fix the changed-call projection before gold or verdicts are opened."""

    adapter = _adapter_for_plane(answer_plane)
    plane = answer_plane
    if adapter in _ADAPTIVE_ADAPTERS:
        profile = (
            adaptive_evidence_solver_judge_adapter.validate_adaptive_judge_plane(
                plane
            )
        )
        _require(
            profile.kind == adapter.kind
            and profile.arm_label == adapter.arm_label,
            "adaptive judge adapter profile changed",
        )
    _require(
        type(expected_question_count) is int and expected_question_count > 0,
        "expected question count must be a positive exact integer",
    )
    for value, label in (
        (plane.run_sha256, "query answer run SHA-256"),
        (plane.replay_sha256, "query answer replay SHA-256"),
        (plane.runtime_ledger_sha256, "query answer runtime-ledger SHA-256"),
        (plane.parent_answer_run_sha256, "query answer parent run SHA-256"),
        (plane.adapter_population_id, "query answer adapter population ID"),
        (plane.retrieval_sha256, "query answer retrieval SHA-256"),
        (plane.snapshot_id, "query answer snapshot ID"),
    ):
        require_sha256(value, label)
    if adapter is _FACT_ADAPTER:
        require_sha256(plane.compression_sha256, "query fact compression SHA-256")
        require_sha256(
            plane.compression_runtime_ledger_sha256,
            "query fact compression runtime-ledger SHA-256",
        )
    _require(
        plane.run_sha256 == plane.replay_sha256,
        "query answer run and replay differ",
    )
    parent = plane.parent_plane
    if adapter in (
        _OPERATOR_ADAPTER,
        _EVIDENCE_SOLVER_V2_ADAPTER,
        *_ADAPTIVE_ADAPTERS,
    ):
        parent_adapter, _parent_projection = _validate_answer_plane(
            parent,
            expected_question_count=expected_question_count,
        )
        _require(
            parent_adapter is _PAYLOAD_ADAPTER,
            "derived answer plane lost its exact direct query-payload parent",
        )
    else:
        _require(
            type(parent) is live.VerifiedS0V2AnswerPlane,
            "query answer plane lost its exact S0-v2 parent",
        )
        judging._validate_preverified_answer_plane(
            parent,
            profile=judging._V2_JUDGE_PROFILE,
        )
    _require(
        plane.parent_answer_run_sha256 == parent.run_sha256
        and parent.run_sha256 == parent.replay_sha256,
        "query answer parent run binding changed",
    )
    # Query construction deliberately has its own cumulative snapshot.  The
    # parent relationship is fixed by the exact parent run, population rows,
    # and per-row prediction hashes rather than by snapshot equality.
    _require(
        type(plane.rows) is tuple
        and len(plane.rows) == len(parent.rows) == expected_question_count,
        "query answer judge population size changed",
    )

    runtime = _runtime_json(plane.runtime_ledger)
    runtime_sha = sha256(canonical_json_bytes(runtime)).hexdigest()
    _require(
        runtime_sha == plane.runtime_ledger_sha256,
        "query answer runtime-ledger artifact SHA-256 changed",
    )
    ledger_identity, answer_row_ids = _validated_runtime_ledger(runtime)
    require_sha256(ledger_identity, "query answer runtime-ledger identity")
    _require(
        runtime.get("snapshot_id") == plane.snapshot_id
        and runtime.get("plan_id") == adapter.answer_plan_id,
        "query answer runtime envelope changed",
    )
    _require(
        runtime.get("row_count") == expected_question_count * 2
        and runtime.get("question_count") == expected_question_count,
        "query answer runtime population changed",
    )
    source_bindings = _expected_source_bindings(plane, adapter, runtime)
    raw_answers = tuple(
        row for row in runtime["rows"] if row.get("event_type") == "answer_observation"
    )
    _require(
        len(raw_answers) == len(plane.rows)
        and len(runtime["rows"]) == len(plane.rows) * 2,
        "query answer runtime must contain one stage and one answer per question",
    )
    _require(
        answer_row_ids == tuple(row.runtime_row_id for row in plane.rows),
        "query answer runtime row order changed",
    )

    projection_rows: list[dict[str, Any]] = []
    ordinals: list[int] = []
    question_ids: list[str] = []
    for child, parent_row, raw in zip(
        plane.rows,
        parent.rows,
        raw_answers,
        strict=True,
    ):
        _require(
            type(child) is adapter.row_type,
            "query answer rows must have their exact verified arm type",
        )
        _require(
            type(child.ordinal) is int and child.ordinal >= 0,
            "query answer ordinal is invalid",
        )
        require_text(child.question_id, "query answer question ID")
        require_text(child.route_id, "query answer route ID")
        require_text(child.prediction_source, "query answer prediction source")
        for value, label in (
            (child.question_sha256, "query answer question SHA-256"),
            (child.dated_question_sha256, "query answer dated-question SHA-256"),
            (child.prediction_sha256, "query answer prediction SHA-256"),
            (child.parent_prediction_sha256, "query answer parent prediction SHA-256"),
            (child.source_row_sha256, "query answer source row SHA-256"),
            (child.runtime_row_id, "query answer runtime row ID"),
        ):
            require_sha256(value, label)
        _require(
            type(child.prediction) is str
            and bool(child.prediction)
            and quote_sha256(child.prediction) == child.prediction_sha256,
            f"query answer prediction changed at ordinal {child.ordinal}",
        )
        _require(
            child.ordinal == parent_row.ordinal
            and child.question_id == parent_row.question_id
            and child.question_sha256 == parent_row.question_sha256
            and child.dated_question_sha256 == parent_row.dated_question_sha256
            and child.parent_prediction_sha256 == parent_row.prediction_sha256,
            f"query answer parent row binding changed at ordinal {child.ordinal}",
        )
        changed = child.prediction_sha256 != parent_row.prediction_sha256
        _require(
            type(child.changed_from_parent) is bool
            and child.changed_from_parent == changed,
            f"query answer change flag changed at ordinal {child.ordinal}",
        )
        _require(
            raw.get("row_id") == child.runtime_row_id
            and raw.get("ordinal") == child.ordinal
            and raw.get("question_id") == child.question_id
            and raw.get("question_sha256") == child.question_sha256
            and raw.get("arm_label") == adapter.arm_label
            and raw.get("parent_arm_label") == adapter.parent_arm_label
            and raw.get("stage_id") == adapter.answer_stage_id
            and raw.get("parent_stage_id") == adapter.parent_stage_id
            and raw.get("renderer_id") == adapter.renderer_id
            and raw.get("prediction") == child.prediction
            and raw.get("prediction_sha256") == child.prediction_sha256
            and raw.get("changed_from_parent") == changed
            and raw.get("source_row_sha256") == child.source_row_sha256,
            f"query answer/runtime binding changed at ordinal {child.ordinal}",
        )
        require_sha256(
            str(raw.get("packet_sha256")),
            "query answer runtime packet_sha256",
        )
        if (
            adapter
            in (
                _OPERATOR_ADAPTER,
                _EVIDENCE_SOLVER_V2_ADAPTER,
                *_ADAPTIVE_ADAPTERS,
            )
            and raw.get("provider_calls") == 0
        ):
            _require(
                raw.get("prompt_id") is None
                and raw.get("prompt_messages_sha256") is None,
                f"query operator zero-call prompt provenance changed at ordinal {child.ordinal}",
            )
        else:
            for key in ("prompt_id", "prompt_messages_sha256"):
                require_sha256(str(raw.get(key)), f"query answer runtime {key}")
        if child.prediction_source == adapter.fallback_prediction_source:
            _require(
                not changed
                and child.prediction == parent_row.prediction
                and raw.get("provider_calls") in adapter.fallback_provider_calls
                and raw.get("mechanism_id") == adapter.fallback_mechanism_id,
                f"query answer fallback changed at ordinal {child.ordinal}",
            )
        elif child.prediction_source == adapter.terra_prediction_source:
            _require(
                raw.get("provider_calls") == 1
                and raw.get("mechanism_id") == adapter.terra_mechanism_id,
                f"query answer Terra binding changed at ordinal {child.ordinal}",
            )
        elif (
            adapter.keep_parent_prediction_source is not None
            and child.prediction_source == adapter.keep_parent_prediction_source
        ):
            _require(
                not changed
                and child.prediction == parent_row.prediction
                and raw.get("provider_calls") == 1
                and raw.get("mechanism_id")
                == adapter.keep_parent_mechanism_id,
                f"query answer Terra keep-parent binding changed at ordinal {child.ordinal}",
            )
        else:
            raise MatchedEvalContractError(
                f"unknown query answer prediction source at ordinal {child.ordinal}"
            )
        if adapter is _OPERATOR_ADAPTER:
            require_sha256(
                child.plan_row_receipt_sha256,
                "query operator plan-row receipt SHA-256",
            )
            _require(
                type(child.operator_trace_valid) is bool
                and type(child.operator_trace_status) is str
                and bool(child.operator_trace_status),
                f"query operator trace provenance changed at ordinal {child.ordinal}",
            )
            provider_calls = raw.get("provider_calls")
            if provider_calls == 1:
                for value, label in (
                    (child.call_key_sha256, "query operator call-key SHA-256"),
                    (
                        child.request_journal_sha256,
                        "query operator request-journal SHA-256",
                    ),
                    (
                        child.response_journal_sha256,
                        "query operator response-journal SHA-256",
                    ),
                    (
                        child.operator_trace_receipt_sha256,
                        "query operator trace-receipt SHA-256",
                    ),
                ):
                    require_sha256(value, label)
            else:
                _require(
                    provider_calls == 0
                    and child.call_key_sha256 is None
                    and child.request_journal_sha256 is None
                    and child.response_journal_sha256 is None
                    and child.operator_trace_receipt_sha256 is None,
                    f"query operator zero-call provenance changed at ordinal {child.ordinal}",
                )
        if adapter is _EVIDENCE_SOLVER_V2_ADAPTER:
            for value, label in (
                (child.map_parse_receipt_sha256, "V2 map parse receipt SHA-256"),
                (child.map_source_row_sha256, "V2 map source-row SHA-256"),
                (
                    child.solver_plan_row_receipt_sha256,
                    "V2 solver plan-row receipt SHA-256",
                ),
            ):
                require_sha256(value, label)
            _require(
                type(child.solver_valid) is bool
                and child.solver_decision
                in {"replace", "keep_parent", "insufficient", "not_submitted"}
                and type(child.solver_used_item_ids) is tuple
                and len(set(child.solver_used_item_ids))
                == len(child.solver_used_item_ids),
                f"V2 solver decision provenance changed at ordinal {child.ordinal}",
            )
            _require(
                raw.get("provider_calls") in (0, 1),
                f"V2 solver provider-call count changed at ordinal {child.ordinal}",
            )
            if raw.get("provider_calls") == 1:
                for value, label in (
                    (child.call_key_sha256, "V2 solver call-key SHA-256"),
                    (
                        child.request_journal_sha256,
                        "V2 solver request-journal SHA-256",
                    ),
                    (
                        child.response_journal_sha256,
                        "V2 solver response-journal SHA-256",
                    ),
                    (
                        child.solver_parse_receipt_sha256,
                        "V2 solver parse-receipt SHA-256",
                    ),
                ):
                    require_sha256(value, label)
            else:
                _require(
                    child.call_key_sha256 is None
                    and child.request_journal_sha256 is None
                    and child.response_journal_sha256 is None
                    and child.solver_parse_receipt_sha256 is None,
                    f"V2 solver zero-call provenance changed at ordinal {child.ordinal}",
                )
        ordinals.append(child.ordinal)
        question_ids.append(child.question_id)
        projection_rows.append(
            {
                "changed_from_parent": changed,
                "dated_question_sha256": child.dated_question_sha256,
                "ordinal": child.ordinal,
                "parent_prediction_sha256": parent_row.prediction_sha256,
                "prediction_sha256": child.prediction_sha256,
                "question_id": child.question_id,
                "question_sha256": child.question_sha256,
                "route_id": child.route_id,
                "runtime_row_id": child.runtime_row_id,
                "source_row_sha256": child.source_row_sha256,
            }
        )
    _require(
        tuple(ordinals) == tuple(sorted(set(ordinals))),
        "query answer row order changed",
    )
    _require(
        len(set(question_ids)) == len(question_ids),
        "query answer question IDs must be unique",
    )
    _require(
        sum(bool(row["changed_from_parent"]) for row in projection_rows)
        == len(plane.changed_rows),
        "query answer changed-row projection changed",
    )
    body: dict[str, Any] = {
        "adapter_population_id": plane.adapter_population_id,
        "answer_run_sha256": plane.run_sha256,
        "arm_kind": adapter.kind,
        "arm_label": adapter.arm_label,
        "format": CHANGE_PROJECTION_FORMAT,
        "parent_answer_run_sha256": parent.run_sha256,
        "population_identity_sha256": (
            _base_population_plane(plane).population_identity_sha256
        ),
        "rows": projection_rows,
        "runtime_ledger_identity_sha256": ledger_identity,
        "source_bindings": source_bindings,
    }
    body["change_projection_sha256"] = identity_sha256(body)
    return adapter, body


def _base_population_plane(answer_plane: Any) -> live.VerifiedS0V2AnswerPlane:
    parent = answer_plane.parent_plane
    if type(answer_plane) in (
        query_operator_refinement_live.VerifiedQueryOperatorRefinementPlane,
        query_evidence_map_solver_v2_live.VerifiedEvidenceSolverPlane,
        adaptive_evidence_solver_judge_adapter.VerifiedAdaptiveEvidenceSolverDirectJudgePlane,
        adaptive_evidence_solver_judge_adapter.VerifiedAdaptiveEvidenceSolverPartitionJudgePlane,
        adaptive_evidence_solver_judge_adapter.VerifiedAdaptiveEvidenceSolverGuidedJudgePlane,
        adaptive_evidence_solver_judge_adapter.VerifiedAdaptiveEvidenceSolverDirectGuidedJudgePlane,
    ):
        parent = parent.parent_plane
    _require(
        type(parent) is live.VerifiedS0V2AnswerPlane,
        "query answer base population lost its exact S0-v2 plane",
    )
    return parent


def _load_gold(
    *,
    dataset_path: str | Path,
    split_path: str | Path,
    answer_plane: Any,
) -> tuple[tuple[judging._GoldRow, ...], str]:
    return judging._load_gold(
        dataset_path=dataset_path,
        split_path=split_path,
        answer_plane=answer_plane,
    )


def _prompt_plan(
    answer_plane: Any,
    gold_rows: Sequence[judging._GoldRow],
) -> tuple[tuple[judging._JudgePromptRow, ...], FastPromptPopulation | None]:
    pending: list[tuple[int, tuple[dict[str, str], ...], str]] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    for source, gold in zip(answer_plane.rows, gold_rows, strict=True):
        if not source.changed_from_parent:
            continue
        messages = tuple(
            dict(message)
            for message in build_judge_prompt(
                gold.question,
                gold.reference,
                source.prediction,
            )
        )
        _require(
            messages
            == tuple(
                dict(message)
                for message in build_judge_prompt(
                    gold.question,
                    gold.reference,
                    source.prediction,
                )
            ),
            "query answer judge prompt contains noncanonical fields",
        )
        pending.append(
            (
                source.ordinal,
                messages,
                route_question(gold.dated_question).style.value,
            )
        )
        prompts.append(messages)
    if not prompts:
        return (), None
    population = preflight_fast_completion_prompts(
        prompts,
        max_prompt_tokens=judging.DEFAULT_MAX_JUDGE_PROMPT_TOKENS,
    )
    _require(
        population.logical_prompt_count
        == population.unique_prompt_count
        == len(prompts),
        "query answer judge prompts must be unique",
    )
    rows = tuple(
        judging._JudgePromptRow(
            answer_ordinal=ordinal,
            messages=messages,
            messages_sha256=receipt.messages_sha256,
            prompt_token_proxy=receipt.prompt_token_proxy,
            demand_class=demand_class,
        )
        for (ordinal, messages, demand_class), receipt in zip(
            pending,
            population.ordered_rows,
            strict=True,
        )
    )
    return rows, population


def _load_query_answer_parent_judge(
    *,
    parent: query_payload_live.VerifiedQueryPayloadAnswerPlane,
    gold_rows: tuple[judging._GoldRow, ...],
    gold_population_sha256: str,
    parent_judge_root: str | Path,
    expected_parent_judge_sha256: str,
    expected_parent_score_ledger_sha256: str,
) -> closure_judging._VerifiedParentJudge:
    """Verify a sealed query-answer judge as an immediate parent.

    The exact expected judge and score hashes are trust anchors.  The parent
    preflight and score are additionally reconstructed from the replayed
    direct answer plane, while each exported outcome is rebound to its answer
    and gold row.  No parent provider journal is replayed.
    """

    expected_judge = require_sha256(
        expected_parent_judge_sha256,
        "expected query-answer parent judge SHA-256",
    )
    expected_score = require_sha256(
        expected_parent_score_ledger_sha256,
        "expected query-answer parent score-ledger SHA-256",
    )
    root = Path(parent_judge_root)
    preflight = read_sealed_json(root / JUDGE_PREFLIGHT_NAME)
    judge = read_sealed_json(root / JUDGE_NAME)
    judge_replay = read_sealed_json(root / JUDGE_REPLAY_NAME)
    score = read_sealed_json(root / SCORE_LEDGER_NAME)
    score_replay = read_sealed_json(root / SCORE_LEDGER_REPLAY_NAME)
    _require(
        judge.sha256 == judge_replay.sha256 == expected_judge
        and judge.payload == judge_replay.payload,
        "sealed query-answer parent judge/replay differ",
    )
    _require(
        score.sha256 == score_replay.sha256 == expected_score
        and score.payload == score_replay.payload,
        "sealed query-answer parent score-ledger/replay differ",
    )

    adapter, change_projection = _validate_answer_plane(
        parent,
        expected_question_count=len(gold_rows),
    )
    _require(
        adapter is _PAYLOAD_ADAPTER,
        "operator parent judge requires the exact direct query-payload plane",
    )
    prompt_rows, prompt_population = _prompt_plan(parent, gold_rows)
    parent_meta = closure_judging._VerifiedParentJudge(
        preflight_sha256=require_sha256(
            str(preflight.payload.get("parent_judge_preflight_sha256")),
            "query-answer parent-of-parent preflight SHA-256",
        ),
        judge_sha256=require_sha256(
            str(preflight.payload.get("parent_judge_sha256")),
            "query-answer parent-of-parent judge SHA-256",
        ),
        score_ledger_sha256=require_sha256(
            str(preflight.payload.get("parent_score_ledger_sha256")),
            "query-answer parent-of-parent score-ledger SHA-256",
        ),
        outcomes=(),
    )
    parent_plan = _QueryAnswerJudgePlan(
        answer_plane=parent,
        adapter=adapter,
        gold_rows=gold_rows,
        gold_population_sha256=gold_population_sha256,
        change_projection=change_projection,
        prompt_rows=prompt_rows,
        prompt_population=prompt_population,
        parent_judge=parent_meta,
    )
    _require(
        preflight.payload == _preflight_payload(parent_plan),
        "sealed query-answer parent judge preflight changed",
    )

    payload = judge.payload
    _require(
        payload.get("format") == JUDGE_FORMAT
        and payload.get("arm_kind") == adapter.kind
        and payload.get("arm_label") == adapter.arm_label
        and payload.get("arm_plan_id") == adapter.arm_plan_id
        and payload.get("answer_plan_id") == adapter.answer_plan_id
        and payload.get("judge_plan_id") == adapter.judge_plan_id
        and payload.get("answer_run_sha256") == parent.run_sha256
        and payload.get("runtime_ledger_sha256")
        == parent.runtime_ledger_sha256
        and payload.get("gold_population_sha256") == gold_population_sha256
        and payload.get("preflight_artifact_sha256") == preflight.sha256
        and payload.get("parent_judge_sha256") == parent_meta.judge_sha256
        and payload.get("parent_score_ledger_sha256")
        == parent_meta.score_ledger_sha256
        and payload.get("prompt_content_contract")
        == "question_reference_prediction_only"
        and payload.get("question_count") == len(parent.rows),
        "sealed query-answer parent judge envelope changed",
    )
    raw_rows = payload.get("questions")
    _require(
        type(raw_rows) is list and len(raw_rows) == len(parent.rows),
        "sealed query-answer parent judge population changed",
    )
    prompts_by_ordinal = {row.answer_ordinal: row for row in prompt_rows}
    outcomes: list[closure_judging._ParentOutcome] = []
    for answer, gold, raw in zip(parent.rows, gold_rows, raw_rows, strict=True):
        _require(
            type(raw) is dict,
            f"sealed query-answer parent judge row changed at {answer.ordinal}",
        )
        unsigned = dict(raw)
        row_sha = unsigned.pop("judge_row_sha256", None)
        _require(
            row_sha == identity_sha256(unsigned),
            f"sealed query-answer parent judge row seal changed at {answer.ordinal}",
        )
        correct = raw.get("correct")
        baseline = raw.get("baseline_correct")
        verdict_sha = require_sha256(
            str(raw.get("verdict_sha256")),
            "query-answer parent verdict SHA-256",
        )
        _require(
            type(correct) is bool
            and type(baseline) is bool
            and raw.get("ordinal") == answer.ordinal
            and raw.get("question_id") == answer.question_id
            and raw.get("question_sha256") == answer.question_sha256
            and raw.get("dated_question_sha256") == answer.dated_question_sha256
            and raw.get("prediction_sha256") == answer.prediction_sha256
            and raw.get("parent_prediction_sha256")
            == answer.parent_prediction_sha256
            and raw.get("changed_from_parent") == answer.changed_from_parent
            and raw.get("runtime_row_id") == answer.runtime_row_id
            and raw.get("route_id") == answer.route_id
            and raw.get("reference_sha256") == gold.reference_sha256
            and raw.get("category") == gold.category
            and raw.get("normalized_exact_match")
            == exact_match(answer.prediction, gold.reference)
            and type(raw.get("normalized_f1")) in (int, float)
            and float(raw.get("normalized_f1"))
            == f1_score(answer.prediction, gold.reference)
            and raw.get("demand_class")
            == route_question(gold.dated_question).style.value,
            f"sealed query-answer parent row binding changed at {answer.ordinal}",
        )
        if answer.changed_from_parent:
            prompt = prompts_by_ordinal[answer.ordinal]
            output = raw.get("judge_output")
            _require(
                type(output) is str
                and bool(output)
                and parse_binary_judge_verdict(output) == correct
                and raw.get("judge_output_sha256") == quote_sha256(output)
                and verdict_sha == raw.get("judge_output_sha256")
                and raw.get("verdict_source") == "new_sol_judge"
                and raw.get("judge_messages_sha256") == prompt.messages_sha256
                and raw.get("judge_prompt_token_proxy")
                == prompt.prompt_token_proxy
                and raw.get("demand_class") == prompt.demand_class,
                f"fresh query-answer parent verdict changed at {answer.ordinal}",
            )
            for key in (
                "call_key_sha256",
                "request_journal_sha256",
                "response_journal_sha256",
            ):
                require_sha256(str(raw.get(key)), f"query-answer parent {key}")
        else:
            _require(
                correct == baseline
                and verdict_sha == raw.get("parent_judge_verdict_sha256")
                and raw.get("verdict_source")
                == _PAYLOAD_ADAPTER.inherited_verdict_source
                and raw.get("judge_output") is None
                and raw.get("judge_output_sha256") is None
                and raw.get("judge_messages_sha256") is None
                and raw.get("judge_prompt_token_proxy") is None
                and raw.get("call_key_sha256") is None
                and raw.get("request_journal_sha256") is None
                and raw.get("response_journal_sha256") is None,
                f"inherited query-answer parent verdict changed at {answer.ordinal}",
            )
        outcomes.append(
            closure_judging._ParentOutcome(
                ordinal=answer.ordinal,
                correct=correct,
                judge_row_sha256=str(row_sha),
                judge_verdict_sha256=verdict_sha,
                demand_class=require_text(
                    raw.get("demand_class"),
                    "query-answer parent demand class",
                ),
            )
        )
    _require(
        payload.get("aggregate", {}).get("correct")
        == sum(outcome.correct for outcome in outcomes),
        "sealed query-answer parent aggregate changed",
    )
    expected_score_payload = _score_payload(
        parent_plan,
        payload,
        judge_sha256=judge.sha256,
    )
    _require(
        score.payload == expected_score_payload,
        "sealed query-answer parent score ledger changed",
    )
    return closure_judging._VerifiedParentJudge(
        preflight_sha256=preflight.sha256,
        judge_sha256=judge.sha256,
        score_ledger_sha256=score.sha256,
        outcomes=tuple(outcomes),
    )


def _build_plan(
    *,
    answer_plane: object,
    dataset_path: str | Path,
    split_path: str | Path,
    parent_judge_root: str | Path,
    expected_parent_judge_sha256: str,
    expected_parent_score_ledger_sha256: str,
    expected_question_count: int,
) -> _QueryAnswerJudgePlan:
    adapter, change_projection = _validate_answer_plane(
        answer_plane,
        expected_question_count=expected_question_count,
    )
    plane = answer_plane
    gold_rows, gold_population_sha256 = _load_gold(
        dataset_path=dataset_path,
        split_path=split_path,
        answer_plane=plane,
    )
    _require(
        len(gold_rows) == len(plane.rows),
        "query answer gold population changed",
    )
    prompt_rows, prompt_population = _prompt_plan(plane, gold_rows)
    if adapter in (
        _OPERATOR_ADAPTER,
        _EVIDENCE_SOLVER_V2_ADAPTER,
        *_ADAPTIVE_ADAPTERS,
    ):
        parent_judge = _load_query_answer_parent_judge(
            parent=plane.parent_plane,
            gold_rows=gold_rows,
            gold_population_sha256=gold_population_sha256,
            parent_judge_root=parent_judge_root,
            expected_parent_judge_sha256=expected_parent_judge_sha256,
            expected_parent_score_ledger_sha256=(
                expected_parent_score_ledger_sha256
            ),
        )
    else:
        parent_judge = closure_judging._load_parent_judge(
            parent=plane.parent_plane,
            gold_rows=gold_rows,
            gold_population_sha256=gold_population_sha256,
            parent_judge_root=parent_judge_root,
            expected_parent_judge_sha256=expected_parent_judge_sha256,
            expected_parent_score_ledger_sha256=(
                expected_parent_score_ledger_sha256
            ),
        )
    return _QueryAnswerJudgePlan(
        answer_plane=plane,
        adapter=adapter,
        gold_rows=gold_rows,
        gold_population_sha256=gold_population_sha256,
        change_projection=change_projection,
        prompt_rows=prompt_rows,
        prompt_population=prompt_population,
        parent_judge=parent_judge,
    )


def _empty_prompt_population() -> dict[str, Any]:
    body: dict[str, Any] = {
        "format": EMPTY_PROMPT_POPULATION_FORMAT,
        "logical_prompt_count": 0,
        "max_prompt_token_proxy": judging.DEFAULT_MAX_JUDGE_PROMPT_TOKENS,
        "ordered_rows": [],
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "unique_prompt_count": 0,
    }
    body["prompt_population_sha256"] = identity_sha256(body)
    return body


def _prompt_population_projection(plan: _QueryAnswerJudgePlan) -> dict[str, Any]:
    if plan.prompt_population is None:
        return _empty_prompt_population()
    return plan.prompt_population.model_dump()


def _preflight_payload(plan: _QueryAnswerJudgePlan) -> dict[str, Any]:
    population = _prompt_population_projection(plan)
    plane = plan.answer_plane
    base_population = _base_population_plane(plane)
    return {
        "adapter_population_id": plane.adapter_population_id,
        "answer_plan_id": plan.adapter.answer_plan_id,
        "answer_run_sha256": plane.run_sha256,
        "arm_kind": plan.adapter.kind,
        "arm_label": plan.adapter.arm_label,
        "arm_plan_id": plan.adapter.arm_plan_id,
        "change_projection": dict(plan.change_projection),
        "changed_prediction_count": plan.changed_count,
        "format": JUDGE_PREFLIGHT_FORMAT,
        "gold_loaded_posthoc": True,
        "gold_population_sha256": plan.gold_population_sha256,
        "inherited_prediction_count": len(plane.rows) - plan.changed_count,
        "judge_model": judging.DEFAULT_SOL_CALLER_MODEL,
        "judge_plan_id": plan.adapter.judge_plan_id,
        "logical_prompt_count": plan.changed_count,
        "matched_population_id": base_population.matched_population_id,
        "maximum_prompt_token_proxy": max(
            (row.prompt_token_proxy for row in plan.prompt_rows),
            default=0,
        ),
        "new_provider_calls": 0,
        "parent_answer_run_sha256": plane.parent_plane.run_sha256,
        "parent_judge_preflight_sha256": plan.parent_judge.preflight_sha256,
        "parent_judge_sha256": plan.parent_judge.judge_sha256,
        "parent_runtime_ledger_sha256": (
            plane.parent_plane.runtime_ledger_sha256
        ),
        "parent_score_ledger_sha256": plan.parent_judge.score_ledger_sha256,
        "population_identity_sha256": base_population.population_identity_sha256,
        "prompt_content_contract": "question_reference_prediction_only",
        "prompt_population": population,
        "prompt_population_sha256": population["prompt_population_sha256"],
        "provider_prompts": [list(row.messages) for row in plan.prompt_rows],
        "renderer_id": plan.adapter.renderer_id,
        "required_authorized_provider_calls": plan.required_calls,
        "retained_transformer_token_state_bytes": 0,
        "retrieval_sha256": plane.retrieval_sha256,
        "runtime_ledger_sha256": plane.runtime_ledger_sha256,
        "snapshot_id": plane.snapshot_id,
        "source_bindings": dict(plan.change_projection["source_bindings"]),
        "unique_prompt_count": plan.required_calls,
    }


def preflight_query_answer_changed_only_judge(
    *,
    answer_plane: object,
    dataset_path: str | Path,
    split_path: str | Path,
    parent_judge_root: str | Path,
    expected_parent_judge_sha256: str,
    expected_parent_score_ledger_sha256: str,
    output_root: str | Path,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> SealedArtifact:
    """Seal the changed-only Sol population without executing a call."""

    plan = _build_plan(
        answer_plane=answer_plane,
        dataset_path=dataset_path,
        split_path=split_path,
        parent_judge_root=parent_judge_root,
        expected_parent_judge_sha256=expected_parent_judge_sha256,
        expected_parent_score_ledger_sha256=(
            expected_parent_score_ledger_sha256
        ),
        expected_question_count=expected_question_count,
    )
    artifact, _created = publish_sealed_json(
        Path(output_root) / JUDGE_PREFLIGHT_NAME,
        _preflight_payload(plan),
    )
    return artifact


def _messages_from_preflight(
    payload: Mapping[str, Any],
    *,
    required_calls: int,
) -> tuple[tuple[dict[str, str], ...], ...]:
    raw_prompts = payload.get("provider_prompts")
    _require(
        type(raw_prompts) is list and len(raw_prompts) == required_calls,
        "sealed query-answer judge prompts changed",
    )
    prompts: list[tuple[dict[str, str], ...]] = []
    for prompt_index, raw_prompt in enumerate(raw_prompts):
        _require(
            type(raw_prompt) is list and bool(raw_prompt),
            f"sealed query-answer judge prompt {prompt_index} changed",
        )
        messages: list[dict[str, str]] = []
        for raw_message in raw_prompt:
            _require(
                type(raw_message) is dict
                and set(raw_message) == {"role", "content"}
                and type(raw_message.get("role")) is str
                and type(raw_message.get("content")) is str,
                f"sealed query-answer judge message {prompt_index} changed",
            )
            messages.append(
                {
                    "role": str(raw_message["role"]),
                    "content": str(raw_message["content"]),
                }
            )
        prompts.append(tuple(messages))
    return tuple(prompts)


def load_query_answer_judge_provider_population(
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
) -> SealedQueryAnswerJudgeProviderPopulation:
    """Load only the sealed Sol prompt population for network execution."""

    expected = require_sha256(
        expected_preflight_sha256,
        "expected query-answer judge preflight",
    )
    output = Path(output_root)
    artifact = read_sealed_json(output / JUDGE_PREFLIGHT_NAME)
    payload = artifact.payload
    _require(
        artifact.sha256 == expected,
        "query-answer judge preflight SHA-256 changed",
    )
    adapter = _adapter_for_label(str(payload.get("arm_label")))
    required = payload.get("required_authorized_provider_calls")
    _require(
        payload.get("format") == JUDGE_PREFLIGHT_FORMAT
        and payload.get("arm_kind") == adapter.kind
        and payload.get("answer_plan_id") == adapter.answer_plan_id
        and payload.get("arm_plan_id") == adapter.arm_plan_id
        and payload.get("judge_plan_id") == adapter.judge_plan_id
        and payload.get("judge_model") == judging.DEFAULT_SOL_CALLER_MODEL
        and payload.get("prompt_content_contract")
        == "question_reference_prediction_only"
        and payload.get("gold_loaded_posthoc") is True
        and payload.get("retained_transformer_token_state_bytes") == 0,
        "query-answer judge provider preflight envelope changed",
    )
    _require(
        type(required) is int
        and required >= 0
        and payload.get("changed_prediction_count") == required
        and payload.get("logical_prompt_count") == required
        and payload.get("unique_prompt_count") == required,
        "query-answer judge provider call population changed",
    )
    prompts = _messages_from_preflight(payload, required_calls=required)
    prompt_population = (
        preflight_fast_completion_prompts(
            prompts,
            max_prompt_tokens=judging.DEFAULT_MAX_JUDGE_PROMPT_TOKENS,
        )
        if prompts
        else None
    )
    observed = (
        _empty_prompt_population()
        if prompt_population is None
        else prompt_population.model_dump()
    )
    _require(
        payload.get("prompt_population") == observed
        and payload.get("prompt_population_sha256")
        == observed["prompt_population_sha256"],
        "query-answer judge prompt population no longer matches its messages",
    )
    return SealedQueryAnswerJudgeProviderPopulation(
        preflight_artifact=artifact,
        output_root=output,
        adapter=adapter,
        prompts=prompts,
        prompt_population=prompt_population,
        required_calls=required,
    )


def _provider_runtime(
    population: SealedQueryAnswerJudgeProviderPopulation,
    *,
    client: Any | None,
    max_concurrency: int,
    gateway_url: str,
) -> FastCompletionRuntime:
    _require(population.required_calls > 0, "empty judge population has no runtime")
    payload = population.preflight_artifact.payload
    return FastCompletionRuntime(
        checkpoint_dir=population.output_root / JUDGE_CHECKPOINT_DIR_NAME,
        prompt_population=population.prompts,
        model=judging.DEFAULT_SOL_GATEWAY_MODEL,
        client=client,
        max_prompt_tokens=judging.DEFAULT_MAX_JUDGE_PROMPT_TOKENS,
        max_new_tokens=JUDGE_MAX_TOKENS,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "answer_run_sha256": payload["answer_run_sha256"],
            "arm_kind": population.adapter.kind,
            "arm_label": population.adapter.arm_label,
            "authorized_unique_calls": population.required_calls,
            "change_projection_sha256": payload["change_projection"][
                "change_projection_sha256"
            ],
            "gateway_url": gateway_url,
            "gold_population_sha256": payload["gold_population_sha256"],
            "judge_plan_id": population.adapter.judge_plan_id,
            "parent_answer_run_sha256": payload["parent_answer_run_sha256"],
            "parent_judge_sha256": payload["parent_judge_sha256"],
            "parent_score_ledger_sha256": payload[
                "parent_score_ledger_sha256"
            ],
            "preflight_artifact_sha256": population.preflight_artifact.sha256,
            "prompt_content_contract": "question_reference_prediction_only",
            "runtime_ledger_sha256": payload["runtime_ledger_sha256"],
        },
    )


def run_sealed_query_answer_judge_provider(
    population: SealedQueryAnswerJudgeProviderPopulation,
    *,
    enable_provider: bool,
    authorized_provider_calls: int,
    client: Any | None,
    max_concurrency: int = 4,
    gateway_url: str = live.DEFAULT_GATEWAY_URL,
) -> QueryAnswerJudgeProviderResult:
    """Fill immutable Sol journals; do not publish judgments or scores."""

    if type(population) is not SealedQueryAnswerJudgeProviderPopulation:
        raise TypeError(
            "population must be an exact SealedQueryAnswerJudgeProviderPopulation"
        )
    _require(type(enable_provider) is bool, "provider enablement must be exact bool")
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == population.required_calls,
        "authorized query-answer judge provider calls must exactly equal "
        f"{population.required_calls}",
    )
    _require(
        enable_provider == bool(population.required_calls),
        "provider enablement must match the sealed judge population",
    )
    if not population.required_calls:
        _require(client is None, "empty judge population forbids a client")
        return QueryAnswerJudgeProviderResult(
            population.preflight_artifact,
            None,
            0,
            0,
        )
    _require(client is not None, "changed query answers require a Sol client")
    runtime = _provider_runtime(
        population,
        client=client,
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    _require(
        batch.usage.physical_calls + batch.usage.checkpoint_hits
        == population.required_calls,
        "query-answer Sol journal population changed",
    )
    return QueryAnswerJudgeProviderResult(
        population.preflight_artifact,
        batch,
        batch.usage.physical_calls,
        batch.usage.checkpoint_hits,
    )


def load_query_answer_judge_journals(
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    max_concurrency: int = 4,
    gateway_url: str = live.DEFAULT_GATEWAY_URL,
) -> QueryAnswerJudgeProviderResult:
    """Rehydrate the complete Sol journal population with ``client=None``."""

    population = load_query_answer_judge_provider_population(
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
    )
    if not population.required_calls:
        return QueryAnswerJudgeProviderResult(
            population.preflight_artifact,
            None,
            0,
            0,
        )
    runtime = _provider_runtime(
        population,
        client=None,
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    _require(
        batch.usage.physical_calls == 0
        and batch.usage.checkpoint_hits == population.required_calls,
        "materialization requires every query-answer Sol response journal",
    )
    return QueryAnswerJudgeProviderResult(
        population.preflight_artifact,
        batch,
        0,
        batch.usage.checkpoint_hits,
    )


def _stable_batch(batch: FastCompletionBatch) -> dict[str, Any]:
    value = batch.model_dump()
    return {
        "logical_completions": value["logical_completions"],
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
        "provenance": value["provenance"],
        "runtime_identity_sha256": value["runtime_identity_sha256"],
        "prompt_population": value["prompt_population"],
    }


def _route_aggregates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["route_id"])].append(row)
    result: list[dict[str, Any]] = []
    for route_id in sorted(groups):
        children = groups[route_id]
        correct = sum(bool(row["correct"]) for row in children)
        baseline = sum(bool(row["baseline_correct"]) for row in children)
        result.append(
            {
                "accuracy": correct / len(children),
                "baseline_accuracy": baseline / len(children),
                "baseline_correct": baseline,
                "changed_predictions": sum(
                    bool(row["changed_from_parent"]) for row in children
                ),
                "correct": correct,
                "incorrect": len(children) - correct,
                "net_marginal": sum(bool(row["rescued"]) for row in children)
                - sum(bool(row["regressed"]) for row in children),
                "questions": len(children),
                "regressed": sum(bool(row["regressed"]) for row in children),
                "rescued": sum(bool(row["rescued"]) for row in children),
                "route_id": route_id,
            }
        )
    return result


def _judge_payload(
    plan: _QueryAnswerJudgePlan,
    batch: FastCompletionBatch | None,
    *,
    preflight_sha256: str,
) -> dict[str, Any]:
    if plan.required_calls:
        _require(batch is not None, "changed query answers require Sol verdicts")
        assert batch is not None
        _require(
            batch.usage.physical_calls == 0
            and batch.usage.checkpoint_hits == plan.required_calls,
            "query-answer judge materialization requires journal replay",
        )
        records = {row.messages_sha256: row for row in batch.unique_records}
        changed_outputs = {
            prompt.answer_ordinal: (
                prompt,
                output,
                records[prompt.messages_sha256],
            )
            for prompt, output in zip(
                plan.prompt_rows,
                batch.logical_completions,
                strict=True,
            )
        }
    else:
        _require(batch is None, "unchanged query-answer plan forbids verdicts")
        changed_outputs = {}
    parent_outcomes = {row.ordinal: row for row in plan.parent_judge.outcomes}
    rows: list[dict[str, Any]] = []
    for answer, gold in zip(
        plan.answer_plane.rows,
        plan.gold_rows,
        strict=True,
    ):
        parent = parent_outcomes[answer.ordinal]
        if answer.changed_from_parent:
            prompt, output, record = changed_outputs[answer.ordinal]
            correct = parse_binary_judge_verdict(output)
            verdict_sha = quote_sha256(output)
            verdict_source = "new_sol_judge"
            call_key = record.call_key_sha256
            request_journal = record.request_journal_sha256
            response_journal = record.response_journal_sha256
            judge_messages = prompt.messages_sha256
            prompt_tokens: int | None = prompt.prompt_token_proxy
            persisted_output: str | None = output
            output_sha: str | None = verdict_sha
            demand_class = prompt.demand_class
        else:
            correct = parent.correct
            verdict_sha = parent.judge_verdict_sha256
            verdict_source = plan.adapter.inherited_verdict_source
            call_key = request_journal = response_journal = None
            judge_messages = None
            prompt_tokens = None
            persisted_output = None
            output_sha = None
            demand_class = parent.demand_class
        rescued = not parent.correct and correct
        regressed = parent.correct and not correct
        body: dict[str, Any] = {
            "baseline_correct": parent.correct,
            "call_key_sha256": call_key,
            "category": gold.category,
            "changed_from_parent": answer.changed_from_parent,
            "correct": correct,
            "dated_question_sha256": gold.dated_question_sha256,
            "demand_class": demand_class,
            "judge_messages_sha256": judge_messages,
            "judge_output": persisted_output,
            "judge_output_sha256": output_sha,
            "judge_prompt_token_proxy": prompt_tokens,
            "normalized_exact_match": exact_match(
                answer.prediction,
                gold.reference,
            ),
            "normalized_f1": f1_score(answer.prediction, gold.reference),
            "ordinal": answer.ordinal,
            "parent_judge_row_sha256": parent.judge_row_sha256,
            "parent_judge_verdict_sha256": parent.judge_verdict_sha256,
            "parent_prediction_sha256": answer.parent_prediction_sha256,
            "prediction_sha256": answer.prediction_sha256,
            "question_id": answer.question_id,
            "question_sha256": answer.question_sha256,
            "reference_sha256": gold.reference_sha256,
            "regressed": regressed,
            "request_journal_sha256": request_journal,
            "rescued": rescued,
            "response_journal_sha256": response_journal,
            "route_id": answer.route_id,
            "runtime_row_id": answer.runtime_row_id,
            "verdict_sha256": verdict_sha,
            "verdict_source": verdict_source,
        }
        body["judge_row_sha256"] = identity_sha256(body)
        rows.append(body)
    correct_count = sum(bool(row["correct"]) for row in rows)
    baseline_correct = sum(bool(row["baseline_correct"]) for row in rows)
    rescued = sum(bool(row["rescued"]) for row in rows)
    regressed = sum(bool(row["regressed"]) for row in rows)
    population = _prompt_population_projection(plan)
    base_population = _base_population_plane(plan.answer_plane)
    return {
        "adapter_population_id": plan.answer_plane.adapter_population_id,
        "aggregate": {
            "accuracy": correct_count / len(rows),
            "baseline_correct": baseline_correct,
            "changed_predictions": plan.changed_count,
            "correct": correct_count,
            "fresh_judgments": plan.changed_count,
            "fresh_unique_provider_prompts": plan.required_calls,
            "gate_passed": correct_count / len(rows) >= judging.TARGET_ACCURACY,
            "incorrect": len(rows) - correct_count,
            "inherited_judgments": len(rows) - plan.changed_count,
            "mean_f1": sum(float(row["normalized_f1"]) for row in rows)
            / len(rows),
            "net_marginal": rescued - regressed,
            "normalized_exact_match": sum(
                bool(row["normalized_exact_match"]) for row in rows
            ),
            "questions": len(rows),
            "regressed": regressed,
            "rescued": rescued,
            "target_accuracy": judging.TARGET_ACCURACY,
        },
        "answer_plan_id": plan.adapter.answer_plan_id,
        "answer_run_sha256": plan.answer_plane.run_sha256,
        "arm_kind": plan.adapter.kind,
        "arm_label": plan.adapter.arm_label,
        "arm_plan_id": plan.adapter.arm_plan_id,
        "category_aggregates": judging._category_aggregates(rows),
        "change_projection_sha256": plan.change_projection[
            "change_projection_sha256"
        ],
        "completion_batch": None if batch is None else _stable_batch(batch),
        "format": JUDGE_FORMAT,
        "gold_loaded_posthoc": True,
        "gold_population_sha256": plan.gold_population_sha256,
        "judge_completions_may_echo_gold": True,
        "judge_model": judging.DEFAULT_SOL_CALLER_MODEL,
        "judge_plan_id": plan.adapter.judge_plan_id,
        "matched_population_id": base_population.matched_population_id,
        "parent_answer_run_sha256": plan.answer_plane.parent_plane.run_sha256,
        "parent_judge_preflight_sha256": plan.parent_judge.preflight_sha256,
        "parent_judge_sha256": plan.parent_judge.judge_sha256,
        "parent_runtime_ledger_sha256": (
            plan.answer_plane.parent_plane.runtime_ledger_sha256
        ),
        "parent_score_ledger_sha256": plan.parent_judge.score_ledger_sha256,
        "population_identity_sha256": base_population.population_identity_sha256,
        "preflight_artifact_sha256": preflight_sha256,
        "prompt_content_contract": "question_reference_prediction_only",
        "prompt_population_sha256": population["prompt_population_sha256"],
        "question_count": len(rows),
        "questions": rows,
        "renderer_id": plan.adapter.renderer_id,
        "retained_request_token_state_bytes": 0,
        "retrieval_sha256": plan.answer_plane.retrieval_sha256,
        "route_aggregates": _route_aggregates(rows),
        "runtime_ledger_identity_sha256": plan.answer_plane.runtime_ledger.get(
            "ledger_identity_sha256"
        ),
        "runtime_ledger_sha256": plan.answer_plane.runtime_ledger_sha256,
        "snapshot_id": plan.answer_plane.snapshot_id,
        "source_bindings": dict(plan.change_projection["source_bindings"]),
        "unique_provider_prompt_count": plan.required_calls,
    }


def _score_payload(
    plan: _QueryAnswerJudgePlan,
    judge_payload: Mapping[str, Any],
    *,
    judge_sha256: str,
) -> dict[str, Any]:
    raw_rows = judge_payload.get("questions")
    _require(type(raw_rows) is list, "query-answer judge rows must be an array")
    entries: list[ScoreLedgerEntry] = []
    for answer, raw in zip(
        plan.answer_plane.rows,
        raw_rows,
        strict=True,
    ):
        _require(
            type(raw) is dict
            and raw.get("runtime_row_id") == answer.runtime_row_id
            and type(raw.get("correct")) is bool
            and type(raw.get("baseline_correct")) is bool,
            f"query-answer judge/runtime binding changed at {answer.ordinal}",
        )
        correct = bool(raw["correct"])
        baseline = bool(raw["baseline_correct"])
        entries.append(
            ScoreLedgerEntry(
                runtime_row_id=answer.runtime_row_id,
                correct=correct,
                baseline_correct=baseline,
                changed_from_baseline=answer.changed_from_parent,
                rescued=not baseline and correct,
                regressed=baseline and not correct,
                question_only_demand_class=str(raw["demand_class"]),
                judge_row_sha256=str(raw["judge_row_sha256"]),
                judge_verdict_sha256=str(raw["verdict_sha256"]),
                baseline_judge_row_sha256=str(raw["parent_judge_row_sha256"]),
                historical_provider_calls=int(not answer.changed_from_parent),
            )
        )
    return build_score_ledger(
        runtime_ledger=_runtime_json(plan.answer_plane.runtime_ledger),
        entries=entries,
        source_artifacts=(
            {"role": f"{plan.adapter.arm_label}:judge", "sha256": judge_sha256},
            {
                "role": f"{plan.adapter.parent_arm_label}:parent_judge",
                "sha256": plan.parent_judge.judge_sha256,
            },
            {
                "role": f"{plan.adapter.parent_arm_label}:parent_score_ledger",
                "sha256": plan.parent_judge.score_ledger_sha256,
            },
        ),
    )


def _verified_preflight(
    plan: _QueryAnswerJudgePlan,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
) -> SealedArtifact:
    expected = require_sha256(
        expected_preflight_sha256,
        "expected query-answer judge preflight",
    )
    artifact = read_sealed_json(Path(output_root) / JUDGE_PREFLIGHT_NAME)
    _require(
        artifact.sha256 == expected and artifact.payload == _preflight_payload(plan),
        "sealed query-answer judge preflight changed",
    )
    return artifact


def materialize_query_answer_changed_only_judge(
    *,
    answer_plane: object,
    dataset_path: str | Path,
    split_path: str | Path,
    parent_judge_root: str | Path,
    expected_parent_judge_sha256: str,
    expected_parent_score_ledger_sha256: str,
    output_root: str | Path,
    expected_preflight_sha256: str,
    completion_batch: FastCompletionBatch | None,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> QueryAnswerJudgeRunResult:
    """Seal judgments and scores from a complete client-free journal replay."""

    plan = _build_plan(
        answer_plane=answer_plane,
        dataset_path=dataset_path,
        split_path=split_path,
        parent_judge_root=parent_judge_root,
        expected_parent_judge_sha256=expected_parent_judge_sha256,
        expected_parent_score_ledger_sha256=(
            expected_parent_score_ledger_sha256
        ),
        expected_question_count=expected_question_count,
    )
    output = Path(output_root)
    preflight = _verified_preflight(
        plan,
        output_root=output,
        expected_preflight_sha256=expected_preflight_sha256,
    )
    _require(
        not (output / JUDGE_NAME).exists()
        and not (output / SCORE_LEDGER_NAME).exists(),
        "query-answer judge already exists; use replay",
    )
    payload = _judge_payload(
        plan,
        completion_batch,
        preflight_sha256=preflight.sha256,
    )
    judge, _created = publish_sealed_json(output / JUDGE_NAME, payload)
    score_payload = _score_payload(plan, payload, judge_sha256=judge.sha256)
    score, _created = publish_sealed_json(
        output / SCORE_LEDGER_NAME,
        score_payload,
    )
    return QueryAnswerJudgeRunResult(
        judge_artifact=judge,
        score_ledger_artifact=score,
        correct=int(payload["aggregate"]["correct"]),
        physical_provider_calls=0,
        checkpoint_hits=plan.required_calls,
    )


def replay_query_answer_changed_only_judge(
    *,
    answer_plane: object,
    dataset_path: str | Path,
    split_path: str | Path,
    parent_judge_root: str | Path,
    expected_parent_judge_sha256: str,
    expected_parent_score_ledger_sha256: str,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_judge_sha256: str,
    expected_score_ledger_sha256: str,
    max_concurrency: int = 4,
    gateway_url: str = live.DEFAULT_GATEWAY_URL,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> QueryAnswerJudgeRunResult:
    """Rebuild judge and score bytes from immutable Sol journals."""

    expected_judge = require_sha256(
        expected_judge_sha256,
        "expected query-answer judge",
    )
    expected_score = require_sha256(
        expected_score_ledger_sha256,
        "expected query-answer score ledger",
    )
    plan = _build_plan(
        answer_plane=answer_plane,
        dataset_path=dataset_path,
        split_path=split_path,
        parent_judge_root=parent_judge_root,
        expected_parent_judge_sha256=expected_parent_judge_sha256,
        expected_parent_score_ledger_sha256=(
            expected_parent_score_ledger_sha256
        ),
        expected_question_count=expected_question_count,
    )
    output = Path(output_root)
    preflight = _verified_preflight(
        plan,
        output_root=output,
        expected_preflight_sha256=expected_preflight_sha256,
    )
    journals = load_query_answer_judge_journals(
        output_root=output,
        expected_preflight_sha256=preflight.sha256,
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
    )
    source = read_sealed_json(output / JUDGE_NAME)
    score = read_sealed_json(output / SCORE_LEDGER_NAME)
    _require(source.sha256 == expected_judge, "query-answer judge SHA-256 changed")
    _require(score.sha256 == expected_score, "query-answer score SHA-256 changed")
    payload = _judge_payload(
        plan,
        journals.batch,
        preflight_sha256=preflight.sha256,
    )
    _require(
        canonical_json_bytes(payload) == canonical_json_bytes(source.payload),
        "query-answer judge differs from sealed Sol journals",
    )
    replay, _created = publish_sealed_json(output / JUDGE_REPLAY_NAME, payload)
    score_payload = _score_payload(plan, payload, judge_sha256=source.sha256)
    _require(
        canonical_json_bytes(score_payload) == canonical_json_bytes(score.payload),
        "query-answer score ledger differs from replayed verdicts",
    )
    score_replay, _created = publish_sealed_json(
        output / SCORE_LEDGER_REPLAY_NAME,
        score_payload,
    )
    return QueryAnswerJudgeRunResult(
        judge_artifact=replay,
        score_ledger_artifact=score_replay,
        correct=int(payload["aggregate"]["correct"]),
        physical_provider_calls=0,
        checkpoint_hits=journals.checkpoint_hits,
    )


__all__ = [
    "CHANGE_PROJECTION_FORMAT",
    "EMPTY_PROMPT_POPULATION_FORMAT",
    "JUDGE_CHECKPOINT_DIR_NAME",
    "JUDGE_FORMAT",
    "JUDGE_NAME",
    "JUDGE_PLAN_ID",
    "JUDGE_PREFLIGHT_FORMAT",
    "JUDGE_PREFLIGHT_NAME",
    "JUDGE_REPLAY_NAME",
    "QueryAnswerJudgeProviderResult",
    "QueryAnswerJudgeRunResult",
    "SCORE_LEDGER_NAME",
    "SCORE_LEDGER_REPLAY_NAME",
    "SealedQueryAnswerJudgeProviderPopulation",
    "load_query_answer_judge_journals",
    "load_query_answer_judge_provider_population",
    "materialize_query_answer_changed_only_judge",
    "preflight_query_answer_changed_only_judge",
    "replay_query_answer_changed_only_judge",
    "run_sealed_query_answer_judge_provider",
]
