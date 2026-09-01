"""Sealed parent-aware answer lifecycle for the routed fixed-S1 fact gate.

The historical EM compression plane is verified with zero calls, then joined
to the verified matched S0-v2 parent by exact question identity.  Only gate
rows admitted by the sealed question-only policy enter the Terra population;
every other row copies the exact parent prediction.  Replay reconstructs the
same answer and runtime artifacts from immutable journals without a client.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

from memory_condense.domain._tokenizer import tokenizer_proxy_identity
from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from tools._locked_em_repair_adapter import (
    LockedEMQuestionView,
    load_locked_em_repair_population,
)
from tools.load_locked_s0_em_facts_arm import load_verified_run as load_verified_em_run
from tools.run_locked_s0_em_facts_arm import (
    COMPRESSION_FORMAT as EM_COMPRESSION_FORMAT,
)

from . import live
from .artifacts import SealedArtifact, publish_sealed_json, read_sealed_json
from .contracts import (
    ArtifactRef,
    EvaluationMemorySnapshot,
    MatchedEvalContractError,
    StageDisposition,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
)
from .fact_gate import (
    MAX_PROMPT_TOKENS,
    FactGateResult,
    compile_fixed_s1_em_fact_gate,
    load_fact_route_policy,
)
from .ledger import RuntimeLedgerEntry, _validated_runtime_ledger, build_runtime_ledger
from .population import (
    EXPECTED_QUESTION_COUNT,
    EXPECTED_RETRIEVAL_SHA256,
    MatchedS0Population,
    MatchedS0Row,
)


ARM_LABEL = "S0_PLUS_ROUTED_EM_FACT_GATE_V1"
PARENT_ARM_LABEL = live.ARM_LABEL
ARM_PLAN_ID = "matched_s0_plus_routed_em_fact_gate_v1"
ANSWER_PLAN_ID = "matched_s0_plus_routed_em_fact_gate_terra_answer_v1"
ANSWER_STAGE_ID = "routed_em_fact_gate_terra_answer"
FACT_GATE_STAGE_ID = "routed_em_fact_gate_compile"
RENDERER_ID = "matched_em_facts_parent_guard_v1"
ANSWER_PREFLIGHT_FORMAT = "memory-condense-matched-fact-gate-answer-preflight-v1"
ANSWER_RUN_FORMAT = "memory-condense-matched-fact-gate-answer-run-v1"
EMPTY_PROMPT_POPULATION_FORMAT = "memory-condense-matched-fact-gate-empty-prompts-v1"
ANSWER_PREFLIGHT_NAME = "answer-preflight.json"
ANSWER_RUN_NAME = live.ANSWER_RUN_NAME
ANSWER_REPLAY_NAME = live.ANSWER_REPLAY_NAME
RUNTIME_LEDGER_NAME = live.RUNTIME_LEDGER_NAME
RUNTIME_LEDGER_REPLAY_NAME = live.RUNTIME_LEDGER_REPLAY_NAME
CHECKPOINT_DIR_NAME = live.CHECKPOINT_DIR_NAME


def _require(condition: object, message: str) -> None:
    if not condition:
        raise MatchedEvalContractError(message)


def _plain_messages(messages: object) -> tuple[dict[str, str], ...]:
    _require(type(messages) is tuple, "fact-gate prompt messages must be immutable")
    rows: list[dict[str, str]] = []
    for message in messages:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        _require(
            type(role) is str and type(content) is str,
            "fact-gate prompt message changed",
        )
        rows.append({"role": role, "content": content})
    return tuple(rows)


def _make_provider_client(api_key: str, gateway_url: str) -> Any:
    return live._make_provider_client(api_key, gateway_url)


@dataclass(frozen=True, slots=True)
class _VerifiedEMInputs:
    questions: tuple[LockedEMQuestionView, ...]
    compression_responses: tuple[str, ...]
    em_run_sha256: str
    compression_sha256: str
    historical_population_identity_sha256: str


@dataclass(frozen=True, slots=True)
class _FactGatePlanRow:
    source: MatchedS0Row
    parent: live.VerifiedS0V2AnswerRow
    em_question: LockedEMQuestionView
    gate: FactGateResult

    @property
    def submitted(self) -> bool:
        return self.gate.requires_provider_answer

    @property
    def prompt_messages_sha256(self) -> str:
        if self.gate.prompt is None:
            return self.source.rendered_prompt.messages_sha256
        return self.gate.prompt.messages_sha256

    @property
    def prompt_id(self) -> str:
        if self.gate.prompt is None:
            return self.source.rendered_prompt.prompt_id
        return identity_sha256(
            {
                "fact_gate_receipt_sha256": self.gate.receipt_sha256,
                "format": "memory-condense-matched-fact-gate-prompt-id-v1",
                "messages_sha256": self.gate.prompt.messages_sha256,
                "renderer_id": RENDERER_ID,
            }
        )

    @property
    def prompt_token_proxy(self) -> int:
        if self.gate.prompt is None:
            return self.source.rendered_prompt.total_prompt_token_proxy
        return self.gate.prompt.prompt_token_proxy


@dataclass(frozen=True, slots=True)
class _FactGateAnswerPlan:
    population: MatchedS0Population
    parent_plane: live.VerifiedS0V2AnswerPlane
    snapshot: EvaluationMemorySnapshot
    rows: tuple[_FactGatePlanRow, ...]
    prompt_population: FastPromptPopulation | None
    em_run_sha256: str
    compression_sha256: str
    historical_population_identity_sha256: str
    route_policy_sha256: str

    @property
    def submitted_rows(self) -> tuple[_FactGatePlanRow, ...]:
        return tuple(row for row in self.rows if row.submitted)

    @property
    def required_calls(self) -> int:
        if self.prompt_population is None:
            return 0
        return self.prompt_population.unique_prompt_count


@dataclass(frozen=True, slots=True)
class FactGateAnswerRunResult:
    answer_artifact: SealedArtifact
    runtime_ledger_artifact: SealedArtifact
    physical_provider_calls: int
    checkpoint_hits: int


@dataclass(frozen=True, slots=True)
class VerifiedFactGateAnswerRow:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    prediction: str
    prediction_sha256: str
    prediction_source: str
    parent_prediction_sha256: str
    changed_from_parent: bool
    route_id: str
    gate_disposition: str
    gate_reason: str
    fact_gate_receipt_sha256: str
    final_packet_id: str
    final_prompt_id: str
    final_prompt_messages_sha256: str
    source_row_sha256: str
    runtime_row_id: str
    call_key_sha256: str | None
    request_journal_sha256: str | None
    response_journal_sha256: str | None

    @property
    def messages_sha256(self) -> str:
        return self.final_prompt_messages_sha256


@dataclass(frozen=True, slots=True)
class VerifiedFactGateAnswerPlane:
    run_sha256: str
    replay_sha256: str
    runtime_ledger_sha256: str
    runtime_ledger: Mapping[str, Any]
    arm_label: str
    parent_arm_label: str
    parent_answer_run_sha256: str
    matched_population_id: str
    population_identity_sha256: str
    retrieval_sha256: str
    source_em_run_sha256: str
    source_em_compression_sha256: str
    source_preflight_sha256: str
    route_policy_sha256: str
    snapshot_id: str
    arm_plan_id: str
    answer_plan_id: str
    renderer_id: str
    rows: tuple[VerifiedFactGateAnswerRow, ...]
    parent_plane: live.VerifiedS0V2AnswerPlane

    @property
    def ordered_rows(self) -> tuple[VerifiedFactGateAnswerRow, ...]:
        return self.rows

    @property
    def changed_rows(self) -> tuple[VerifiedFactGateAnswerRow, ...]:
        """Exact population for a later changed-only judge plan."""

        return tuple(row for row in self.rows if row.changed_from_parent)

    @property
    def runtime_ledger_projection(self) -> Mapping[str, Any]:
        return self.runtime_ledger

    def runtime_ledger_json(self) -> dict[str, Any]:
        value = live._thaw_json(self.runtime_ledger)
        assert type(value) is dict
        return value


def _load_verified_em_inputs(
    *,
    em_run_path: str | Path,
    expected_em_run_sha256: str,
    retrieval_path: str | Path,
    expected_retrieval_sha256: str,
    baseline_answers_path: str | Path,
    expected_baseline_answers_sha256: str,
    max_concurrency: int,
) -> _VerifiedEMInputs:
    run_path = Path(em_run_path)
    run, run_sha256 = load_verified_em_run(
        run_path,
        expected_run_sha256=expected_em_run_sha256,
        retrieval_path=retrieval_path,
        baseline_answers_path=baseline_answers_path,
        max_concurrency=max_concurrency,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_baseline_answers_sha256=expected_baseline_answers_sha256,
    )
    compression = read_sealed_json(run_path.parent / "compression.json")
    _require(
        compression.sha256 == run.get("compression_artifact_sha256"),
        "sealed EM run lost its compression artifact",
    )
    _require(
        compression.payload.get("format") == EM_COMPRESSION_FORMAT,
        "sealed EM compression format changed",
    )
    batch = compression.payload.get("completion_batch")
    responses = None if type(batch) is not dict else batch.get("logical_completions")
    _require(
        type(responses) is list and all(type(row) is str for row in responses),
        "sealed EM compression responses changed",
    )
    population = load_locked_em_repair_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        baseline_final_answers_path=baseline_answers_path,
        expected_baseline_final_answers_sha256=expected_baseline_answers_sha256,
    )
    _require(
        len(responses) == population.question_count,
        "sealed EM compression population changed",
    )
    return _VerifiedEMInputs(
        questions=population.questions,
        compression_responses=tuple(responses),
        em_run_sha256=run_sha256,
        compression_sha256=compression.sha256,
        historical_population_identity_sha256=population.population_identity_sha256,
    )


def _load_base_and_parent(
    *,
    retrieval_path: str | Path,
    parent_root: str | Path,
    expected_parent_answer_run_sha256: str,
    max_concurrency: int,
    expected_retrieval_sha256: str | None,
    expected_question_count: int,
) -> tuple[MatchedS0Population, live.VerifiedS0V2AnswerPlane]:
    profile = live.execution_profile(live.RENDERER_ID)
    population = live._load_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
        renderer_id=live.RENDERER_ID,
    )
    root = Path(parent_root)
    parent = live._load_verified_s0_v2_answer_plane_for_population(
        population=population,
        run_path=root / live.ANSWER_RUN_NAME,
        replay_path=root / live.ANSWER_REPLAY_NAME,
        expected_run_sha256=expected_parent_answer_run_sha256,
        checkpoint_dir=root / live.CHECKPOINT_DIR_NAME,
        max_concurrency=max_concurrency,
        profile=profile,
    )
    return population, parent


def _build_plan(
    *,
    retrieval_path: str | Path,
    baseline_answers_path: str | Path,
    expected_baseline_answers_sha256: str,
    em_run_path: str | Path,
    expected_em_run_sha256: str,
    parent_root: str | Path,
    expected_parent_answer_run_sha256: str,
    max_concurrency: int,
    expected_retrieval_sha256: str | None,
    expected_question_count: int,
) -> _FactGateAnswerPlan:
    require_sha256(expected_em_run_sha256, "expected sealed EM run")
    require_sha256(expected_parent_answer_run_sha256, "expected parent answer run")
    require_sha256(
        expected_baseline_answers_sha256,
        "expected historical baseline answers",
    )
    retrieval_sha256 = require_sha256(
        expected_retrieval_sha256,
        "expected retrieval",
    )
    population, parent = _load_base_and_parent(
        retrieval_path=retrieval_path,
        parent_root=parent_root,
        expected_parent_answer_run_sha256=expected_parent_answer_run_sha256,
        max_concurrency=max_concurrency,
        expected_retrieval_sha256=retrieval_sha256,
        expected_question_count=expected_question_count,
    )
    em = _load_verified_em_inputs(
        em_run_path=em_run_path,
        expected_em_run_sha256=expected_em_run_sha256,
        retrieval_path=retrieval_path,
        expected_retrieval_sha256=retrieval_sha256,
        baseline_answers_path=baseline_answers_path,
        expected_baseline_answers_sha256=expected_baseline_answers_sha256,
        max_concurrency=max_concurrency,
    )
    _require(
        len(population.rows)
        == len(parent.rows)
        == len(em.questions)
        == len(em.compression_responses)
        == expected_question_count,
        "fact-gate source populations changed",
    )
    policy = load_fact_route_policy()
    snapshot = replace(
        population.snapshot,
        overlay_revisions=(
            *population.snapshot.overlay_revisions,
            ArtifactRef(role="fact_gate_em_run", sha256=em.em_run_sha256),
            ArtifactRef(
                role="fact_gate_em_compression",
                sha256=em.compression_sha256,
            ),
            ArtifactRef(role="fact_gate_route_policy", sha256=policy.sha256),
        ),
        policy_id=policy.policy_id,
        renderer_id=RENDERER_ID,
        implementation_id="tools_matched_eval_fact_gate_v1",
    )
    rows: list[_FactGatePlanRow] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    for source, parent_row, question, response in zip(
        population.rows,
        parent.rows,
        em.questions,
        em.compression_responses,
        strict=True,
    ):
        _require(
            source.ordinal == parent_row.ordinal == question.ordinal
            and source.packet.question_id
            == parent_row.question_id
            == question.question_id
            and source.packet.question_sha256
            == parent_row.question_sha256
            == question.question_sha256
            and source.packet.dated_question_sha256
            == parent_row.dated_question_sha256
            == question.dated_question_sha256,
            f"fact-gate source binding changed at ordinal {source.ordinal}",
        )
        gate = compile_fixed_s1_em_fact_gate(
            question,
            parent_prediction=parent_row.prediction,
            compression_response=response,
            route_policy=policy,
        )
        _require(
            gate.question_id == source.packet.question_id
            and gate.dated_question_sha256
            == source.packet.dated_question_sha256,
            f"fact-gate compiler changed question {source.ordinal}",
        )
        if gate.requires_provider_answer:
            _require(gate.prompt is not None, "compiled fact gate lost its prompt")
            assert gate.prompt is not None
            prompts.append(_plain_messages(gate.prompt.messages))
        else:
            _require(
                gate.prompt is None
                and gate.fallback_prediction == parent_row.prediction,
                f"fact-gate fallback changed parent at ordinal {source.ordinal}",
            )
        rows.append(
            _FactGatePlanRow(
                source=source,
                parent=parent_row,
                em_question=question,
                gate=gate,
            )
        )
    prompt_population = (
        preflight_fast_completion_prompts(prompts, max_prompt_tokens=MAX_PROMPT_TOKENS)
        if prompts
        else None
    )
    if prompt_population is not None:
        _require(
            prompt_population.logical_prompt_count
            == prompt_population.unique_prompt_count
            == len(prompts),
            "fact-gate prompts must be unique per admitted row",
        )
    plan = _FactGateAnswerPlan(
        population=population,
        parent_plane=parent,
        snapshot=snapshot,
        rows=tuple(rows),
        prompt_population=prompt_population,
        em_run_sha256=em.em_run_sha256,
        compression_sha256=em.compression_sha256,
        historical_population_identity_sha256=(
            em.historical_population_identity_sha256
        ),
        route_policy_sha256=policy.sha256,
    )
    _require(
        len(plan.submitted_rows) == plan.required_calls,
        "fact-gate authorization population changed",
    )
    return plan


def _empty_prompt_population() -> dict[str, Any]:
    body: dict[str, Any] = {
        "format": EMPTY_PROMPT_POPULATION_FORMAT,
        "logical_prompt_count": 0,
        "max_prompt_token_proxy": MAX_PROMPT_TOKENS,
        "ordered_rows": [],
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "unique_prompt_count": 0,
    }
    body["prompt_population_sha256"] = identity_sha256(body)
    return body


def _prompt_population_projection(plan: _FactGateAnswerPlan) -> dict[str, Any]:
    if plan.prompt_population is None:
        return _empty_prompt_population()
    return plan.prompt_population.model_dump()


def _preflight_row(row: _FactGatePlanRow) -> dict[str, Any]:
    return {
        "dated_question_sha256": row.source.packet.dated_question_sha256,
        "fact_gate": row.gate.projection(),
        "final_packet_id": row.source.packet.packet_id,
        "final_prompt_id": row.prompt_id,
        "final_prompt_messages_sha256": row.prompt_messages_sha256,
        "final_prompt_token_proxy": row.prompt_token_proxy,
        "ordinal": row.source.ordinal,
        "parent_prediction_sha256": row.parent.prediction_sha256,
        "provider_call_planned": row.submitted,
        "question_id": row.source.packet.question_id,
        "question_part_sha256": row.source.question_part_sha256,
        "question_sha256": row.source.packet.question_sha256,
    }


def _preflight_artifact(plan: _FactGateAnswerPlan) -> dict[str, Any]:
    prompt_population = _prompt_population_projection(plan)
    result: dict[str, Any] = {
        "answer_model": live.DEFAULT_TERRA_CALLER_MODEL,
        "answer_plan_id": ANSWER_PLAN_ID,
        "arm_label": ARM_LABEL,
        "arm_plan_id": ARM_PLAN_ID,
        "construction_recall_claimed": False,
        "format": ANSWER_PREFLIGHT_FORMAT,
        "gold_loaded": False,
        "hard_prompt_token_cap": MAX_PROMPT_TOKENS,
        "historical_em_population_identity_sha256": (
            plan.historical_population_identity_sha256
        ),
        "logical_prompt_count": plan.required_calls,
        "matched_population_id": plan.population.population_id,
        "ordered_rows": [_preflight_row(row) for row in plan.rows],
        "parent_answer_run_sha256": plan.parent_plane.run_sha256,
        "parent_answer_runtime_ledger_sha256": (
            plan.parent_plane.runtime_ledger_sha256
        ),
        "parent_arm_label": PARENT_ARM_LABEL,
        "population_identity_sha256": (
            plan.population.snapshot.population_identity_sha256
        ),
        "prompt_population": prompt_population,
        "prompt_population_sha256": prompt_population[
            "prompt_population_sha256"
        ],
        "provider_calls": 0,
        "question_count": len(plan.rows),
        "renderer_id": RENDERER_ID,
        "required_authorized_provider_calls": plan.required_calls,
        "retained_request_token_state_bytes": 0,
        "retrieval_sha256": plan.population.retrieval_sha256,
        "route_policy_sha256": plan.route_policy_sha256,
        "snapshot": plan.snapshot.projection(),
        "snapshot_id": plan.snapshot.snapshot_id,
        "source_em_compression_sha256": plan.compression_sha256,
        "source_em_run_sha256": plan.em_run_sha256,
        "source_target_expansion_claimed": False,
        "unique_prompt_count": plan.required_calls,
    }
    assert_gold_blind(result, path="fact_gate_answer_preflight")
    return result


def _ensure_output_isolation(
    *,
    output_root: str | Path,
    parent_root: str | Path,
    em_run_path: str | Path,
) -> Path:
    output = Path(output_root)
    resolved = output.resolve()
    _require(
        resolved != Path(parent_root).resolve(),
        "fact-gate output must not reuse the parent root",
    )
    _require(
        resolved != Path(em_run_path).parent.resolve(),
        "fact-gate output must not reuse the historical EM root",
    )
    return output


def preflight_fact_gate_answers(
    *,
    retrieval_path: str | Path,
    baseline_answers_path: str | Path,
    expected_baseline_answers_sha256: str,
    em_run_path: str | Path,
    expected_em_run_sha256: str,
    parent_root: str | Path,
    expected_parent_answer_run_sha256: str,
    output_root: str | Path,
    max_concurrency: int = 4,
    expected_retrieval_sha256: str | None = EXPECTED_RETRIEVAL_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> SealedArtifact:
    plan = _build_plan(
        retrieval_path=retrieval_path,
        baseline_answers_path=baseline_answers_path,
        expected_baseline_answers_sha256=expected_baseline_answers_sha256,
        em_run_path=em_run_path,
        expected_em_run_sha256=expected_em_run_sha256,
        parent_root=parent_root,
        expected_parent_answer_run_sha256=expected_parent_answer_run_sha256,
        max_concurrency=max_concurrency,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
    )
    output = _ensure_output_isolation(
        output_root=output_root,
        parent_root=parent_root,
        em_run_path=em_run_path,
    )
    artifact, _created = publish_sealed_json(
        output / ANSWER_PREFLIGHT_NAME,
        _preflight_artifact(plan),
    )
    return artifact


def _runtime(
    plan: _FactGateAnswerPlan,
    *,
    checkpoint_dir: str | Path,
    client: Any | None,
    max_concurrency: int,
    preflight_artifact_sha256: str,
) -> FastCompletionRuntime:
    require_sha256(preflight_artifact_sha256, "fact-gate answer preflight")
    _require(bool(plan.submitted_rows), "empty fact-gate population has no runtime")
    return FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=[
            _plain_messages(row.gate.prompt.messages)
            for row in plan.submitted_rows
            if row.gate.prompt is not None
        ],
        model=live.DEFAULT_TERRA_GATEWAY_MODEL,
        client=client,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_new_tokens=live.DEFAULT_MAX_NEW_TOKENS,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "answer_plan_id": ANSWER_PLAN_ID,
            "arm_label": ARM_LABEL,
            "arm_plan_id": ARM_PLAN_ID,
            "authorized_unique_calls": plan.required_calls,
            "gateway_url": live.DEFAULT_GATEWAY_URL,
            "gold_loaded": False,
            "parent_answer_run_sha256": plan.parent_plane.run_sha256,
            "preflight_artifact_sha256": preflight_artifact_sha256,
            "renderer_id": RENDERER_ID,
            "retrieval_sha256": plan.population.retrieval_sha256,
            "route_policy_sha256": plan.route_policy_sha256,
            "snapshot_id": plan.snapshot.snapshot_id,
            "source_em_compression_sha256": plan.compression_sha256,
            "source_em_run_sha256": plan.em_run_sha256,
        },
    )


def _answer_artifact(
    plan: _FactGateAnswerPlan,
    batch: FastCompletionBatch | None,
    *,
    preflight_artifact_sha256: str,
) -> dict[str, Any]:
    if plan.required_calls:
        _require(batch is not None, "fact-gate answer completion batch is missing")
        assert batch is not None
        _require(
            plan.prompt_population is not None
            and batch.prompt_population.prompt_population_sha256
            == plan.prompt_population.prompt_population_sha256,
            "fact-gate Terra prompt population changed",
        )
        completions = iter(batch.logical_completions)
        records = {row.messages_sha256: row for row in batch.unique_records}
    else:
        _require(batch is None, "empty fact-gate plan acquired a completion batch")
        completions = iter(())
        records = {}
    questions: list[dict[str, Any]] = []
    changed = 0
    for row in plan.rows:
        if row.submitted:
            prediction = next(completions)
            record = records[row.prompt_messages_sha256]
            _require(
                type(prediction) is str
                and bool(prediction)
                and quote_sha256(prediction) == record.completion_sha256,
                f"fact-gate Terra completion changed at ordinal {row.source.ordinal}",
            )
            prediction_source = "terra_fact_gate"
            call_key = record.call_key_sha256
            request_journal = record.request_journal_sha256
            response_journal = record.response_journal_sha256
            provider_calls = 1
        else:
            prediction = row.parent.prediction
            prediction_source = "sealed_parent_fallback"
            call_key = request_journal = response_journal = None
            provider_calls = 0
        prediction_sha = quote_sha256(prediction)
        changed_from_parent = prediction_sha != row.parent.prediction_sha256
        changed += int(changed_from_parent)
        body: dict[str, Any] = {
            "call_key_sha256": call_key,
            "changed_from_parent": changed_from_parent,
            "dated_question_sha256": row.source.packet.dated_question_sha256,
            "fact_gate_receipt_sha256": row.gate.receipt_sha256,
            "final_packet_id": row.source.packet.packet_id,
            "final_prompt_id": row.prompt_id,
            "final_prompt_messages_sha256": row.prompt_messages_sha256,
            "final_prompt_token_proxy": row.prompt_token_proxy,
            "gate_disposition": row.gate.disposition,
            "gate_reason": row.gate.reason,
            "ordinal": row.source.ordinal,
            "parent_prediction_sha256": row.parent.prediction_sha256,
            "parent_runtime_row_id": row.parent.runtime_row_id,
            "parent_source_row_sha256": row.parent.source_row_sha256,
            "prediction": prediction,
            "prediction_sha256": prediction_sha,
            "prediction_source": prediction_source,
            "provider_calls": provider_calls,
            "question_id": row.source.packet.question_id,
            "question_part_sha256": row.source.question_part_sha256,
            "question_sha256": row.source.packet.question_sha256,
            "request_journal_sha256": request_journal,
            "response_journal_sha256": response_journal,
            "route_id": row.gate.route_id,
        }
        body["source_row_sha256"] = identity_sha256(body)
        questions.append(body)
    try:
        next(completions)
    except StopIteration:
        pass
    else:
        raise MatchedEvalContractError("fact-gate Terra completion count changed")
    prompt_population = _prompt_population_projection(plan)
    stable_batch = None if batch is None else live._stable_batch(batch)
    if stable_batch is not None:
        _require(
            stable_batch["provenance"]["retained_transformer_token_state_bytes"]
            == 0,
            "fact-gate answer retained transformer token state",
        )
    result: dict[str, Any] = {
        "answer_model": live.DEFAULT_TERRA_CALLER_MODEL,
        "answer_plan_id": ANSWER_PLAN_ID,
        "arm_label": ARM_LABEL,
        "arm_plan_id": ARM_PLAN_ID,
        "changed_prediction_count": changed,
        "completion_batch": stable_batch,
        "construction_recall_claimed": False,
        "format": ANSWER_RUN_FORMAT,
        "gold_loaded": False,
        "logical_prediction_count": len(questions),
        "matched_population_id": plan.population.population_id,
        "parent_answer_run_sha256": plan.parent_plane.run_sha256,
        "parent_answer_runtime_ledger_sha256": (
            plan.parent_plane.runtime_ledger_sha256
        ),
        "parent_arm_label": PARENT_ARM_LABEL,
        "parent_fallback_count": len(questions) - plan.required_calls,
        "population_identity_sha256": (
            plan.population.snapshot.population_identity_sha256
        ),
        "preflight_artifact_sha256": preflight_artifact_sha256,
        "prompt_population": prompt_population,
        "prompt_population_sha256": prompt_population[
            "prompt_population_sha256"
        ],
        "provider_route": {
            "caller_model": live.DEFAULT_TERRA_CALLER_MODEL,
            "gateway_model": live.DEFAULT_TERRA_GATEWAY_MODEL,
            "gateway_url": live.DEFAULT_GATEWAY_URL,
            "max_new_tokens": live.DEFAULT_MAX_NEW_TOKENS,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "retries": 0,
        },
        "question_count": len(questions),
        "questions": questions,
        "renderer_id": RENDERER_ID,
        "retained_request_token_state_bytes": 0,
        "retrieval_sha256": plan.population.retrieval_sha256,
        "route_policy_sha256": plan.route_policy_sha256,
        "snapshot_id": plan.snapshot.snapshot_id,
        "source_em_compression_sha256": plan.compression_sha256,
        "source_em_run_sha256": plan.em_run_sha256,
        "source_target_expansion_claimed": False,
        "submitted_fact_gate_count": plan.required_calls,
        "unique_provider_prompt_count": plan.required_calls,
    }
    assert_gold_blind(result, path="fact_gate_answer_run")
    return result


def _runtime_entries(
    plan: _FactGateAnswerPlan,
    answer_payload: Mapping[str, Any],
) -> tuple[RuntimeLedgerEntry, ...]:
    raw_rows = answer_payload.get("questions")
    _require(
        type(raw_rows) is list and len(raw_rows) == len(plan.rows),
        "fact-gate answer/runtime population changed",
    )
    entries: list[RuntimeLedgerEntry] = []
    for source, raw in zip(plan.rows, raw_rows, strict=True):
        _require(type(raw) is dict, "fact-gate answer row must be an object")
        unsigned = dict(raw)
        source_row_sha = unsigned.pop("source_row_sha256", None)
        _require(
            source_row_sha == identity_sha256(unsigned),
            f"fact-gate answer row seal changed at ordinal {source.source.ordinal}",
        )
        prediction = raw.get("prediction")
        prediction_sha = raw.get("prediction_sha256")
        _require(
            type(prediction) is str
            and bool(prediction)
            and prediction_sha == quote_sha256(prediction),
            f"fact-gate prediction changed at ordinal {source.source.ordinal}",
        )
        provider_calls = raw.get("provider_calls")
        _require(
            type(provider_calls) is int
            and provider_calls == int(source.submitted),
            f"fact-gate provider attribution changed at {source.source.ordinal}",
        )
        selected = source.gate.selected_evidence_ids_before_dedup
        excluded = source.gate.dedup_excluded_evidence_ids
        admitted = source.gate.admitted_delta_evidence_ids
        partitioned = set(excluded) | set(admitted)
        not_admitted = tuple(row for row in selected if row not in partitioned)
        stage_disposition = (
            StageDisposition.ADDED
            if admitted
            else (
                StageDisposition.INVALID
                if source.gate.reason.startswith("invalid_")
                else StageDisposition.NO_OP
            )
        )
        entries.append(
            RuntimeLedgerEntry(
                event_type="stage",
                ordinal=source.source.ordinal,
                question_id=source.source.packet.question_id,
                question_sha256=source.source.packet.question_sha256,
                arm_label=ARM_LABEL,
                parent_arm_label=PARENT_ARM_LABEL,
                stage_id=FACT_GATE_STAGE_ID,
                parent_stage_id=source.source.packet.stage_id,
                mechanism_id="provider_free_routed_em_fact_gate",
                delta_kind="fact_memory_representation",
                renderer_id=RENDERER_ID,
                legacy_renderer=False,
                disposition=stage_disposition,
                candidate_ids=selected,
                selected_before_dedup_ids=selected,
                dedup_excluded_ids=excluded,
                not_admitted_ids=not_admitted,
                admitted_ids=admitted,
                provider_calls=0,
                provider_prompt_cap=0,
                provider_prompt_reserved=0,
                global_provider_prompt_cap=plan.required_calls,
                max_final_prompt_tokens=MAX_PROMPT_TOKENS,
                prompt_token_proxy=source.prompt_token_proxy,
                parent_packet_sha256=source.source.packet.packet_id,
                packet_sha256=source.source.packet.packet_id,
                prompt_id=source.prompt_id,
                prompt_messages_sha256=source.prompt_messages_sha256,
                delta_sha256=source.gate.receipt_sha256,
                stage_receipt_sha256=source.gate.receipt_sha256,
                reason=source.gate.reason,
            )
        )
        entries.append(
            RuntimeLedgerEntry(
                event_type="answer_observation",
                ordinal=source.source.ordinal,
                question_id=source.source.packet.question_id,
                question_sha256=source.source.packet.question_sha256,
                arm_label=ARM_LABEL,
                parent_arm_label=PARENT_ARM_LABEL,
                stage_id=ANSWER_STAGE_ID,
                parent_stage_id=FACT_GATE_STAGE_ID,
                mechanism_id=(
                    "routed_em_fact_gate_terra_responder"
                    if source.submitted
                    else "sealed_parent_prediction_reuse"
                ),
                delta_kind="observation",
                renderer_id=RENDERER_ID,
                legacy_renderer=False,
                disposition=StageDisposition.NO_OP,
                provider_calls=provider_calls,
                provider_prompt_cap=provider_calls,
                provider_prompt_reserved=provider_calls,
                global_provider_prompt_cap=plan.required_calls,
                max_final_prompt_tokens=MAX_PROMPT_TOKENS,
                prompt_token_proxy=source.prompt_token_proxy,
                parent_packet_sha256=source.source.packet.packet_id,
                packet_sha256=source.source.packet.packet_id,
                prompt_id=source.prompt_id,
                prompt_messages_sha256=source.prompt_messages_sha256,
                prediction=prediction,
                prediction_sha256=str(prediction_sha),
                changed_from_parent=raw.get("changed_from_parent"),
                source_row_sha256=str(source_row_sha),
                reason=(
                    "sealed_terra_routed_em_fact_gate_prediction"
                    if source.submitted
                    else "sealed_s0_v2_parent_prediction_reuse"
                ),
            )
        )
    return tuple(entries)


def _runtime_ledger(
    plan: _FactGateAnswerPlan,
    answer_payload: Mapping[str, Any],
    *,
    answer_artifact_sha256: str,
    preflight_artifact_sha256: str,
) -> dict[str, Any]:
    return build_runtime_ledger(
        snapshot_id=plan.snapshot.snapshot_id,
        plan_id=ANSWER_PLAN_ID,
        entries=_runtime_entries(plan, answer_payload),
        source_artifacts=(
            {
                "role": f"{ARM_LABEL}:sealed_retrieval",
                "sha256": plan.population.retrieval_sha256,
            },
            {
                "role": f"{ARM_LABEL}:em_run",
                "sha256": plan.em_run_sha256,
            },
            {
                "role": f"{ARM_LABEL}:em_compression",
                "sha256": plan.compression_sha256,
            },
            {
                "role": f"{ARM_LABEL}:route_policy",
                "sha256": plan.route_policy_sha256,
            },
            {
                "role": f"{ARM_LABEL}:parent_answer_run",
                "sha256": plan.parent_plane.run_sha256,
            },
            {
                "role": f"{ARM_LABEL}:answer_preflight",
                "sha256": preflight_artifact_sha256,
            },
            {
                "role": f"{ARM_LABEL}:answer_run",
                "sha256": answer_artifact_sha256,
            },
        ),
    )


def _verified_plane(
    *,
    plan: _FactGateAnswerPlan,
    run: SealedArtifact,
    replay: SealedArtifact,
    runtime_ledger: SealedArtifact,
) -> VerifiedFactGateAnswerPlane:
    payload = run.payload
    raw_preflight_sha256 = payload.get("preflight_artifact_sha256")
    _require(
        type(raw_preflight_sha256) is str,
        "fact-gate answer preflight SHA-256 changed",
    )
    source_preflight_sha256 = require_sha256(
        raw_preflight_sha256,
        "fact-gate answer preflight SHA-256",
    )
    _require(
        run.sha256 == replay.sha256
        and canonical_json_bytes(payload) == canonical_json_bytes(replay.payload),
        "fact-gate answer run/replay differ",
    )
    _require(
        payload.get("format") == ANSWER_RUN_FORMAT
        and payload.get("arm_label") == ARM_LABEL
        and payload.get("parent_arm_label") == PARENT_ARM_LABEL
        and payload.get("parent_answer_run_sha256") == plan.parent_plane.run_sha256
        and payload.get("snapshot_id") == plan.snapshot.snapshot_id
        and payload.get("arm_plan_id") == ARM_PLAN_ID
        and payload.get("answer_plan_id") == ANSWER_PLAN_ID
        and payload.get("renderer_id") == RENDERER_ID
        and payload.get("source_em_run_sha256") == plan.em_run_sha256
        and payload.get("source_em_compression_sha256") == plan.compression_sha256
        and payload.get("route_policy_sha256") == plan.route_policy_sha256,
        "fact-gate verified answer envelope changed",
    )
    ledger_identity, answer_row_ids = _validated_runtime_ledger(
        runtime_ledger.payload
    )
    require_sha256(ledger_identity, "fact-gate runtime ledger identity")
    _require(
        runtime_ledger.payload.get("snapshot_id") == plan.snapshot.snapshot_id
        and runtime_ledger.payload.get("plan_id") == ANSWER_PLAN_ID
        and len(answer_row_ids) == len(plan.rows),
        "fact-gate verified runtime ledger changed",
    )
    raw_rows = payload.get("questions")
    ledger_rows = tuple(
        row
        for row in runtime_ledger.payload["rows"]
        if row["event_type"] == "answer_observation"
    )
    _require(
        type(raw_rows) is list
        and len(raw_rows) == len(ledger_rows) == len(plan.rows),
        "fact-gate verified answer population changed",
    )
    rows: list[VerifiedFactGateAnswerRow] = []
    for source, raw, ledger, runtime_row_id in zip(
        plan.rows,
        raw_rows,
        ledger_rows,
        answer_row_ids,
        strict=True,
    ):
        _require(type(raw) is dict, "fact-gate verified row changed")
        unsigned = dict(raw)
        source_row_sha = unsigned.pop("source_row_sha256", None)
        prediction = raw.get("prediction")
        prediction_sha = raw.get("prediction_sha256")
        changed = raw.get("changed_from_parent")
        _require(
            source_row_sha == identity_sha256(unsigned)
            and ledger.get("source_row_sha256") == source_row_sha
            and ledger.get("row_id") == runtime_row_id
            and ledger.get("prediction_sha256") == prediction_sha,
            f"fact-gate answer/runtime binding changed at {source.source.ordinal}",
        )
        _require(
            type(prediction) is str
            and bool(prediction)
            and prediction_sha == quote_sha256(prediction)
            and type(changed) is bool
            and changed == (prediction_sha != source.parent.prediction_sha256),
            f"fact-gate verified prediction changed at {source.source.ordinal}",
        )
        call_key = raw.get("call_key_sha256")
        request_journal = raw.get("request_journal_sha256")
        response_journal = raw.get("response_journal_sha256")
        if source.submitted:
            _require(
                raw.get("prediction_source") == "terra_fact_gate",
                "submitted fact-gate row lost Terra provenance",
            )
            for value, label in (
                (call_key, "fact-gate call key"),
                (request_journal, "fact-gate request journal"),
                (response_journal, "fact-gate response journal"),
            ):
                require_sha256(str(value), label)
        else:
            _require(
                call_key is None
                and request_journal is None
                and response_journal is None
                and raw.get("prediction_source") == "sealed_parent_fallback"
                and prediction == source.parent.prediction,
                "fact-gate fallback changed its sealed parent prediction",
            )
        rows.append(
            VerifiedFactGateAnswerRow(
                ordinal=source.source.ordinal,
                question_id=source.source.packet.question_id,
                question_sha256=source.source.packet.question_sha256,
                dated_question_sha256=source.source.packet.dated_question_sha256,
                prediction=prediction,
                prediction_sha256=str(prediction_sha),
                prediction_source=str(raw["prediction_source"]),
                parent_prediction_sha256=source.parent.prediction_sha256,
                changed_from_parent=bool(changed),
                route_id=source.gate.route_id,
                gate_disposition=source.gate.disposition,
                gate_reason=source.gate.reason,
                fact_gate_receipt_sha256=source.gate.receipt_sha256,
                final_packet_id=source.source.packet.packet_id,
                final_prompt_id=source.prompt_id,
                final_prompt_messages_sha256=source.prompt_messages_sha256,
                source_row_sha256=str(source_row_sha),
                runtime_row_id=runtime_row_id,
                call_key_sha256=None if call_key is None else str(call_key),
                request_journal_sha256=(
                    None if request_journal is None else str(request_journal)
                ),
                response_journal_sha256=(
                    None if response_journal is None else str(response_journal)
                ),
            )
        )
    return VerifiedFactGateAnswerPlane(
        run_sha256=run.sha256,
        replay_sha256=replay.sha256,
        runtime_ledger_sha256=runtime_ledger.sha256,
        runtime_ledger=live._freeze_json(runtime_ledger.payload),
        arm_label=ARM_LABEL,
        parent_arm_label=PARENT_ARM_LABEL,
        parent_answer_run_sha256=plan.parent_plane.run_sha256,
        matched_population_id=plan.population.population_id,
        population_identity_sha256=(
            plan.population.snapshot.population_identity_sha256
        ),
        retrieval_sha256=plan.population.retrieval_sha256,
        source_em_run_sha256=plan.em_run_sha256,
        source_em_compression_sha256=plan.compression_sha256,
        source_preflight_sha256=source_preflight_sha256,
        route_policy_sha256=plan.route_policy_sha256,
        snapshot_id=plan.snapshot.snapshot_id,
        arm_plan_id=ARM_PLAN_ID,
        answer_plan_id=ANSWER_PLAN_ID,
        renderer_id=RENDERER_ID,
        rows=tuple(rows),
        parent_plane=plan.parent_plane,
    )


def _replay_plan(
    plan: _FactGateAnswerPlan,
    *,
    source: SealedArtifact,
    expected_run_sha256: str,
    output_root: str | Path,
    max_concurrency: int,
) -> VerifiedFactGateAnswerPlane:
    _require(source.sha256 == expected_run_sha256, "fact-gate answer run changed")
    output = Path(output_root)
    preflight = read_sealed_json(output / ANSWER_PREFLIGHT_NAME)
    _require(
        preflight.payload == _preflight_artifact(plan),
        "fact-gate preflight changed during replay",
    )
    batch = (
        _runtime(
            plan,
            checkpoint_dir=output / CHECKPOINT_DIR_NAME,
            client=None,
            max_concurrency=max_concurrency,
            preflight_artifact_sha256=preflight.sha256,
        ).run()
        if plan.required_calls
        else None
    )
    if batch is not None:
        _require(batch.usage.physical_calls == 0, "fact-gate replay made calls")
        _require(
            batch.usage.checkpoint_hits == plan.required_calls,
            "fact-gate replay checkpoint population changed",
        )
    expected = _answer_artifact(
        plan,
        batch,
        preflight_artifact_sha256=preflight.sha256,
    )
    _require(
        canonical_json_bytes(expected) == canonical_json_bytes(source.payload),
        "fact-gate answer differs from immutable Terra journals",
    )
    replay, _created = publish_sealed_json(output / ANSWER_REPLAY_NAME, expected)
    expected_ledger = _runtime_ledger(
        plan,
        expected,
        answer_artifact_sha256=source.sha256,
        preflight_artifact_sha256=preflight.sha256,
    )
    ledger = read_sealed_json(output / RUNTIME_LEDGER_NAME)
    _require(
        canonical_json_bytes(expected_ledger)
        == canonical_json_bytes(ledger.payload),
        "fact-gate runtime ledger differs from replayed answers",
    )
    publish_sealed_json(output / RUNTIME_LEDGER_REPLAY_NAME, expected_ledger)
    return _verified_plane(
        plan=plan,
        run=source,
        replay=replay,
        runtime_ledger=ledger,
    )


def _authorize(
    plan: _FactGateAnswerPlan,
    *,
    enable_provider: bool,
    authorized_provider_calls: int,
) -> None:
    _require(type(enable_provider) is bool, "provider enablement must be exact bool")
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == plan.required_calls,
        "authorized fact-gate provider calls must exactly equal "
        f"{plan.required_calls}",
    )
    if plan.required_calls:
        _require(enable_provider, "fact-gate answer run requires provider enablement")
    else:
        _require(not enable_provider, "empty fact-gate run forbids provider enablement")


def run_fact_gate_answers(
    *,
    retrieval_path: str | Path,
    baseline_answers_path: str | Path,
    expected_baseline_answers_sha256: str,
    em_run_path: str | Path,
    expected_em_run_sha256: str,
    parent_root: str | Path,
    expected_parent_answer_run_sha256: str,
    output_root: str | Path,
    enable_provider: bool,
    authorized_provider_calls: int,
    api_key_env: str = live.DEFAULT_API_KEY_ENV,
    max_concurrency: int = 4,
    expected_retrieval_sha256: str | None = EXPECTED_RETRIEVAL_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> FactGateAnswerRunResult:
    plan = _build_plan(
        retrieval_path=retrieval_path,
        baseline_answers_path=baseline_answers_path,
        expected_baseline_answers_sha256=expected_baseline_answers_sha256,
        em_run_path=em_run_path,
        expected_em_run_sha256=expected_em_run_sha256,
        parent_root=parent_root,
        expected_parent_answer_run_sha256=expected_parent_answer_run_sha256,
        max_concurrency=max_concurrency,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
    )
    _authorize(
        plan,
        enable_provider=enable_provider,
        authorized_provider_calls=authorized_provider_calls,
    )
    output = _ensure_output_isolation(
        output_root=output_root,
        parent_root=parent_root,
        em_run_path=em_run_path,
    )
    preflight, _created = publish_sealed_json(
        output / ANSWER_PREFLIGHT_NAME,
        _preflight_artifact(plan),
    )
    existing = output / ANSWER_RUN_NAME
    if existing.exists():
        source = read_sealed_json(existing)
        _replay_plan(
            plan,
            source=source,
            expected_run_sha256=source.sha256,
            output_root=output,
            max_concurrency=max_concurrency,
        )
        return FactGateAnswerRunResult(
            answer_artifact=source,
            runtime_ledger_artifact=read_sealed_json(
                output / RUNTIME_LEDGER_NAME
            ),
            physical_provider_calls=0,
            checkpoint_hits=plan.required_calls,
        )
    batch: FastCompletionBatch | None = None
    if plan.required_calls:
        load_dotenv()
        api_key = os.environ.get(api_key_env, "").strip()
        _require(bool(api_key), f"provider API key is empty: {api_key_env}")
        client = _make_provider_client(api_key, live.DEFAULT_GATEWAY_URL)
        try:
            batch = _runtime(
                plan,
                checkpoint_dir=output / CHECKPOINT_DIR_NAME,
                client=client,
                max_concurrency=max_concurrency,
                preflight_artifact_sha256=preflight.sha256,
            ).run()
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        _require(
            batch.usage.physical_calls + batch.usage.checkpoint_hits
            == plan.required_calls,
            "fact-gate Terra journal population changed",
        )
    payload = _answer_artifact(
        plan,
        batch,
        preflight_artifact_sha256=preflight.sha256,
    )
    answer, _created = publish_sealed_json(output / ANSWER_RUN_NAME, payload)
    ledger_payload = _runtime_ledger(
        plan,
        payload,
        answer_artifact_sha256=answer.sha256,
        preflight_artifact_sha256=preflight.sha256,
    )
    ledger, _created = publish_sealed_json(
        output / RUNTIME_LEDGER_NAME,
        ledger_payload,
    )
    return FactGateAnswerRunResult(
        answer_artifact=answer,
        runtime_ledger_artifact=ledger,
        physical_provider_calls=0 if batch is None else batch.usage.physical_calls,
        checkpoint_hits=0 if batch is None else batch.usage.checkpoint_hits,
    )


def replay_fact_gate_answers(
    *,
    retrieval_path: str | Path,
    baseline_answers_path: str | Path,
    expected_baseline_answers_sha256: str,
    em_run_path: str | Path,
    expected_em_run_sha256: str,
    parent_root: str | Path,
    expected_parent_answer_run_sha256: str,
    output_root: str | Path,
    expected_run_sha256: str,
    max_concurrency: int = 4,
    expected_retrieval_sha256: str | None = EXPECTED_RETRIEVAL_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> VerifiedFactGateAnswerPlane:
    require_sha256(expected_run_sha256, "expected fact-gate answer run")
    plan = _build_plan(
        retrieval_path=retrieval_path,
        baseline_answers_path=baseline_answers_path,
        expected_baseline_answers_sha256=expected_baseline_answers_sha256,
        em_run_path=em_run_path,
        expected_em_run_sha256=expected_em_run_sha256,
        parent_root=parent_root,
        expected_parent_answer_run_sha256=expected_parent_answer_run_sha256,
        max_concurrency=max_concurrency,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
    )
    output = _ensure_output_isolation(
        output_root=output_root,
        parent_root=parent_root,
        em_run_path=em_run_path,
    )
    source = read_sealed_json(output / ANSWER_RUN_NAME)
    return _replay_plan(
        plan,
        source=source,
        expected_run_sha256=expected_run_sha256,
        output_root=output,
        max_concurrency=max_concurrency,
    )


__all__ = [
    "ANSWER_PLAN_ID",
    "ANSWER_PREFLIGHT_FORMAT",
    "ANSWER_PREFLIGHT_NAME",
    "ANSWER_REPLAY_NAME",
    "ANSWER_RUN_FORMAT",
    "ANSWER_RUN_NAME",
    "ARM_LABEL",
    "ARM_PLAN_ID",
    "CHECKPOINT_DIR_NAME",
    "FactGateAnswerRunResult",
    "RENDERER_ID",
    "RUNTIME_LEDGER_NAME",
    "RUNTIME_LEDGER_REPLAY_NAME",
    "VerifiedFactGateAnswerPlane",
    "VerifiedFactGateAnswerRow",
    "preflight_fact_gate_answers",
    "replay_fact_gate_answers",
    "run_fact_gate_answers",
]
