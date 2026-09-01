"""Parent-aware Terra answers for the two independent closure arms.

The closure retriever and common runner stay provider-free.  This module
submits only descendants whose closure stage actually added evidence and
reuses the fully verified matched S0-v2 parent prediction everywhere else.
Run and replay artifacts are byte-identical; replay reconstructs every live
completion from immutable journals before exposing a verified answer plane.
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

from . import live
from .artifacts import SealedArtifact, publish_sealed_json, read_sealed_json
from .closure import (
    ARM_LABELS,
    GLOBAL_ARM,
    MAX_FINAL_PROMPT_TOKENS,
    REPRESENTATIVE_ARM,
    IndependentClosureGeneration,
    IndependentClosureMembershipAdapter,
    IndependentClosureQuestion,
    independent_closure_arm_plan,
    load_independent_closure_generation,
)
from .contracts import (
    ArmPlan,
    EvaluationMemorySnapshot,
    MatchedEvalContractError,
    StageDisposition,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
)
from .ledger import (
    RuntimeLedgerEntry,
    _validated_runtime_ledger,
    build_runtime_ledger,
    runtime_entry_from_stage_run,
)
from .population import (
    EXPECTED_QUESTION_COUNT,
    EXPECTED_RETRIEVAL_SHA256,
    MatchedS0Population,
    MatchedS0Row,
)
from .renderer import RENDERER_ID
from .runner import ArmRunResult, MatchedEvalRunner, StageRunResult


ANSWER_PREFLIGHT_FORMAT = (
    "memory-condense-matched-independent-closure-answer-preflight-v1"
)
ANSWER_RUN_FORMAT = "memory-condense-matched-independent-closure-answer-run-v1"
EMPTY_PROMPT_POPULATION_FORMAT = (
    "memory-condense-matched-independent-closure-empty-prompts-v1"
)
ANSWER_PREFLIGHT_NAME = "answer-preflight.json"
ANSWER_RUN_NAME = live.ANSWER_RUN_NAME
ANSWER_REPLAY_NAME = live.ANSWER_REPLAY_NAME
RUNTIME_LEDGER_NAME = live.RUNTIME_LEDGER_NAME
RUNTIME_LEDGER_REPLAY_NAME = live.RUNTIME_LEDGER_REPLAY_NAME
CHECKPOINT_DIR_NAME = live.CHECKPOINT_DIR_NAME
PARENT_ARM_LABEL = live.ARM_LABEL

_ANSWER_PLAN_IDS = {
    REPRESENTATIVE_ARM: "matched_representative_bridge_closure_terra_answer_v1",
    GLOBAL_ARM: "matched_artifact_global_closure_terra_answer_v1",
}
_ANSWER_STAGE_IDS = {
    REPRESENTATIVE_ARM: "representative_bridge_closure_terra_answer",
    GLOBAL_ARM: "artifact_global_closure_terra_answer",
}


def _require(condition: object, message: str) -> None:
    if not condition:
        raise MatchedEvalContractError(message)


def _plain_messages(messages: object) -> tuple[dict[str, str], ...]:
    _require(type(messages) is tuple, "closure prompt messages must be immutable")
    result: list[dict[str, str]] = []
    for message in messages:
        _require(isinstance(message, Mapping), "closure prompt message changed")
        result.append(
            {
                "role": str(message["role"]),
                "content": str(message["content"]),
            }
        )
    return tuple(result)


def _make_provider_client(api_key: str, gateway_url: str) -> Any:
    return live._make_provider_client(api_key, gateway_url)


@dataclass(frozen=True, slots=True)
class _ClosureAnswerPlanRow:
    source: MatchedS0Row
    parent: live.VerifiedS0V2AnswerRow
    question: IndependentClosureQuestion
    run: ArmRunResult
    stage: StageRunResult

    @property
    def added(self) -> bool:
        return self.stage.trace.disposition is StageDisposition.ADDED

    @property
    def parent_fallback(self) -> bool:
        return self.stage.trace.disposition in {
            StageDisposition.NO_OP,
            StageDisposition.OVERFLOW,
        }


@dataclass(frozen=True, slots=True)
class _ClosureAnswerPlan:
    population: MatchedS0Population
    parent_plane: live.VerifiedS0V2AnswerPlane
    generation: IndependentClosureGeneration
    snapshot: EvaluationMemorySnapshot
    arm_label: str
    arm_plan: ArmPlan
    answer_plan_id: str
    rows: tuple[_ClosureAnswerPlanRow, ...]
    prompt_population: FastPromptPopulation | None

    @property
    def required_calls(self) -> int:
        return 0 if self.prompt_population is None else (
            self.prompt_population.unique_prompt_count
        )

    @property
    def added_rows(self) -> tuple[_ClosureAnswerPlanRow, ...]:
        return tuple(row for row in self.rows if row.added)


@dataclass(frozen=True, slots=True)
class ClosureAnswerRunResult:
    answer_artifact: SealedArtifact
    runtime_ledger_artifact: SealedArtifact
    physical_provider_calls: int
    checkpoint_hits: int


@dataclass(frozen=True, slots=True)
class VerifiedClosureAnswerRow:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    prediction: str
    prediction_sha256: str
    prediction_source: str
    parent_prediction_sha256: str
    changed_from_parent: bool
    stage_disposition: str
    final_packet_id: str
    final_prompt_id: str
    final_prompt_messages_sha256: str
    final_stage_receipt_sha256: str
    source_row_sha256: str
    runtime_row_id: str
    call_key_sha256: str | None
    request_journal_sha256: str | None
    response_journal_sha256: str | None

    @property
    def messages_sha256(self) -> str:
        """Compatibility name for the final verified responder prompt."""

        return self.final_prompt_messages_sha256


@dataclass(frozen=True, slots=True)
class VerifiedClosureAnswerPlane:
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
    source_retrieval_generation_sha256: str
    source_eligibility_manifest_sha256: str
    source_preflight_sha256: str
    snapshot_id: str
    arm_plan_id: str
    answer_plan_id: str
    renderer_id: str
    rows: tuple[VerifiedClosureAnswerRow, ...]
    parent_plane: live.VerifiedS0V2AnswerPlane

    @property
    def runtime_ledger_projection(self) -> Mapping[str, Any]:
        """Public immutable runtime-ledger projection."""

        return self.runtime_ledger

    @property
    def ordered_rows(self) -> tuple[VerifiedClosureAnswerRow, ...]:
        return self.rows

    def runtime_ledger_json(self) -> dict[str, Any]:
        value = live._thaw_json(self.runtime_ledger)
        assert type(value) is dict
        return value


def _empty_prompt_population() -> dict[str, Any]:
    body: dict[str, Any] = {
        "format": EMPTY_PROMPT_POPULATION_FORMAT,
        "logical_prompt_count": 0,
        "max_prompt_token_proxy": MAX_FINAL_PROMPT_TOKENS,
        "ordered_rows": [],
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "unique_prompt_count": 0,
    }
    body["prompt_population_sha256"] = identity_sha256(body)
    return body


def _prompt_population_projection(plan: _ClosureAnswerPlan) -> dict[str, Any]:
    if plan.prompt_population is None:
        return _empty_prompt_population()
    return plan.prompt_population.model_dump()


def _load_base_and_parent(
    *,
    retrieval_path: str | Path,
    parent_root: str | Path,
    expected_parent_answer_run_sha256: str,
    max_concurrency: int,
    expected_retrieval_sha256: str | None,
    expected_question_count: int,
) -> tuple[MatchedS0Population, live.VerifiedS0V2AnswerPlane]:
    profile = live.execution_profile(RENDERER_ID)
    population = live._load_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
        renderer_id=RENDERER_ID,
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
    arm_label: str,
    retrieval_path: str | Path,
    generation_path: str | Path,
    expected_generation_sha256: str,
    eligibility_manifest_path: str | Path,
    expected_eligibility_manifest_sha256: str,
    parent_root: str | Path,
    expected_parent_answer_run_sha256: str,
    max_concurrency: int,
    expected_retrieval_sha256: str | None,
    expected_question_count: int,
) -> _ClosureAnswerPlan:
    _require(arm_label in ARM_LABELS, "unknown independent closure answer arm")
    require_sha256(expected_generation_sha256, "expected closure generation")
    require_sha256(
        expected_eligibility_manifest_sha256,
        "expected closure eligibility manifest",
    )
    require_sha256(
        expected_parent_answer_run_sha256,
        "expected parent answer run",
    )
    population, parent = _load_base_and_parent(
        retrieval_path=retrieval_path,
        parent_root=parent_root,
        expected_parent_answer_run_sha256=expected_parent_answer_run_sha256,
        max_concurrency=max_concurrency,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
    )
    _require(
        population.renderer_id == parent.renderer_id == RENDERER_ID,
        "closure answers require the verified matched S0-v2 renderer",
    )
    _require(
        parent.run_sha256
        == parent.replay_sha256
        == expected_parent_answer_run_sha256,
        "closure parent answer run/replay changed",
    )
    _require(
        parent.matched_population_id == population.population_id
        and parent.population_identity_sha256
        == population.snapshot.population_identity_sha256
        and parent.snapshot_id == population.snapshot.snapshot_id,
        "closure parent belongs to another matched population",
    )
    generation = load_independent_closure_generation(
        generation_path,
        expected_generation_sha256=expected_generation_sha256,
        eligibility_manifest_path=eligibility_manifest_path,
        expected_eligibility_manifest_sha256=(
            expected_eligibility_manifest_sha256
        ),
        population=population,
    )
    snapshot = replace(
        population.snapshot,
        overlay_revisions=(
            *population.snapshot.overlay_revisions,
            generation.artifact_ref,
        ),
    )
    arm_plan = independent_closure_arm_plan(arm_label)
    adapter = IndependentClosureMembershipAdapter(generation, arm_label)
    runner = MatchedEvalRunner({adapter.mechanism_id: adapter})
    parents = {row.ordinal: row for row in parent.rows}
    rows: list[_ClosureAnswerPlanRow] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    for source in population.rows:
        parent_row = parents.get(source.ordinal)
        _require(parent_row is not None, "closure parent answer order changed")
        _require(
            parent_row.question_id == source.packet.question_id
            and parent_row.question_sha256 == source.packet.question_sha256
            and parent_row.dated_question_sha256
            == source.packet.dated_question_sha256
            and parent_row.messages_sha256
            == source.rendered_prompt.messages_sha256,
            f"closure parent answer binding changed at ordinal {source.ordinal}",
        )
        question = generation.question(source.packet.question_id)
        run = runner.run(
            snapshot=snapshot,
            root_packet=source.packet,
            plan=arm_plan,
        )
        stage = run.stage(arm_plan.stages[0].stage_id)
        disposition = stage.trace.disposition
        _require(
            disposition
            in {
                StageDisposition.ADDED,
                StageDisposition.NO_OP,
                StageDisposition.OVERFLOW,
            },
            "closure answer refuses stage disposition "
            f"{disposition.value} at ordinal {source.ordinal}: "
            f"{stage.trace.reason or 'no reason'}",
        )
        added = disposition is StageDisposition.ADDED
        if added:
            _require(
                question.eligible
                and stage.packet.packet_id != source.packet.packet_id
                and bool(stage.packet.admitted_evidence),
                f"closure added stage is not a valid descendant at {source.ordinal}",
            )
            prompts.append(_plain_messages(stage.rendered_prompt.messages))
        else:
            _require(
                stage.packet.packet_id == source.packet.packet_id
                and stage.rendered_prompt.prompt_id
                == source.rendered_prompt.prompt_id,
                f"closure non-added stage changed S0 at ordinal {source.ordinal}",
            )
        rows.append(
            _ClosureAnswerPlanRow(
                source=source,
                parent=parent_row,
                question=question,
                run=run,
                stage=stage,
            )
        )
    prompt_population = (
        preflight_fast_completion_prompts(
            prompts,
            max_prompt_tokens=MAX_FINAL_PROMPT_TOKENS,
        )
        if prompts
        else None
    )
    if prompt_population is not None:
        _require(
            prompt_population.logical_prompt_count
            == prompt_population.unique_prompt_count
            == len(prompts),
            "closure answer prompts must be unique per added descendant",
        )
    result = _ClosureAnswerPlan(
        population=population,
        parent_plane=parent,
        generation=generation,
        snapshot=snapshot,
        arm_label=arm_label,
        arm_plan=arm_plan,
        answer_plan_id=_ANSWER_PLAN_IDS[arm_label],
        rows=tuple(rows),
        prompt_population=prompt_population,
    )
    _require(
        len(result.rows) == expected_question_count
        and len(result.added_rows) == result.required_calls,
        "closure answer population changed",
    )
    return result


def _preflight_row(row: _ClosureAnswerPlanRow) -> dict[str, Any]:
    return {
        "eligible": row.question.eligible,
        "eligibility_row_identity_sha256": (
            row.question.eligibility_row_identity_sha256
        ),
        "final_packet_id": row.stage.packet.packet_id,
        "final_prompt_id": row.stage.rendered_prompt.prompt_id,
        "final_prompt_messages_sha256": (
            row.stage.rendered_prompt.messages_sha256
        ),
        "final_prompt_token_proxy": (
            row.stage.rendered_prompt.total_prompt_token_proxy
        ),
        "final_stage_receipt_sha256": row.stage.receipt.receipt_sha256,
        "ordinal": row.source.ordinal,
        "parent_prediction_sha256": row.parent.prediction_sha256,
        "provider_call_planned": row.added,
        "question_id": row.source.packet.question_id,
        "question_part_sha256": row.source.question_part_sha256,
        "question_sha256": row.source.packet.question_sha256,
        "source_closure_question_artifact_sha256": (
            row.question.source_question_artifact_sha256
        ),
        "stage_disposition": row.stage.trace.disposition.value,
        "stage_reason": row.stage.trace.reason,
    }


def _preflight_artifact(plan: _ClosureAnswerPlan) -> dict[str, Any]:
    prompt_population = _prompt_population_projection(plan)
    result: dict[str, Any] = {
        "answer_model": live.DEFAULT_TERRA_CALLER_MODEL,
        "answer_plan_id": plan.answer_plan_id,
        "arm_label": plan.arm_label,
        "arm_plan_id": plan.arm_plan.plan_id,
        "format": ANSWER_PREFLIGHT_FORMAT,
        "gold_loaded": False,
        "hard_prompt_token_cap": MAX_FINAL_PROMPT_TOKENS,
        "logical_prompt_count": len(plan.added_rows),
        "matched_population_id": plan.population.population_id,
        "new_provider_calls": 0,
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
        "snapshot": plan.snapshot.projection(),
        "snapshot_id": plan.snapshot.snapshot_id,
        "source_eligibility_manifest_sha256": (
            plan.generation.source_eligibility_manifest_sha256
        ),
        "source_preflight_sha256": plan.generation.preflight_sha256,
        "source_retrieval_generation_sha256": (
            plan.generation.source_retrieval_generation_sha256
        ),
        "unique_prompt_count": plan.required_calls,
    }
    assert_gold_blind(result, path="closure_answer_preflight")
    return result


def _ensure_output_isolation(
    *,
    output_root: str | Path,
    parent_root: str | Path,
    generation_path: str | Path,
) -> Path:
    output = Path(output_root)
    resolved = output.resolve()
    _require(
        resolved != Path(parent_root).resolve(),
        "closure answer output must not reuse the parent S0 root",
    )
    _require(
        resolved != Path(generation_path).parent.resolve(),
        "closure answer output must not reuse the retrieval-generation root",
    )
    return output


def preflight_closure_answers(
    *,
    arm_label: str,
    retrieval_path: str | Path,
    generation_path: str | Path,
    expected_generation_sha256: str,
    eligibility_manifest_path: str | Path,
    expected_eligibility_manifest_sha256: str,
    parent_root: str | Path,
    expected_parent_answer_run_sha256: str,
    output_root: str | Path,
    max_concurrency: int = 4,
    expected_retrieval_sha256: str | None = EXPECTED_RETRIEVAL_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> SealedArtifact:
    plan = _build_plan(
        arm_label=arm_label,
        retrieval_path=retrieval_path,
        generation_path=generation_path,
        expected_generation_sha256=expected_generation_sha256,
        eligibility_manifest_path=eligibility_manifest_path,
        expected_eligibility_manifest_sha256=(
            expected_eligibility_manifest_sha256
        ),
        parent_root=parent_root,
        expected_parent_answer_run_sha256=expected_parent_answer_run_sha256,
        max_concurrency=max_concurrency,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
    )
    output = _ensure_output_isolation(
        output_root=output_root,
        parent_root=parent_root,
        generation_path=generation_path,
    )
    artifact, _created = publish_sealed_json(
        output / ANSWER_PREFLIGHT_NAME,
        _preflight_artifact(plan),
    )
    return artifact


def _runtime(
    plan: _ClosureAnswerPlan,
    *,
    checkpoint_dir: str | Path,
    client: Any | None,
    max_concurrency: int,
    preflight_artifact_sha256: str,
) -> FastCompletionRuntime:
    require_sha256(preflight_artifact_sha256, "closure answer preflight")
    _require(bool(plan.added_rows), "empty closure prompt population has no runtime")
    return FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=[
            _plain_messages(row.stage.rendered_prompt.messages)
            for row in plan.added_rows
        ],
        model=live.DEFAULT_TERRA_GATEWAY_MODEL,
        client=client,
        max_prompt_tokens=MAX_FINAL_PROMPT_TOKENS,
        max_new_tokens=live.DEFAULT_MAX_NEW_TOKENS,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "answer_plan_id": plan.answer_plan_id,
            "arm_label": plan.arm_label,
            "arm_plan_id": plan.arm_plan.plan_id,
            "authorized_unique_calls": plan.required_calls,
            "gateway_url": live.DEFAULT_GATEWAY_URL,
            "gold_loaded": False,
            "parent_answer_run_sha256": plan.parent_plane.run_sha256,
            "preflight_artifact_sha256": preflight_artifact_sha256,
            "renderer_id": RENDERER_ID,
            "retrieval_sha256": plan.population.retrieval_sha256,
            "snapshot_id": plan.snapshot.snapshot_id,
            "source_retrieval_generation_sha256": (
                plan.generation.source_retrieval_generation_sha256
            ),
        },
    )


def _answer_artifact(
    plan: _ClosureAnswerPlan,
    batch: FastCompletionBatch | None,
    *,
    preflight_artifact_sha256: str,
) -> dict[str, Any]:
    if plan.required_calls:
        _require(batch is not None, "closure answer completion batch is missing")
        assert batch is not None
        _require(
            plan.prompt_population is not None
            and batch.prompt_population.prompt_population_sha256
            == plan.prompt_population.prompt_population_sha256,
            "closure Terra prompt population changed",
        )
        completions = iter(batch.logical_completions)
        records = {row.messages_sha256: row for row in batch.unique_records}
    else:
        _require(batch is None, "empty closure answer acquired a completion batch")
        completions = iter(())
        records = {}

    questions: list[dict[str, Any]] = []
    changed = 0
    for row in plan.rows:
        prompt = row.stage.rendered_prompt
        if row.added:
            prediction = next(completions)
            record = records[prompt.messages_sha256]
            _require(
                quote_sha256(prediction) == record.completion_sha256,
                f"closure Terra completion changed at ordinal {row.source.ordinal}",
            )
            prediction_source = "terra_descendant"
            call_key = record.call_key_sha256
            request_journal = record.request_journal_sha256
            response_journal = record.response_journal_sha256
            provider_calls = 1
        else:
            _require(
                row.parent_fallback,
                "closure parent fallback requires a no-op or overflow stage",
            )
            prediction = row.parent.prediction
            prediction_source = "sealed_parent_fallback"
            call_key = request_journal = response_journal = None
            provider_calls = 0
        prediction_sha = quote_sha256(prediction)
        changed_from_parent = prediction_sha != row.parent.prediction_sha256
        changed += changed_from_parent
        body: dict[str, Any] = {
            "call_key_sha256": call_key,
            "changed_from_parent": changed_from_parent,
            "dated_question_sha256": row.source.packet.dated_question_sha256,
            "eligible": row.question.eligible,
            "eligibility_row_identity_sha256": (
                row.question.eligibility_row_identity_sha256
            ),
            "final_packet_id": row.stage.packet.packet_id,
            "final_prompt_id": prompt.prompt_id,
            "final_prompt_messages_sha256": prompt.messages_sha256,
            "final_prompt_token_proxy": prompt.total_prompt_token_proxy,
            "final_stage_receipt_sha256": row.stage.receipt.receipt_sha256,
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
            "source_closure_question_artifact_sha256": (
                row.question.source_question_artifact_sha256
            ),
            "stage_disposition": row.stage.trace.disposition.value,
            "stage_reason": row.stage.trace.reason,
        }
        body["source_row_sha256"] = identity_sha256(body)
        questions.append(body)
    try:
        next(completions)
    except StopIteration:
        pass
    else:
        raise MatchedEvalContractError("closure Terra completion count changed")

    prompt_population = _prompt_population_projection(plan)
    stable_batch = None if batch is None else live._stable_batch(batch)
    if stable_batch is not None:
        _require(
            stable_batch["provenance"]["retained_transformer_token_state_bytes"]
            == 0,
            "closure answer retained transformer token state",
        )
    result: dict[str, Any] = {
        "added_descendant_count": plan.required_calls,
        "answer_model": live.DEFAULT_TERRA_CALLER_MODEL,
        "answer_plan_id": plan.answer_plan_id,
        "arm_label": plan.arm_label,
        "arm_plan_id": plan.arm_plan.plan_id,
        "changed_prediction_count": changed,
        "completion_batch": stable_batch,
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
            "max_prompt_tokens": MAX_FINAL_PROMPT_TOKENS,
            "retries": 0,
        },
        "question_count": len(questions),
        "questions": questions,
        "renderer_id": RENDERER_ID,
        "retained_request_token_state_bytes": 0,
        "retrieval_sha256": plan.population.retrieval_sha256,
        "snapshot_id": plan.snapshot.snapshot_id,
        "source_eligibility_manifest_sha256": (
            plan.generation.source_eligibility_manifest_sha256
        ),
        "source_preflight_sha256": plan.generation.preflight_sha256,
        "source_retrieval_generation_sha256": (
            plan.generation.source_retrieval_generation_sha256
        ),
        "unique_provider_prompt_count": plan.required_calls,
    }
    assert_gold_blind(result, path="closure_answer_run")
    return result


def _runtime_entries(
    plan: _ClosureAnswerPlan,
    answer_payload: Mapping[str, Any],
) -> tuple[RuntimeLedgerEntry, ...]:
    raw_rows = answer_payload.get("questions")
    _require(type(raw_rows) is list, "closure answer questions must be an array")
    _require(
        len(raw_rows) == len(plan.rows),
        "closure answer/runtime population changed",
    )
    entries: list[RuntimeLedgerEntry] = []
    answer_stage_id = _ANSWER_STAGE_IDS[plan.arm_label]
    for source, raw in zip(plan.rows, raw_rows, strict=True):
        _require(
            source.added or source.parent_fallback,
            "closure runtime refuses an unsafe stage disposition",
        )
        _require(type(raw) is dict, "closure answer row must be an exact object")
        unsigned = dict(raw)
        source_row_sha = unsigned.pop("source_row_sha256", None)
        _require(
            source_row_sha == identity_sha256(unsigned),
            f"closure answer row seal changed at ordinal {source.source.ordinal}",
        )
        _require(
            raw.get("ordinal") == source.source.ordinal
            and raw.get("question_id") == source.source.packet.question_id
            and raw.get("question_sha256")
            == source.source.packet.question_sha256
            and raw.get("final_packet_id") == source.stage.packet.packet_id
            and raw.get("final_stage_receipt_sha256")
            == source.stage.receipt.receipt_sha256,
            f"closure answer row binding changed at ordinal {source.source.ordinal}",
        )
        prediction = raw.get("prediction")
        prediction_sha = raw.get("prediction_sha256")
        _require(
            type(prediction) is str
            and bool(prediction)
            and prediction_sha == quote_sha256(prediction),
            f"closure prediction changed at ordinal {source.source.ordinal}",
        )
        provider_calls = raw.get("provider_calls")
        _require(
            type(provider_calls) is int
            and provider_calls == int(source.added),
            f"closure provider-call attribution changed at {source.source.ordinal}",
        )
        entries.append(
            runtime_entry_from_stage_run(
                ordinal=source.source.ordinal,
                arm_label=plan.arm_label,
                parent_arm_label=PARENT_ARM_LABEL,
                run=source.run,
                stage_id=source.stage.stage_id,
            )
        )
        entries.append(
            RuntimeLedgerEntry(
                event_type="answer_observation",
                ordinal=source.source.ordinal,
                question_id=source.source.packet.question_id,
                question_sha256=source.source.packet.question_sha256,
                arm_label=plan.arm_label,
                parent_arm_label=PARENT_ARM_LABEL,
                stage_id=answer_stage_id,
                parent_stage_id=source.stage.stage_id,
                mechanism_id=(
                    "terra_responder"
                    if source.added
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
                max_final_prompt_tokens=MAX_FINAL_PROMPT_TOKENS,
                prompt_token_proxy=(
                    source.stage.rendered_prompt.total_prompt_token_proxy
                ),
                parent_packet_sha256=source.stage.packet.packet_id,
                packet_sha256=source.stage.packet.packet_id,
                prompt_id=source.stage.rendered_prompt.prompt_id,
                prompt_messages_sha256=(
                    source.stage.rendered_prompt.messages_sha256
                ),
                prediction=prediction,
                prediction_sha256=prediction_sha,
                changed_from_parent=raw.get("changed_from_parent"),
                source_row_sha256=str(source_row_sha),
                reason=(
                    "sealed_terra_independent_closure_prediction"
                    if source.added
                    else "sealed_s0_v2_parent_prediction_reuse"
                ),
            )
        )
    return tuple(entries)


def _runtime_ledger(
    plan: _ClosureAnswerPlan,
    answer_payload: Mapping[str, Any],
    *,
    answer_artifact_sha256: str,
    preflight_artifact_sha256: str,
) -> dict[str, Any]:
    return build_runtime_ledger(
        snapshot_id=plan.snapshot.snapshot_id,
        plan_id=plan.answer_plan_id,
        entries=_runtime_entries(plan, answer_payload),
        source_artifacts=(
            {
                "role": f"{plan.arm_label}:sealed_retrieval",
                "sha256": plan.population.retrieval_sha256,
            },
            {
                "role": f"{plan.arm_label}:closure_generation",
                "sha256": plan.generation.source_retrieval_generation_sha256,
            },
            {
                "role": f"{plan.arm_label}:eligibility_manifest",
                "sha256": plan.generation.source_eligibility_manifest_sha256,
            },
            {
                "role": f"{plan.arm_label}:parent_answer_run",
                "sha256": plan.parent_plane.run_sha256,
            },
            {
                "role": f"{plan.arm_label}:answer_preflight",
                "sha256": preflight_artifact_sha256,
            },
            {
                "role": f"{plan.arm_label}:answer_run",
                "sha256": answer_artifact_sha256,
            },
        ),
    )


def _verified_plane(
    *,
    plan: _ClosureAnswerPlan,
    run: SealedArtifact,
    replay: SealedArtifact,
    runtime_ledger: SealedArtifact,
) -> VerifiedClosureAnswerPlane:
    payload = run.payload
    _require(
        run.sha256 == replay.sha256
        and canonical_json_bytes(payload) == canonical_json_bytes(replay.payload),
        "closure answer run/replay differ",
    )
    _require(
        payload.get("format") == ANSWER_RUN_FORMAT
        and payload.get("arm_label") == plan.arm_label
        and payload.get("parent_arm_label") == PARENT_ARM_LABEL
        and payload.get("parent_answer_run_sha256")
        == plan.parent_plane.run_sha256
        and payload.get("snapshot_id") == plan.snapshot.snapshot_id
        and payload.get("arm_plan_id") == plan.arm_plan.plan_id
        and payload.get("answer_plan_id") == plan.answer_plan_id
        and payload.get("renderer_id") == RENDERER_ID,
        "closure verified answer envelope changed",
    )
    ledger_identity, answer_row_ids = _validated_runtime_ledger(
        runtime_ledger.payload
    )
    _require(
        runtime_ledger.payload.get("snapshot_id") == plan.snapshot.snapshot_id
        and runtime_ledger.payload.get("plan_id") == plan.answer_plan_id
        and len(answer_row_ids) == len(plan.rows),
        "closure verified runtime ledger changed",
    )
    require_sha256(ledger_identity, "closure runtime ledger identity")
    raw_rows = payload.get("questions")
    ledger_rows = [
        row
        for row in runtime_ledger.payload["rows"]
        if row["event_type"] == "answer_observation"
    ]
    _require(
        type(raw_rows) is list
        and len(raw_rows) == len(ledger_rows) == len(plan.rows),
        "closure verified answer population changed",
    )
    rows: list[VerifiedClosureAnswerRow] = []
    for source, raw, ledger, runtime_row_id in zip(
        plan.rows,
        raw_rows,
        ledger_rows,
        answer_row_ids,
        strict=True,
    ):
        _require(type(raw) is dict, "closure verified answer row changed")
        unsigned = dict(raw)
        source_row_sha = unsigned.pop("source_row_sha256", None)
        _require(
            source_row_sha == identity_sha256(unsigned)
            and ledger.get("source_row_sha256") == source_row_sha
            and ledger.get("row_id") == runtime_row_id
            and ledger.get("prediction_sha256") == raw.get("prediction_sha256"),
            f"closure answer/runtime binding changed at {source.source.ordinal}",
        )
        prediction = raw.get("prediction")
        prediction_sha = raw.get("prediction_sha256")
        changed_from_parent = raw.get("changed_from_parent")
        _require(
            type(prediction) is str
            and prediction_sha == quote_sha256(prediction),
            f"closure verified prediction changed at {source.source.ordinal}",
        )
        _require(
            type(changed_from_parent) is bool
            and changed_from_parent
            == (prediction_sha != source.parent.prediction_sha256),
            f"closure parent-change flag changed at {source.source.ordinal}",
        )
        call_key = raw.get("call_key_sha256")
        request_journal = raw.get("request_journal_sha256")
        response_journal = raw.get("response_journal_sha256")
        if source.added:
            for value, label in (
                (call_key, "call key"),
                (request_journal, "request journal"),
                (response_journal, "response journal"),
            ):
                require_sha256(str(value), label)
            _require(
                raw.get("prediction_source") == "terra_descendant",
                "closure added row lost Terra provenance",
            )
        else:
            _require(
                source.parent_fallback,
                "closure verification refuses an unsafe parent fallback",
            )
            _require(
                call_key is None
                and request_journal is None
                and response_journal is None
                and raw.get("prediction_source") == "sealed_parent_fallback"
                and prediction == source.parent.prediction,
                "closure fallback changed its sealed parent prediction",
            )
        rows.append(
            VerifiedClosureAnswerRow(
                ordinal=source.source.ordinal,
                question_id=source.source.packet.question_id,
                question_sha256=source.source.packet.question_sha256,
                dated_question_sha256=(
                    source.source.packet.dated_question_sha256
                ),
                prediction=prediction,
                prediction_sha256=str(prediction_sha),
                prediction_source=str(raw["prediction_source"]),
                parent_prediction_sha256=source.parent.prediction_sha256,
                changed_from_parent=changed_from_parent,
                stage_disposition=source.stage.trace.disposition.value,
                final_packet_id=source.stage.packet.packet_id,
                final_prompt_id=source.stage.rendered_prompt.prompt_id,
                final_prompt_messages_sha256=(
                    source.stage.rendered_prompt.messages_sha256
                ),
                final_stage_receipt_sha256=source.stage.receipt.receipt_sha256,
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
    return VerifiedClosureAnswerPlane(
        run_sha256=run.sha256,
        replay_sha256=replay.sha256,
        runtime_ledger_sha256=runtime_ledger.sha256,
        runtime_ledger=live._freeze_json(runtime_ledger.payload),
        arm_label=plan.arm_label,
        parent_arm_label=PARENT_ARM_LABEL,
        parent_answer_run_sha256=plan.parent_plane.run_sha256,
        matched_population_id=plan.population.population_id,
        population_identity_sha256=(
            plan.population.snapshot.population_identity_sha256
        ),
        retrieval_sha256=plan.population.retrieval_sha256,
        source_retrieval_generation_sha256=(
            plan.generation.source_retrieval_generation_sha256
        ),
        source_eligibility_manifest_sha256=(
            plan.generation.source_eligibility_manifest_sha256
        ),
        source_preflight_sha256=plan.generation.preflight_sha256,
        snapshot_id=plan.snapshot.snapshot_id,
        arm_plan_id=plan.arm_plan.plan_id,
        answer_plan_id=plan.answer_plan_id,
        renderer_id=RENDERER_ID,
        rows=tuple(rows),
        parent_plane=plan.parent_plane,
    )


def _replay_plan(
    plan: _ClosureAnswerPlan,
    *,
    source: SealedArtifact,
    expected_run_sha256: str,
    output_root: str | Path,
    max_concurrency: int,
) -> VerifiedClosureAnswerPlane:
    _require(source.sha256 == expected_run_sha256, "closure answer run changed")
    output = Path(output_root)
    preflight = read_sealed_json(output / ANSWER_PREFLIGHT_NAME)
    _require(
        preflight.payload == _preflight_artifact(plan),
        "closure answer preflight changed during replay",
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
        _require(batch.usage.physical_calls == 0, "closure replay made provider calls")
        _require(
            batch.usage.checkpoint_hits == plan.required_calls,
            "closure replay checkpoint population changed",
        )
    expected = _answer_artifact(
        plan,
        batch,
        preflight_artifact_sha256=preflight.sha256,
    )
    _require(
        canonical_json_bytes(expected) == canonical_json_bytes(source.payload),
        "closure answer differs from immutable Terra journals",
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
        "closure runtime ledger differs from replayed answers",
    )
    publish_sealed_json(output / RUNTIME_LEDGER_REPLAY_NAME, expected_ledger)
    return _verified_plane(
        plan=plan,
        run=source,
        replay=replay,
        runtime_ledger=ledger,
    )


def run_closure_answers(
    *,
    arm_label: str,
    retrieval_path: str | Path,
    generation_path: str | Path,
    expected_generation_sha256: str,
    eligibility_manifest_path: str | Path,
    expected_eligibility_manifest_sha256: str,
    parent_root: str | Path,
    expected_parent_answer_run_sha256: str,
    output_root: str | Path,
    enable_provider: bool,
    authorized_provider_calls: int,
    api_key_env: str = live.DEFAULT_API_KEY_ENV,
    max_concurrency: int = 4,
    expected_retrieval_sha256: str | None = EXPECTED_RETRIEVAL_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> ClosureAnswerRunResult:
    plan = _build_plan(
        arm_label=arm_label,
        retrieval_path=retrieval_path,
        generation_path=generation_path,
        expected_generation_sha256=expected_generation_sha256,
        eligibility_manifest_path=eligibility_manifest_path,
        expected_eligibility_manifest_sha256=(
            expected_eligibility_manifest_sha256
        ),
        parent_root=parent_root,
        expected_parent_answer_run_sha256=expected_parent_answer_run_sha256,
        max_concurrency=max_concurrency,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
    )
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == plan.required_calls,
        "authorized closure answer provider calls must exactly equal "
        f"{plan.required_calls}",
    )
    if plan.required_calls:
        _require(enable_provider, "closure answer run requires provider enablement")
    else:
        _require(
            not enable_provider,
            "empty closure answer run forbids provider enablement",
        )
    output = _ensure_output_isolation(
        output_root=output_root,
        parent_root=parent_root,
        generation_path=generation_path,
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
        return ClosureAnswerRunResult(
            answer_artifact=source,
            runtime_ledger_artifact=read_sealed_json(
                output / RUNTIME_LEDGER_NAME
            ),
            physical_provider_calls=0,
            checkpoint_hits=plan.required_calls,
        )

    batch: FastCompletionBatch | None = None
    client: Any | None = None
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
            "closure Terra journal population changed",
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
    return ClosureAnswerRunResult(
        answer_artifact=answer,
        runtime_ledger_artifact=ledger,
        physical_provider_calls=0 if batch is None else batch.usage.physical_calls,
        checkpoint_hits=0 if batch is None else batch.usage.checkpoint_hits,
    )


def replay_closure_answers(
    *,
    arm_label: str,
    retrieval_path: str | Path,
    generation_path: str | Path,
    expected_generation_sha256: str,
    eligibility_manifest_path: str | Path,
    expected_eligibility_manifest_sha256: str,
    parent_root: str | Path,
    expected_parent_answer_run_sha256: str,
    output_root: str | Path,
    expected_run_sha256: str,
    max_concurrency: int = 4,
    expected_retrieval_sha256: str | None = EXPECTED_RETRIEVAL_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> VerifiedClosureAnswerPlane:
    require_sha256(expected_run_sha256, "expected closure answer run")
    plan = _build_plan(
        arm_label=arm_label,
        retrieval_path=retrieval_path,
        generation_path=generation_path,
        expected_generation_sha256=expected_generation_sha256,
        eligibility_manifest_path=eligibility_manifest_path,
        expected_eligibility_manifest_sha256=(
            expected_eligibility_manifest_sha256
        ),
        parent_root=parent_root,
        expected_parent_answer_run_sha256=expected_parent_answer_run_sha256,
        max_concurrency=max_concurrency,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
    )
    output = _ensure_output_isolation(
        output_root=output_root,
        parent_root=parent_root,
        generation_path=generation_path,
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
    "ANSWER_PREFLIGHT_FORMAT",
    "ANSWER_PREFLIGHT_NAME",
    "ANSWER_REPLAY_NAME",
    "ANSWER_RUN_FORMAT",
    "ANSWER_RUN_NAME",
    "CHECKPOINT_DIR_NAME",
    "ClosureAnswerRunResult",
    "RUNTIME_LEDGER_NAME",
    "RUNTIME_LEDGER_REPLAY_NAME",
    "VerifiedClosureAnswerPlane",
    "VerifiedClosureAnswerRow",
    "preflight_closure_answers",
    "replay_closure_answers",
    "run_closure_answers",
]
