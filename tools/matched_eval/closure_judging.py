"""Changed-only Sol judging for verified independent-closure answers.

The closure answer plane already binds every candidate prediction to its
sealed S0-v2 parent.  This adapter judges only prediction hashes that changed
and reuses the fully verified parent verdict for every unchanged row.  The
selection boundary is structural and is sealed before gold or parent judge
outcomes are loaded.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

from memory_condense.domain._tokenizer import count_tokens, tokenizer_proxy_identity
from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval._binary_judge_protocol import (
    JUDGE_MAX_TOKENS,
    parse_binary_judge_verdict,
)
from memory_condense.eval.benchmark import build_judge_prompt, exact_match, f1_score
from memory_condense.eval.fast_completion_runtime import (
    FAST_COMPLETION_REQUEST_FORMAT,
    FAST_COMPLETION_RESPONSE_FORMAT,
    FAST_COMPLETION_RUNTIME_FORMAT,
    FastCompletionBatch,
    FastCompletionRuntime,
    FastPromptPopulation,
    preflight_fast_completion_prompts,
)
from tools._routed_repair_routing import route_question

from . import judging
from .artifacts import SealedArtifact, publish_sealed_json, read_sealed_json
from .closure import (
    ARM_LABELS,
    GLOBAL_ARM,
    REPRESENTATIVE_ARM,
    independent_closure_arm_plan,
)
from .closure_live import (
    PARENT_ARM_LABEL,
    VerifiedClosureAnswerPlane,
    VerifiedClosureAnswerRow,
    _ANSWER_PLAN_IDS,
)
from .contracts import (
    MatchedEvalContractError,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from .ledger import (
    SCORE_LEDGER_FORMAT,
    ScoreLedgerEntry,
    _validated_runtime_ledger,
    build_score_ledger,
)
from .live import DEFAULT_API_KEY_ENV, DEFAULT_GATEWAY_URL, VerifiedS0V2AnswerPlane
from .population import EXPECTED_QUESTION_COUNT
from .renderer import RENDERER_ID


CLOSURE_JUDGE_FORMAT = (
    "memory-condense-matched-independent-closure-changed-only-sol-judge-v1"
)
CLOSURE_JUDGE_PREFLIGHT_FORMAT = (
    "memory-condense-matched-independent-closure-changed-only-sol-preflight-v1"
)
EMPTY_PROMPT_POPULATION_FORMAT = (
    "memory-condense-matched-independent-closure-changed-only-empty-prompts-v1"
)

JUDGE_PREFLIGHT_NAME = judging.JUDGE_PREFLIGHT_NAME
JUDGE_NAME = judging.JUDGE_NAME
JUDGE_REPLAY_NAME = judging.JUDGE_REPLAY_NAME
SCORE_LEDGER_NAME = judging.SCORE_LEDGER_NAME
SCORE_LEDGER_REPLAY_NAME = judging.SCORE_LEDGER_REPLAY_NAME
JUDGE_CHECKPOINT_DIR_NAME = judging.JUDGE_CHECKPOINT_DIR_NAME

_PLAN_IDS = {
    REPRESENTATIVE_ARM: (
        "matched_representative_bridge_closure_changed_only_sol_judge_v1"
    ),
    GLOBAL_ARM: "matched_artifact_global_closure_changed_only_sol_judge_v1",
}

_PARENT_JUDGE_ROW_KEYS = frozenset(
    {
        "call_key_sha256",
        "category",
        "correct",
        "dated_question_sha256",
        "demand_class",
        "judge_messages_sha256",
        "judge_output",
        "judge_output_sha256",
        "judge_prompt_token_proxy",
        "judge_row_sha256",
        "normalized_exact_match",
        "normalized_f1",
        "ordinal",
        "prediction_sha256",
        "question_id",
        "question_sha256",
        "reference_sha256",
        "request_journal_sha256",
        "response_journal_sha256",
        "runtime_row_id",
    }
)
_PARENT_JUDGE_KEYS = frozenset(
    {
        "aggregate",
        "answer_run_sha256",
        "category_aggregates",
        "completion_batch",
        "format",
        "gold_loaded_posthoc",
        "gold_population_sha256",
        "judge_completions_may_echo_gold",
        "judge_model",
        "preflight_artifact_sha256",
        "prompt_population_sha256",
        "question_count",
        "questions",
        "retained_request_token_state_bytes",
        "runtime_ledger_identity_sha256",
        "runtime_ledger_sha256",
        "unique_provider_prompt_count",
    }
)
_STABLE_BATCH_KEYS = frozenset(
    {
        "logical_completions",
        "prompt_population",
        "provenance",
        "runtime_identity_sha256",
        "unique_records",
        "usage",
    }
)
_STABLE_PROVENANCE_KEYS = frozenset(
    {
        "benchmark_provenance",
        "external_provider_persistence_certified",
        "format",
        "max_concurrency",
        "max_new_tokens",
        "max_prompt_token_proxy",
        "model",
        "persisted_transformer_token_state",
        "prompt_population_sha256",
        "prompt_token_proxy_identity",
        "request_options",
        "retained_transformer_token_state_bytes",
        "retries",
    }
)
_STABLE_RECORD_KEYS = frozenset(
    {
        "call_key_sha256",
        "completion",
        "completion_sha256",
        "completion_token_proxy",
        "finish_reason",
        "messages_sha256",
        "prompt_token_proxy",
        "provider_elapsed_s",
        "reported_completion_tokens",
        "reported_prompt_tokens",
        "reported_total_tokens",
        "request_journal_sha256",
        "requested_model",
        "response_id",
        "response_journal_sha256",
        "response_model",
    }
)
_STABLE_USAGE_KEYS = frozenset(
    {
        "completion_token_proxy",
        "deduplicated_logical_calls",
        "logical_calls",
        "prompt_token_proxy",
        "recorded_provider_elapsed_s",
        "recorded_reported_completion_tokens",
        "recorded_reported_prompt_tokens",
        "recorded_reported_total_tokens",
        "reported_completion_tokens_complete",
        "reported_prompt_tokens_complete",
        "reported_total_tokens_complete",
        "unique_calls",
    }
)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise MatchedEvalContractError(message)


@dataclass(frozen=True, slots=True)
class _ClosureJudgeProfile:
    arm_label: str
    plan_id: str


@dataclass(frozen=True, slots=True)
class _ParentOutcome:
    ordinal: int
    correct: bool
    judge_row_sha256: str
    judge_verdict_sha256: str
    demand_class: str


@dataclass(frozen=True, slots=True)
class _VerifiedParentJudge:
    preflight_sha256: str
    judge_sha256: str
    score_ledger_sha256: str
    outcomes: tuple[_ParentOutcome, ...]


@dataclass(frozen=True, slots=True)
class _ClosureJudgePlan:
    answer_plane: VerifiedClosureAnswerPlane
    gold_rows: tuple[judging._GoldRow, ...]
    gold_population_sha256: str
    change_projection: Mapping[str, Any]
    prompt_rows: tuple[judging._JudgePromptRow, ...]
    prompt_population: FastPromptPopulation | None
    parent_judge: _VerifiedParentJudge
    profile: _ClosureJudgeProfile

    @property
    def required_calls(self) -> int:
        return 0 if self.prompt_population is None else (
            self.prompt_population.unique_prompt_count
        )

    @property
    def changed_count(self) -> int:
        return len(self.prompt_rows)


@dataclass(frozen=True, slots=True)
class ClosureJudgeRunResult:
    judge_artifact: SealedArtifact
    score_ledger_artifact: SealedArtifact
    correct: int
    physical_provider_calls: int
    checkpoint_hits: int


def _profile(arm_label: str) -> _ClosureJudgeProfile:
    _require(arm_label in ARM_LABELS, f"unsupported closure arm: {arm_label!r}")
    return _ClosureJudgeProfile(arm_label=arm_label, plan_id=_PLAN_IDS[arm_label])


def _load_gold(
    *,
    dataset_path: str | Path,
    split_path: str | Path,
    answer_plane: VerifiedClosureAnswerPlane,
) -> tuple[tuple[judging._GoldRow, ...], str]:
    """Neutral seam retained so tests and callers can provide a sealed gold loader."""

    return judging._load_gold(
        dataset_path=dataset_path,
        split_path=split_path,
        answer_plane=answer_plane,  # type: ignore[arg-type]
    )


def _validate_answer_plane(
    answer_plane: VerifiedClosureAnswerPlane,
    *,
    profile: _ClosureJudgeProfile,
    expected_question_count: int,
) -> dict[str, Any]:
    """Verify the child/parent structural boundary without loading gold."""

    _require(
        type(answer_plane) is VerifiedClosureAnswerPlane,
        "closure judge requires an exact verified closure answer plane",
    )
    _require(
        type(expected_question_count) is int and expected_question_count > 0,
        "expected question count must be a positive exact integer",
    )
    _require(
        answer_plane.arm_label == profile.arm_label
        and answer_plane.arm_label in ARM_LABELS,
        "closure judge arm profile changed",
    )
    _require(
        answer_plane.parent_arm_label == PARENT_ARM_LABEL,
        "closure parent arm changed",
    )
    _require(
        answer_plane.renderer_id == RENDERER_ID,
        "closure answer renderer changed",
    )
    for value, label in (
        (answer_plane.run_sha256, "closure answer run SHA-256"),
        (answer_plane.replay_sha256, "closure answer replay SHA-256"),
        (answer_plane.runtime_ledger_sha256, "closure runtime-ledger SHA-256"),
        (answer_plane.matched_population_id, "closure matched population ID"),
        (
            answer_plane.population_identity_sha256,
            "closure population identity SHA-256",
        ),
        (answer_plane.retrieval_sha256, "closure retrieval SHA-256"),
        (
            answer_plane.source_retrieval_generation_sha256,
            "closure generation SHA-256",
        ),
        (
            answer_plane.source_eligibility_manifest_sha256,
            "closure eligibility SHA-256",
        ),
        (answer_plane.source_preflight_sha256, "closure source preflight SHA-256"),
        (answer_plane.snapshot_id, "closure snapshot ID"),
    ):
        require_sha256(value, label)
    _require(
        answer_plane.arm_plan_id
        == independent_closure_arm_plan(profile.arm_label).plan_id,
        "closure arm plan ID changed",
    )
    _require(
        answer_plane.answer_plan_id == _ANSWER_PLAN_IDS[profile.arm_label],
        "closure answer plan ID changed",
    )
    _require(
        answer_plane.run_sha256 == answer_plane.replay_sha256,
        "closure answer run and replay differ",
    )

    parent = answer_plane.parent_plane
    _require(
        type(parent) is VerifiedS0V2AnswerPlane,
        "closure answer plane lost its exact verified S0-v2 parent",
    )
    judging._validate_preverified_answer_plane(
        parent, profile=judging._V2_JUDGE_PROFILE
    )
    _require(
        answer_plane.parent_answer_run_sha256 == parent.run_sha256
        and parent.run_sha256 == parent.replay_sha256,
        "closure parent answer-run binding changed",
    )
    _require(
        answer_plane.matched_population_id == parent.matched_population_id
        and answer_plane.population_identity_sha256
        == parent.population_identity_sha256
        and answer_plane.renderer_id == parent.renderer_id,
        "closure and parent populations differ",
    )
    # Closure snapshots legitimately add a sealed overlay; only the common
    # source-population identity is required to match the parent snapshot.
    _require(
        type(answer_plane.rows) is tuple
        and len(answer_plane.rows) == len(parent.rows) == expected_question_count,
        "closure judge population size changed",
    )

    runtime = answer_plane.runtime_ledger_json()
    ledger_identity, answer_row_ids = _validated_runtime_ledger(runtime)
    _require(
        runtime.get("snapshot_id") == answer_plane.snapshot_id
        and runtime.get("plan_id") == answer_plane.answer_plan_id,
        "closure runtime envelope changed",
    )
    _require(
        answer_row_ids == tuple(row.runtime_row_id for row in answer_plane.rows),
        "closure runtime answer-row order changed",
    )
    source_map = {
        row["role"]: row["sha256"] for row in runtime["source_artifacts"]
    }
    for role, expected in (
        (f"{profile.arm_label}:sealed_retrieval", answer_plane.retrieval_sha256),
        (
            f"{profile.arm_label}:closure_generation",
            answer_plane.source_retrieval_generation_sha256,
        ),
        (
            f"{profile.arm_label}:eligibility_manifest",
            answer_plane.source_eligibility_manifest_sha256,
        ),
        (f"{profile.arm_label}:parent_answer_run", parent.run_sha256),
        (f"{profile.arm_label}:answer_run", answer_plane.run_sha256),
    ):
        _require(source_map.get(role) == expected, f"closure runtime lost {role}")

    raw_answers = tuple(
        row for row in runtime["rows"] if row["event_type"] == "answer_observation"
    )
    projection_rows: list[dict[str, Any]] = []
    ordinals: list[int] = []
    question_ids: list[str] = []
    for child, parent_row, raw in zip(
        answer_plane.rows, parent.rows, raw_answers, strict=True
    ):
        _require(
            type(child) is VerifiedClosureAnswerRow,
            "closure answer rows must have the exact verified type",
        )
        _require(
            type(child.ordinal) is int and child.ordinal >= 0,
            "closure answer ordinal is invalid",
        )
        require_text(child.question_id, "closure question ID")
        for value, label in (
            (child.question_sha256, "closure question SHA-256"),
            (child.dated_question_sha256, "closure dated-question SHA-256"),
            (child.prediction_sha256, "closure prediction SHA-256"),
            (child.parent_prediction_sha256, "closure parent prediction SHA-256"),
            (child.final_packet_id, "closure final packet ID"),
            (child.final_prompt_id, "closure final prompt ID"),
            (child.final_prompt_messages_sha256, "closure prompt SHA-256"),
            (child.final_stage_receipt_sha256, "closure stage receipt SHA-256"),
            (child.source_row_sha256, "closure source row SHA-256"),
            (child.runtime_row_id, "closure runtime row ID"),
        ):
            require_sha256(value, label)
        _require(
            type(child.prediction) is str
            and bool(child.prediction)
            and quote_sha256(child.prediction) == child.prediction_sha256,
            f"closure prediction changed at ordinal {child.ordinal}",
        )
        _require(
            child.ordinal == parent_row.ordinal
            and child.question_id == parent_row.question_id
            and child.question_sha256 == parent_row.question_sha256
            and child.dated_question_sha256 == parent_row.dated_question_sha256
            and child.parent_prediction_sha256 == parent_row.prediction_sha256,
            f"closure parent row binding changed at ordinal {child.ordinal}",
        )
        changed = child.prediction_sha256 != parent_row.prediction_sha256
        _require(
            type(child.changed_from_parent) is bool
            and child.changed_from_parent == changed,
            f"closure change flag changed at ordinal {child.ordinal}",
        )
        _require(
            raw.get("row_id") == child.runtime_row_id
            and raw.get("ordinal") == child.ordinal
            and raw.get("question_id") == child.question_id
            and raw.get("question_sha256") == child.question_sha256
            and raw.get("arm_label") == profile.arm_label
            and raw.get("parent_arm_label") == PARENT_ARM_LABEL
            and raw.get("renderer_id") == RENDERER_ID
            and raw.get("prompt_messages_sha256")
            == child.final_prompt_messages_sha256
            and raw.get("prediction") == child.prediction
            and raw.get("prediction_sha256") == child.prediction_sha256
            and raw.get("changed_from_parent") == changed
            and raw.get("source_row_sha256") == child.source_row_sha256,
            f"closure answer/runtime binding changed at ordinal {child.ordinal}",
        )
        if child.prediction_source == "sealed_parent_fallback":
            _require(
                not changed
                and child.prediction == parent_row.prediction
                and child.call_key_sha256 is None
                and child.request_journal_sha256 is None
                and child.response_journal_sha256 is None,
                f"closure parent fallback changed at ordinal {child.ordinal}",
            )
        elif child.prediction_source == "terra_descendant":
            _require(
                child.stage_disposition == "added",
                f"closure Terra row was not admitted at ordinal {child.ordinal}",
            )
            for value, label in (
                (child.call_key_sha256, "closure Terra call key"),
                (child.request_journal_sha256, "closure Terra request journal"),
                (child.response_journal_sha256, "closure Terra response journal"),
            ):
                require_sha256(str(value), label)
        else:
            raise MatchedEvalContractError(
                f"unknown closure prediction source at ordinal {child.ordinal}"
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
                "runtime_row_id": child.runtime_row_id,
            }
        )
    _require(
        tuple(ordinals) == tuple(sorted(set(ordinals))),
        "closure answer row order changed",
    )
    _require(
        len(set(question_ids)) == len(question_ids),
        "closure question IDs must be unique",
    )
    body: dict[str, Any] = {
        "answer_run_sha256": answer_plane.run_sha256,
        "arm_label": profile.arm_label,
        "format": "memory-condense-matched-closure-change-projection-v1",
        "parent_answer_run_sha256": parent.run_sha256,
        "population_identity_sha256": answer_plane.population_identity_sha256,
        "rows": projection_rows,
        "runtime_ledger_identity_sha256": ledger_identity,
    }
    body["change_projection_sha256"] = identity_sha256(body)
    return body


def _prompt_plan(
    answer_plane: VerifiedClosureAnswerPlane,
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
                gold.question, gold.reference, source.prediction
            )
        )
        demand_class = route_question(gold.dated_question).style.value
        pending.append((source.ordinal, messages, demand_class))
        prompts.append(messages)
    if not prompts:
        return (), None
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=judging.DEFAULT_MAX_JUDGE_PROMPT_TOKENS
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
            pending, population.ordered_rows, strict=True
        )
    )
    return rows, population


def _parent_plan(
    parent: VerifiedS0V2AnswerPlane,
    gold_rows: tuple[judging._GoldRow, ...],
    gold_population_sha256: str,
) -> judging._JudgePlan:
    pending: list[tuple[int, tuple[dict[str, str], ...], str]] = []
    prompts: list[tuple[dict[str, str], ...]] = []
    for source, gold in zip(parent.rows, gold_rows, strict=True):
        messages = tuple(
            dict(message)
            for message in build_judge_prompt(
                gold.question, gold.reference, source.prediction
            )
        )
        pending.append(
            (source.ordinal, messages, route_question(gold.dated_question).style.value)
        )
        prompts.append(messages)
    population = preflight_fast_completion_prompts(
        prompts, max_prompt_tokens=judging.DEFAULT_MAX_JUDGE_PROMPT_TOKENS
    )
    _require(
        population.logical_prompt_count == population.unique_prompt_count
        == len(parent.rows),
        "sealed parent judge prompts are no longer unique",
    )
    prompt_rows = tuple(
        judging._JudgePromptRow(
            answer_ordinal=ordinal,
            messages=messages,
            messages_sha256=receipt.messages_sha256,
            prompt_token_proxy=receipt.prompt_token_proxy,
            demand_class=demand,
        )
        for (ordinal, messages, demand), receipt in zip(
            pending, population.ordered_rows, strict=True
        )
    )
    return judging._JudgePlan(
        answer_plane=parent,
        gold_rows=gold_rows,
        prompt_rows=prompt_rows,
        prompt_population=population,
        gold_population_sha256=gold_population_sha256,
        profile=judging._V2_JUDGE_PROFILE,
    )


def _validate_parent_completion_batch(
    batch: object,
    *,
    plan: judging._JudgePlan,
    preflight_sha256: str,
    raw_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Validate the embedded stable batch without reopening 100 journals."""

    _require(
        type(batch) is dict and set(batch) == _STABLE_BATCH_KEYS,
        "sealed parent completion batch schema changed",
    )
    assert type(batch) is dict
    expected_population = plan.prompt_population.model_dump()
    _require(
        batch.get("prompt_population") == expected_population,
        "sealed parent completion prompt population changed",
    )
    provenance = batch.get("provenance")
    _require(
        type(provenance) is dict
        and set(provenance) == _STABLE_PROVENANCE_KEYS,
        "sealed parent completion provenance schema changed",
    )
    assert type(provenance) is dict
    expected_benchmark_provenance = {
        "answer_run_sha256": plan.answer_plane.run_sha256,
        "arm_label": plan.profile.arm_label,
        "authorized_unique_calls": plan.required_calls,
        "gateway_url": DEFAULT_GATEWAY_URL,
        "gold_population_sha256": plan.gold_population_sha256,
        "judge_plan_id": plan.profile.plan_id,
        "preflight_artifact_sha256": preflight_sha256,
        "runtime_ledger_sha256": plan.answer_plane.runtime_ledger_sha256,
    }
    _require(
        provenance.get("format") == FAST_COMPLETION_RUNTIME_FORMAT
        and provenance.get("model") == judging.DEFAULT_SOL_GATEWAY_MODEL
        and type(provenance.get("max_new_tokens")) is int
        and provenance.get("max_new_tokens") == JUDGE_MAX_TOKENS
        and type(provenance.get("max_prompt_token_proxy")) is int
        and provenance.get("max_prompt_token_proxy")
        == judging.DEFAULT_MAX_JUDGE_PROMPT_TOKENS
        and type(provenance.get("max_concurrency")) is int
        and provenance["max_concurrency"] > 0
        and type(provenance.get("retries")) is int
        and provenance.get("retries") == 0
        and type(provenance.get("request_options")) is dict
        and provenance.get("request_options") == {}
        and provenance.get("prompt_population_sha256")
        == plan.prompt_population.prompt_population_sha256
        and provenance.get("prompt_token_proxy_identity")
        == dict(plan.prompt_population.prompt_token_proxy_identity)
        and type(provenance.get("benchmark_provenance")) is dict
        and provenance.get("benchmark_provenance")
        == expected_benchmark_provenance
        and provenance.get("persisted_transformer_token_state") is False
        and type(provenance.get("retained_transformer_token_state_bytes"))
        is int
        and provenance.get("retained_transformer_token_state_bytes") == 0
        and provenance.get("external_provider_persistence_certified") is False,
        "sealed parent completion provenance changed",
    )
    runtime_identity_sha256 = identity_sha256(provenance)
    _require(
        batch.get("runtime_identity_sha256") == runtime_identity_sha256,
        "sealed parent completion runtime identity changed",
    )

    logical = batch.get("logical_completions")
    expected_logical = [str(row["judge_output"]) for row in raw_rows]
    _require(
        type(logical) is list and logical == expected_logical,
        "sealed parent logical completions changed",
    )
    raw_records = batch.get("unique_records")
    _require(
        type(raw_records) is list
        and len(raw_records) == plan.required_calls == len(raw_rows),
        "sealed parent completion record population changed",
    )
    assert type(raw_records) is list
    records: list[dict[str, Any]] = []
    for prompt, row, record in zip(
        plan.prompt_rows, raw_rows, raw_records, strict=True
    ):
        _require(
            type(record) is dict and set(record) == _STABLE_RECORD_KEYS,
            f"sealed parent completion record schema changed at ordinal "
            f"{prompt.answer_ordinal}",
        )
        assert type(record) is dict
        completion = row["judge_output"]
        completion_sha256 = quote_sha256(completion)
        expected_call_key = identity_sha256(
            {
                "format": FAST_COMPLETION_REQUEST_FORMAT,
                "runtime_identity_sha256": runtime_identity_sha256,
                "prompt_population_sha256": (
                    plan.prompt_population.prompt_population_sha256
                ),
                "messages_sha256": prompt.messages_sha256,
                "prompt_token_proxy": prompt.prompt_token_proxy,
                "max_new_tokens": JUDGE_MAX_TOKENS,
            }
        )
        _require(
            record.get("call_key_sha256") == expected_call_key
            and record.get("messages_sha256") == prompt.messages_sha256
            and record.get("completion") == completion
            and record.get("completion_sha256") == completion_sha256
            and record.get("requested_model")
            == judging.DEFAULT_SOL_GATEWAY_MODEL
            and type(record.get("response_id")) is str
            and type(record.get("response_model")) is str
            and record.get("finish_reason") == "stop"
            and type(record.get("prompt_token_proxy")) is int
            and record.get("prompt_token_proxy") == prompt.prompt_token_proxy
            and type(record.get("completion_token_proxy")) is int
            and record.get("completion_token_proxy") == count_tokens(completion),
            f"sealed parent completion record changed at ordinal "
            f"{prompt.answer_ordinal}",
        )
        for key in (
            "reported_prompt_tokens",
            "reported_completion_tokens",
            "reported_total_tokens",
        ):
            value = record.get(key)
            _require(
                value is None
                or (type(value) is int and value >= 1),
                f"sealed parent completion {key} changed at ordinal "
                f"{prompt.answer_ordinal}",
            )
        reported_prompt_tokens = record.get("reported_prompt_tokens")
        _require(
            reported_prompt_tokens is None
            or reported_prompt_tokens
            <= judging.DEFAULT_MAX_JUDGE_PROMPT_TOKENS,
            f"sealed parent reported prompt usage changed at ordinal "
            f"{prompt.answer_ordinal}",
        )
        elapsed = record.get("provider_elapsed_s")
        _require(
            type(elapsed) is float and math.isfinite(elapsed) and elapsed >= 0,
            f"sealed parent completion elapsed time changed at ordinal "
            f"{prompt.answer_ordinal}",
        )
        request_body = {
            "format": FAST_COMPLETION_REQUEST_FORMAT,
            "call_key_sha256": expected_call_key,
            "runtime_identity_sha256": runtime_identity_sha256,
            "runtime_identity": provenance,
            "prompt_population_sha256": (
                plan.prompt_population.prompt_population_sha256
            ),
            "messages_sha256": prompt.messages_sha256,
            "prompt_token_proxy": prompt.prompt_token_proxy,
            "max_new_tokens": JUDGE_MAX_TOKENS,
        }
        request_journal_sha256 = identity_sha256(request_body)
        response_body = {
            "format": FAST_COMPLETION_RESPONSE_FORMAT,
            "call_key_sha256": expected_call_key,
            "request_journal_sha256": request_journal_sha256,
            "messages_sha256": prompt.messages_sha256,
            "completion": completion,
            "completion_sha256": completion_sha256,
            "requested_model": judging.DEFAULT_SOL_GATEWAY_MODEL,
            "response_id": record["response_id"],
            "response_model": record["response_model"],
            "finish_reason": "stop",
            "prompt_token_proxy": prompt.prompt_token_proxy,
            "completion_token_proxy": record["completion_token_proxy"],
            "reported_prompt_tokens": record["reported_prompt_tokens"],
            "reported_completion_tokens": record[
                "reported_completion_tokens"
            ],
            "reported_total_tokens": record["reported_total_tokens"],
            "provider_elapsed_s": elapsed,
        }
        response_journal_sha256 = identity_sha256(response_body)
        _require(
            record.get("request_journal_sha256") == request_journal_sha256
            and record.get("response_journal_sha256")
            == response_journal_sha256
            and row.get("call_key_sha256") == expected_call_key
            and row.get("request_journal_sha256") == request_journal_sha256
            and row.get("response_journal_sha256")
            == response_journal_sha256,
            f"sealed parent completion journal binding changed at ordinal "
            f"{prompt.answer_ordinal}",
        )
        records.append(record)

    def _known_sum(name: str) -> int:
        return sum(
            int(value)
            for record in records
            if (value := record[name]) is not None
        )

    expected_usage: dict[str, Any] = {
        "completion_token_proxy": sum(
            int(record["completion_token_proxy"]) for record in records
        ),
        "deduplicated_logical_calls": 0,
        "logical_calls": len(raw_rows),
        "prompt_token_proxy": sum(
            int(record["prompt_token_proxy"]) for record in records
        ),
        "recorded_provider_elapsed_s": sum(
            float(record["provider_elapsed_s"]) for record in records
        ),
        "recorded_reported_completion_tokens": _known_sum(
            "reported_completion_tokens"
        ),
        "recorded_reported_prompt_tokens": _known_sum(
            "reported_prompt_tokens"
        ),
        "recorded_reported_total_tokens": _known_sum("reported_total_tokens"),
        "reported_completion_tokens_complete": all(
            record["reported_completion_tokens"] is not None
            for record in records
        ),
        "reported_prompt_tokens_complete": all(
            record["reported_prompt_tokens"] is not None for record in records
        ),
        "reported_total_tokens_complete": all(
            record["reported_total_tokens"] is not None for record in records
        ),
        "unique_calls": len(records),
    }
    usage = batch.get("usage")
    _require(
        type(usage) is dict and set(usage) == _STABLE_USAGE_KEYS,
        "sealed parent completion usage schema changed",
    )
    assert type(usage) is dict
    for key, expected in expected_usage.items():
        observed = usage.get(key)
        expected_type = type(expected)
        _require(
            type(observed) is expected_type and observed == expected,
            f"sealed parent completion usage changed: {key}",
        )


def _load_parent_judge(
    *,
    parent: VerifiedS0V2AnswerPlane,
    gold_rows: tuple[judging._GoldRow, ...],
    gold_population_sha256: str,
    parent_judge_root: str | Path,
    expected_parent_judge_sha256: str,
    expected_parent_score_ledger_sha256: str,
) -> _VerifiedParentJudge:
    """Validate sealed parent results without replaying 100 provider journals."""

    require_sha256(expected_parent_judge_sha256, "expected parent judge SHA-256")
    require_sha256(
        expected_parent_score_ledger_sha256,
        "expected parent score-ledger SHA-256",
    )
    root = Path(parent_judge_root)
    preflight = read_sealed_json(root / judging.JUDGE_PREFLIGHT_NAME)
    judge = read_sealed_json(root / judging.JUDGE_NAME)
    judge_replay = read_sealed_json(root / judging.JUDGE_REPLAY_NAME)
    score = read_sealed_json(root / judging.SCORE_LEDGER_NAME)
    score_replay = read_sealed_json(root / judging.SCORE_LEDGER_REPLAY_NAME)
    _require(
        judge.sha256 == judge_replay.sha256 == expected_parent_judge_sha256
        and judge.payload == judge_replay.payload,
        "sealed parent judge/replay differ",
    )
    _require(
        score.sha256
        == score_replay.sha256
        == expected_parent_score_ledger_sha256
        and score.payload == score_replay.payload,
        "sealed parent score-ledger/replay differ",
    )
    parent_plan = _parent_plan(parent, gold_rows, gold_population_sha256)
    _require(
        preflight.payload == judging._preflight_artifact(parent_plan),
        "sealed parent judge preflight changed",
    )
    payload = judge.payload
    _require(
        set(payload) == _PARENT_JUDGE_KEYS
        and payload.get("format") == judging.JUDGE_FORMAT
        and payload.get("answer_run_sha256") == parent.run_sha256
        and payload.get("runtime_ledger_sha256") == parent.runtime_ledger_sha256
        and payload.get("runtime_ledger_identity_sha256")
        == parent.runtime_ledger.get("ledger_identity_sha256")
        and payload.get("gold_population_sha256") == gold_population_sha256
        and payload.get("preflight_artifact_sha256") == preflight.sha256
        and payload.get("prompt_population_sha256")
        == parent_plan.prompt_population.prompt_population_sha256
        and payload.get("question_count") == len(parent.rows),
        "sealed parent judge envelope changed",
    )
    _require(
        payload.get("gold_loaded_posthoc") is True
        and payload.get("judge_completions_may_echo_gold") is True
        and payload.get("judge_model") == judging.DEFAULT_SOL_CALLER_MODEL
        and type(payload.get("retained_request_token_state_bytes")) is int
        and payload.get("retained_request_token_state_bytes") == 0
        and type(payload.get("unique_provider_prompt_count")) is int
        and payload.get("unique_provider_prompt_count")
        == parent_plan.required_calls,
        "sealed parent judge semantic envelope changed",
    )
    raw_rows = payload.get("questions")
    _require(
        type(raw_rows) is list and len(raw_rows) == len(parent.rows),
        "sealed parent judge population changed",
    )
    outcomes: list[_ParentOutcome] = []
    for source, gold, prompt, raw in zip(
        parent.rows,
        gold_rows,
        parent_plan.prompt_rows,
        raw_rows,
        strict=True,
    ):
        _require(
            type(raw) is dict and set(raw) == _PARENT_JUDGE_ROW_KEYS,
            f"sealed parent judge row schema changed at ordinal {source.ordinal}",
        )
        unsigned = dict(raw)
        row_sha = unsigned.pop("judge_row_sha256", None)
        _require(
            row_sha == identity_sha256(unsigned),
            f"sealed parent judge row seal changed at ordinal {source.ordinal}",
        )
        output = raw.get("judge_output")
        correct = raw.get("correct")
        _require(
            type(output) is str
            and bool(output)
            and raw.get("judge_output_sha256") == quote_sha256(output)
            and type(correct) is bool
            and parse_binary_judge_verdict(output) == correct,
            f"sealed parent verdict changed at ordinal {source.ordinal}",
        )
        _require(
            raw.get("ordinal") == source.ordinal
            and raw.get("question_id") == source.question_id
            and raw.get("question_sha256") == source.question_sha256
            and raw.get("dated_question_sha256") == source.dated_question_sha256
            and raw.get("prediction_sha256") == source.prediction_sha256
            and raw.get("runtime_row_id") == source.runtime_row_id
            and raw.get("reference_sha256") == gold.reference_sha256
            and raw.get("category") == gold.category
            and raw.get("demand_class") == prompt.demand_class
            and raw.get("judge_messages_sha256") == prompt.messages_sha256
            and raw.get("judge_prompt_token_proxy") == prompt.prompt_token_proxy,
            f"sealed parent judge binding changed at ordinal {source.ordinal}",
        )
        for key in (
            "call_key_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
        ):
            require_sha256(str(raw[key]), f"sealed parent {key}")
        _require(
            type(raw.get("normalized_exact_match")) is bool
            and raw["normalized_exact_match"]
            == exact_match(source.prediction, gold.reference)
            and type(raw.get("normalized_f1")) in (int, float)
            and float(raw["normalized_f1"])
            == f1_score(source.prediction, gold.reference),
            f"sealed parent normalized score changed at ordinal {source.ordinal}",
        )
        outcomes.append(
            _ParentOutcome(
                ordinal=source.ordinal,
                correct=correct,
                judge_row_sha256=str(row_sha),
                judge_verdict_sha256=str(raw["judge_output_sha256"]),
                demand_class=str(raw["demand_class"]),
            )
        )
    correct_count = sum(row.correct for row in outcomes)
    exact_count = sum(bool(row["normalized_exact_match"]) for row in raw_rows)
    expected_aggregate = {
        "accuracy": correct_count / len(outcomes),
        "correct": correct_count,
        "gate_passed": correct_count / len(outcomes) >= judging.TARGET_ACCURACY,
        "incorrect": len(outcomes) - correct_count,
        "mean_f1": sum(float(row["normalized_f1"]) for row in raw_rows)
        / len(raw_rows),
        "normalized_exact_match": exact_count,
        "questions": len(outcomes),
        "target_accuracy": judging.TARGET_ACCURACY,
    }
    _require(
        payload.get("aggregate") == expected_aggregate
        and payload.get("category_aggregates")
        == judging._category_aggregates(raw_rows),
        "sealed parent judge aggregate changed",
    )
    _validate_parent_completion_batch(
        payload.get("completion_batch"),
        plan=parent_plan,
        preflight_sha256=preflight.sha256,
        raw_rows=raw_rows,
    )
    expected_score = judging._score_ledger(
        parent_plan, payload, judge_artifact_sha256=judge.sha256
    )
    _require(
        score.payload == expected_score
        and score.payload.get("format") == SCORE_LEDGER_FORMAT,
        "sealed parent score ledger changed",
    )
    return _VerifiedParentJudge(
        preflight_sha256=preflight.sha256,
        judge_sha256=judge.sha256,
        score_ledger_sha256=score.sha256,
        outcomes=tuple(outcomes),
    )


def _build_plan(
    *,
    answer_plane: VerifiedClosureAnswerPlane,
    dataset_path: str | Path,
    split_path: str | Path,
    parent_judge_root: str | Path,
    expected_parent_judge_sha256: str,
    expected_parent_score_ledger_sha256: str,
    expected_question_count: int,
) -> _ClosureJudgePlan:
    profile = _profile(answer_plane.arm_label)
    # This projection decides the provider-call subset.  It is complete before
    # the first gold or parent-verdict read below.
    change_projection = _validate_answer_plane(
        answer_plane,
        profile=profile,
        expected_question_count=expected_question_count,
    )
    gold_rows, gold_population_sha256 = _load_gold(
        dataset_path=dataset_path,
        split_path=split_path,
        answer_plane=answer_plane,
    )
    _require(
        len(gold_rows) == len(answer_plane.rows),
        "closure gold population changed",
    )
    prompt_rows, prompt_population = _prompt_plan(answer_plane, gold_rows)
    parent_judge = _load_parent_judge(
        parent=answer_plane.parent_plane,
        gold_rows=gold_rows,
        gold_population_sha256=gold_population_sha256,
        parent_judge_root=parent_judge_root,
        expected_parent_judge_sha256=expected_parent_judge_sha256,
        expected_parent_score_ledger_sha256=(
            expected_parent_score_ledger_sha256
        ),
    )
    return _ClosureJudgePlan(
        answer_plane=answer_plane,
        gold_rows=gold_rows,
        gold_population_sha256=gold_population_sha256,
        change_projection=change_projection,
        prompt_rows=prompt_rows,
        prompt_population=prompt_population,
        parent_judge=parent_judge,
        profile=profile,
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


def _prompt_population_projection(plan: _ClosureJudgePlan) -> dict[str, Any]:
    if plan.prompt_population is None:
        return _empty_prompt_population()
    return plan.prompt_population.model_dump()


def _preflight_artifact(plan: _ClosureJudgePlan) -> dict[str, Any]:
    prompt_population = _prompt_population_projection(plan)
    return {
        "answer_plan_id": plan.answer_plane.answer_plan_id,
        "answer_run_sha256": plan.answer_plane.run_sha256,
        "arm_plan_id": plan.answer_plane.arm_plan_id,
        "arm_label": plan.profile.arm_label,
        "change_projection": dict(plan.change_projection),
        "changed_prediction_count": plan.changed_count,
        "format": CLOSURE_JUDGE_PREFLIGHT_FORMAT,
        "gold_loaded_posthoc": True,
        "gold_population_sha256": plan.gold_population_sha256,
        "inherited_prediction_count": len(plan.answer_plane.rows)
        - plan.changed_count,
        "judge_model": judging.DEFAULT_SOL_CALLER_MODEL,
        "logical_prompt_count": plan.changed_count,
        "matched_population_id": plan.answer_plane.matched_population_id,
        "maximum_prompt_token_proxy": max(
            (row.prompt_token_proxy for row in plan.prompt_rows), default=0
        ),
        "new_provider_calls": 0,
        "parent_answer_run_sha256": plan.answer_plane.parent_plane.run_sha256,
        "parent_runtime_ledger_sha256": (
            plan.answer_plane.parent_plane.runtime_ledger_sha256
        ),
        "parent_judge_preflight_sha256": plan.parent_judge.preflight_sha256,
        "parent_judge_sha256": plan.parent_judge.judge_sha256,
        "parent_score_ledger_sha256": plan.parent_judge.score_ledger_sha256,
        "population_identity_sha256": (
            plan.answer_plane.population_identity_sha256
        ),
        "prompt_population": prompt_population,
        "renderer_id": plan.answer_plane.renderer_id,
        "required_authorized_provider_calls": plan.required_calls,
        "retrieval_sha256": plan.answer_plane.retrieval_sha256,
        "runtime_ledger_sha256": plan.answer_plane.runtime_ledger_sha256,
        "snapshot_id": plan.answer_plane.snapshot_id,
        "source_eligibility_manifest_sha256": (
            plan.answer_plane.source_eligibility_manifest_sha256
        ),
        "source_preflight_sha256": plan.answer_plane.source_preflight_sha256,
        "source_retrieval_generation_sha256": (
            plan.answer_plane.source_retrieval_generation_sha256
        ),
        "unique_prompt_count": plan.required_calls,
    }


def _runtime(
    plan: _ClosureJudgePlan,
    *,
    checkpoint_dir: str | Path,
    client: Any | None,
    max_concurrency: int,
    preflight_artifact_sha256: str,
) -> FastCompletionRuntime:
    _require(bool(plan.prompt_rows), "empty closure judge plan has no runtime")
    return FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=[row.messages for row in plan.prompt_rows],
        model=judging.DEFAULT_SOL_GATEWAY_MODEL,
        client=client,
        max_prompt_tokens=judging.DEFAULT_MAX_JUDGE_PROMPT_TOKENS,
        max_new_tokens=JUDGE_MAX_TOKENS,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "answer_run_sha256": plan.answer_plane.run_sha256,
            "arm_label": plan.profile.arm_label,
            "authorized_unique_calls": plan.required_calls,
            "change_projection_sha256": plan.change_projection[
                "change_projection_sha256"
            ],
            "gateway_url": DEFAULT_GATEWAY_URL,
            "gold_population_sha256": plan.gold_population_sha256,
            "judge_plan_id": plan.profile.plan_id,
            "parent_answer_run_sha256": plan.answer_plane.parent_plane.run_sha256,
            "parent_judge_sha256": plan.parent_judge.judge_sha256,
            "parent_score_ledger_sha256": plan.parent_judge.score_ledger_sha256,
            "preflight_artifact_sha256": preflight_artifact_sha256,
            "runtime_ledger_sha256": plan.answer_plane.runtime_ledger_sha256,
        },
    )


def _judge_artifact(
    plan: _ClosureJudgePlan,
    batch: FastCompletionBatch | None,
    *,
    preflight_artifact_sha256: str,
) -> dict[str, Any]:
    if plan.required_calls:
        _require(batch is not None, "changed closure predictions require Sol verdicts")
        assert batch is not None
        _require(
            len(batch.logical_completions) == plan.changed_count,
            "changed Sol verdict population changed",
        )
        records = {row.messages_sha256: row for row in batch.unique_records}
        changed_outputs = {
            prompt.answer_ordinal: (prompt, verdict, records[prompt.messages_sha256])
            for prompt, verdict in zip(
                plan.prompt_rows, batch.logical_completions, strict=True
            )
        }
    else:
        _require(batch is None, "unchanged closure plan cannot carry Sol verdicts")
        changed_outputs = {}
    parent_outcomes = {row.ordinal: row for row in plan.parent_judge.outcomes}
    rows: list[dict[str, Any]] = []
    for answer, gold in zip(plan.answer_plane.rows, plan.gold_rows, strict=True):
        parent = parent_outcomes[answer.ordinal]
        changed = answer.changed_from_parent
        if changed:
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
            verdict_source = "sealed_parent_s0_v2_judge"
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
            "changed_from_parent": changed,
            "correct": correct,
            "dated_question_sha256": gold.dated_question_sha256,
            "demand_class": demand_class,
            "judge_messages_sha256": judge_messages,
            "judge_output": persisted_output,
            "judge_output_sha256": output_sha,
            "judge_prompt_token_proxy": prompt_tokens,
            "normalized_exact_match": exact_match(
                answer.prediction, gold.reference
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
    prompt_population = _prompt_population_projection(plan)
    return {
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
        "answer_plan_id": plan.answer_plane.answer_plan_id,
        "answer_run_sha256": plan.answer_plane.run_sha256,
        "arm_plan_id": plan.answer_plane.arm_plan_id,
        "arm_label": plan.profile.arm_label,
        "category_aggregates": judging._category_aggregates(rows),
        "change_projection_sha256": plan.change_projection[
            "change_projection_sha256"
        ],
        "completion_batch": None if batch is None else judging._stable_batch(batch),
        "format": CLOSURE_JUDGE_FORMAT,
        "gold_loaded_posthoc": True,
        "gold_population_sha256": plan.gold_population_sha256,
        "judge_completions_may_echo_gold": True,
        "judge_model": judging.DEFAULT_SOL_CALLER_MODEL,
        "matched_population_id": plan.answer_plane.matched_population_id,
        "parent_answer_run_sha256": plan.answer_plane.parent_plane.run_sha256,
        "parent_runtime_ledger_sha256": (
            plan.answer_plane.parent_plane.runtime_ledger_sha256
        ),
        "parent_judge_preflight_sha256": plan.parent_judge.preflight_sha256,
        "parent_judge_sha256": plan.parent_judge.judge_sha256,
        "parent_score_ledger_sha256": plan.parent_judge.score_ledger_sha256,
        "population_identity_sha256": (
            plan.answer_plane.population_identity_sha256
        ),
        "preflight_artifact_sha256": preflight_artifact_sha256,
        "prompt_population_sha256": prompt_population[
            "prompt_population_sha256"
        ],
        "question_count": len(rows),
        "questions": rows,
        "renderer_id": plan.answer_plane.renderer_id,
        "retained_request_token_state_bytes": 0,
        "retrieval_sha256": plan.answer_plane.retrieval_sha256,
        "runtime_ledger_identity_sha256": plan.answer_plane.runtime_ledger.get(
            "ledger_identity_sha256"
        ),
        "runtime_ledger_sha256": plan.answer_plane.runtime_ledger_sha256,
        "snapshot_id": plan.answer_plane.snapshot_id,
        "source_eligibility_manifest_sha256": (
            plan.answer_plane.source_eligibility_manifest_sha256
        ),
        "source_preflight_sha256": plan.answer_plane.source_preflight_sha256,
        "source_retrieval_generation_sha256": (
            plan.answer_plane.source_retrieval_generation_sha256
        ),
        "unique_provider_prompt_count": plan.required_calls,
    }


def _score_ledger(
    plan: _ClosureJudgePlan,
    judge_payload: Mapping[str, Any],
    *,
    judge_artifact_sha256: str,
) -> dict[str, Any]:
    raw_rows = judge_payload.get("questions")
    _require(type(raw_rows) is list, "closure judge questions must be an array")
    entries: list[ScoreLedgerEntry] = []
    for answer, raw in zip(plan.answer_plane.rows, raw_rows, strict=True):
        _require(
            type(raw) is dict
            and raw.get("runtime_row_id") == answer.runtime_row_id
            and type(raw.get("correct")) is bool
            and type(raw.get("baseline_correct")) is bool,
            f"closure judge/runtime binding changed at ordinal {answer.ordinal}",
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
        runtime_ledger=plan.answer_plane.runtime_ledger_json(),
        entries=entries,
        source_artifacts=(
            {
                "role": f"{plan.profile.arm_label}:judge",
                "sha256": judge_artifact_sha256,
            },
            {
                "role": f"{PARENT_ARM_LABEL}:parent_judge",
                "sha256": plan.parent_judge.judge_sha256,
            },
            {
                "role": f"{PARENT_ARM_LABEL}:parent_score_ledger",
                "sha256": plan.parent_judge.score_ledger_sha256,
            },
        ),
    )


def _authorize(
    plan: _ClosureJudgePlan,
    *,
    enable_provider: bool,
    authorized_provider_calls: int,
) -> None:
    _require(type(enable_provider) is bool, "provider enablement must be an exact bool")
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == plan.required_calls,
        f"authorized provider calls must exactly equal {plan.required_calls}",
    )
    if plan.required_calls:
        _require(enable_provider, "changed closure judge requires provider enablement")
    else:
        _require(
            not enable_provider,
            "unchanged closure judge forbids provider enablement",
        )


def preflight_closure_changed_only_judge(
    *,
    answer_plane: VerifiedClosureAnswerPlane,
    dataset_path: str | Path,
    split_path: str | Path,
    parent_judge_root: str | Path,
    expected_parent_judge_sha256: str,
    expected_parent_score_ledger_sha256: str,
    output_root: str | Path,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> SealedArtifact:
    plan = _build_plan(
        answer_plane=answer_plane,
        dataset_path=dataset_path,
        split_path=split_path,
        parent_judge_root=parent_judge_root,
        expected_parent_judge_sha256=expected_parent_judge_sha256,
        expected_parent_score_ledger_sha256=expected_parent_score_ledger_sha256,
        expected_question_count=expected_question_count,
    )
    artifact, _created = publish_sealed_json(
        Path(output_root) / JUDGE_PREFLIGHT_NAME, _preflight_artifact(plan)
    )
    return artifact


def run_closure_changed_only_judge(
    *,
    answer_plane: VerifiedClosureAnswerPlane,
    dataset_path: str | Path,
    split_path: str | Path,
    parent_judge_root: str | Path,
    expected_parent_judge_sha256: str,
    expected_parent_score_ledger_sha256: str,
    output_root: str | Path,
    enable_provider: bool,
    authorized_provider_calls: int,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    max_concurrency: int = 4,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> ClosureJudgeRunResult:
    plan = _build_plan(
        answer_plane=answer_plane,
        dataset_path=dataset_path,
        split_path=split_path,
        parent_judge_root=parent_judge_root,
        expected_parent_judge_sha256=expected_parent_judge_sha256,
        expected_parent_score_ledger_sha256=expected_parent_score_ledger_sha256,
        expected_question_count=expected_question_count,
    )
    # Authorization is checked before the first output mutation or client access.
    _authorize(
        plan,
        enable_provider=enable_provider,
        authorized_provider_calls=authorized_provider_calls,
    )
    output = Path(output_root)
    preflight, _created = publish_sealed_json(
        output / JUDGE_PREFLIGHT_NAME, _preflight_artifact(plan)
    )
    existing = output / JUDGE_NAME
    if existing.exists():
        source = read_sealed_json(existing)
        return _replay_plan(
            plan,
            expected_judge_sha256=source.sha256,
            output_root=output,
            max_concurrency=max_concurrency,
        )
    batch: FastCompletionBatch | None = None
    if plan.required_calls:
        load_dotenv()
        api_key = os.environ.get(api_key_env, "").strip()
        _require(bool(api_key), f"provider API key is empty: {api_key_env}")
        client = judging._make_provider_client(api_key, DEFAULT_GATEWAY_URL)
        try:
            batch = _runtime(
                plan,
                checkpoint_dir=output / JUDGE_CHECKPOINT_DIR_NAME,
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
            "changed Sol completion journal population changed",
        )
    payload = _judge_artifact(
        plan, batch, preflight_artifact_sha256=preflight.sha256
    )
    judge, _created = publish_sealed_json(output / JUDGE_NAME, payload)
    score_payload = _score_ledger(
        plan, payload, judge_artifact_sha256=judge.sha256
    )
    score, _created = publish_sealed_json(output / SCORE_LEDGER_NAME, score_payload)
    return ClosureJudgeRunResult(
        judge_artifact=judge,
        score_ledger_artifact=score,
        correct=int(payload["aggregate"]["correct"]),
        physical_provider_calls=0 if batch is None else batch.usage.physical_calls,
        checkpoint_hits=0 if batch is None else batch.usage.checkpoint_hits,
    )


def _replay_plan(
    plan: _ClosureJudgePlan,
    *,
    expected_judge_sha256: str,
    output_root: str | Path,
    max_concurrency: int,
) -> ClosureJudgeRunResult:
    require_sha256(expected_judge_sha256, "expected closure judge SHA-256")
    output = Path(output_root)
    preflight = read_sealed_json(output / JUDGE_PREFLIGHT_NAME)
    _require(
        preflight.payload == _preflight_artifact(plan),
        "closure judge preflight changed during replay",
    )
    source = read_sealed_json(output / JUDGE_NAME)
    _require(source.sha256 == expected_judge_sha256, "closure judge SHA-256 changed")
    batch = (
        _runtime(
            plan,
            checkpoint_dir=output / JUDGE_CHECKPOINT_DIR_NAME,
            client=None,
            max_concurrency=max_concurrency,
            preflight_artifact_sha256=preflight.sha256,
        ).run()
        if plan.required_calls
        else None
    )
    if batch is not None:
        _require(batch.usage.physical_calls == 0, "closure judge replay made calls")
        _require(
            batch.usage.checkpoint_hits == plan.required_calls,
            "closure judge replay checkpoint population changed",
        )
    expected = _judge_artifact(
        plan, batch, preflight_artifact_sha256=preflight.sha256
    )
    _require(
        canonical_json_bytes(expected) == canonical_json_bytes(source.payload),
        "closure judge differs from sealed changed-only journals",
    )
    replay, _created = publish_sealed_json(output / JUDGE_REPLAY_NAME, expected)
    score_expected = _score_ledger(
        plan, expected, judge_artifact_sha256=source.sha256
    )
    score = read_sealed_json(output / SCORE_LEDGER_NAME)
    _require(
        score.payload == score_expected,
        "closure score ledger differs from replayed verdicts",
    )
    score_replay, _created = publish_sealed_json(
        output / SCORE_LEDGER_REPLAY_NAME, score_expected
    )
    return ClosureJudgeRunResult(
        judge_artifact=replay,
        score_ledger_artifact=score_replay,
        correct=int(expected["aggregate"]["correct"]),
        physical_provider_calls=0,
        checkpoint_hits=0 if batch is None else batch.usage.checkpoint_hits,
    )


def replay_closure_changed_only_judge(
    *,
    answer_plane: VerifiedClosureAnswerPlane,
    expected_judge_sha256: str,
    dataset_path: str | Path,
    split_path: str | Path,
    parent_judge_root: str | Path,
    expected_parent_judge_sha256: str,
    expected_parent_score_ledger_sha256: str,
    output_root: str | Path,
    max_concurrency: int = 4,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
) -> ClosureJudgeRunResult:
    plan = _build_plan(
        answer_plane=answer_plane,
        dataset_path=dataset_path,
        split_path=split_path,
        parent_judge_root=parent_judge_root,
        expected_parent_judge_sha256=expected_parent_judge_sha256,
        expected_parent_score_ledger_sha256=expected_parent_score_ledger_sha256,
        expected_question_count=expected_question_count,
    )
    return _replay_plan(
        plan,
        expected_judge_sha256=expected_judge_sha256,
        output_root=output_root,
        max_concurrency=max_concurrency,
    )


# Compact names for callers that already operate inside the closure namespace.
preflight_closure_judge = preflight_closure_changed_only_judge
run_closure_judge = run_closure_changed_only_judge
replay_closure_judge = replay_closure_changed_only_judge


__all__ = [
    "CLOSURE_JUDGE_FORMAT",
    "CLOSURE_JUDGE_PREFLIGHT_FORMAT",
    "ClosureJudgeRunResult",
    "EMPTY_PROMPT_POPULATION_FORMAT",
    "JUDGE_CHECKPOINT_DIR_NAME",
    "JUDGE_NAME",
    "JUDGE_PREFLIGHT_NAME",
    "JUDGE_REPLAY_NAME",
    "SCORE_LEDGER_NAME",
    "SCORE_LEDGER_REPLAY_NAME",
    "preflight_closure_changed_only_judge",
    "preflight_closure_judge",
    "replay_closure_changed_only_judge",
    "replay_closure_judge",
    "run_closure_changed_only_judge",
    "run_closure_judge",
]
