"""Provider-free composition of V2 map/solver and adaptive source fallback.

The existing V2 solver population cannot be known until the evidence-map run
has a sealed terminal replay.  This module therefore distinguishes a sealed
stage population from a deferred, conservatively bounded population.  It also
binds the source-gate policy and mapper contract without pretending that the
current solver can consume post-gate facts.

No function here opens an artifact, writes a file, or calls a provider.
Callers pass already verified :class:`~tools.matched_eval.artifacts.SealedArtifact`
values from the public V2 preflight functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

from .artifacts import SealedArtifact
from .contracts import (
    ArtifactRef,
    MatchedEvalContractError,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
    require_text,
)
from .query_evidence_map_solver_v2_live import (
    ANSWER_PLAN_ID,
    ARM_LABEL,
    MAP_OUTPUT_TOKEN_RESERVE,
    MAP_PLAN_ID,
    MAP_PREFLIGHT_FORMAT,
    MAP_RENDERER_ID,
    MAX_PROMPT_TOKENS,
    SOLVER_OUTPUT_TOKEN_RESERVE,
    SOLVER_PREFLIGHT_FORMAT,
    SOLVER_RENDERER_ID,
)
from .source_gate_controller import (
    FORMAT as SOURCE_GATE_FORMAT,
    SourceGatePolicy,
)


FORMAT = "memory-condense-adaptive-memory-preflight-v1"
HARD_PROMPT_TOKEN_CAP = 8_000
SOURCE_GATE_INSERTION_POINT = (
    "after_terminal_v2_evidence_map_before_final_solver"
)
SOURCE_GATE_INPUT_ROLES = (
    "direct_query_run",
    "partition_r96_generation",
    "guided_run",
)


class AdaptiveMemoryPreflightError(MatchedEvalContractError):
    """A stage population, budget, or conditional binding changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise AdaptiveMemoryPreflightError(message)


def _int(value: object, label: str, minimum: int = 0) -> int:
    _require(
        type(value) is int and value >= minimum,
        f"{label} must be an exact integer >= {minimum}",
    )
    return value  # type: ignore[return-value]


def _optional_sha(value: str | None, label: str) -> str | None:
    return None if value is None else require_sha256(value, label)


@dataclass(frozen=True, slots=True)
class QuestionCallBinding:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    provider_call_planned: bool

    def __post_init__(self) -> None:
        _int(self.ordinal, "question ordinal")
        require_text(self.question_id, "question ID")
        require_sha256(self.question_sha256, "question SHA-256")
        require_sha256(self.dated_question_sha256, "dated-question SHA-256")
        _require(
            type(self.provider_call_planned) is bool,
            "provider-call disposition must be an exact bool",
        )

    def projection(self) -> dict[str, object]:
        return {
            "dated_question_sha256": self.dated_question_sha256,
            "ordinal": self.ordinal,
            "provider_call_planned": self.provider_call_planned,
            "question_id": self.question_id,
            "question_sha256": self.question_sha256,
        }


@dataclass(frozen=True, slots=True)
class StagePreflightBinding:
    stage: Literal["map", "solver"]
    availability: Literal["sealed", "deferred"]
    contract_format: str
    plan_id: str
    renderer_id: str
    preflight_sha256: str | None
    plan_identity_sha256: str | None
    prompt_population_sha256: str | None
    snapshot_id: str
    retrieval_sha256: str
    adapter_population_id: str
    direct_answer_run_sha256: str
    questions: tuple[QuestionCallBinding, ...]
    exact_required_calls: int | None
    call_upper_bound: int
    hard_prompt_token_cap: int
    output_token_reserve: int
    observed_max_prompt_token_proxy: int | None
    prerequisite: str | None = None
    retained_request_token_state_bytes: Literal[0] = 0

    def __post_init__(self) -> None:
        _require(self.stage in {"map", "solver"}, "unknown adaptive stage")
        _require(
            self.availability in {"sealed", "deferred"},
            "unknown stage availability",
        )
        require_text(self.contract_format, "stage format")
        require_text(self.plan_id, "stage plan ID")
        require_text(self.renderer_id, "stage renderer ID")
        _optional_sha(self.preflight_sha256, "stage preflight")
        _optional_sha(self.plan_identity_sha256, "stage plan identity")
        _optional_sha(self.prompt_population_sha256, "prompt population")
        for value, label in (
            (self.snapshot_id, "stage snapshot"),
            (self.retrieval_sha256, "stage retrieval"),
            (self.adapter_population_id, "stage adapter population"),
            (self.direct_answer_run_sha256, "stage direct-answer run"),
        ):
            require_sha256(value, label)
        _require(
            type(self.questions) is tuple
            and all(type(row) is QuestionCallBinding for row in self.questions)
            and bool(self.questions),
            "stage questions must be a non-empty exact tuple",
        )
        ordinals = tuple(row.ordinal for row in self.questions)
        question_ids = tuple(row.question_id for row in self.questions)
        _require(
            len(set(ordinals)) == len(ordinals)
            and ordinals == tuple(sorted(ordinals))
            and len(set(question_ids)) == len(question_ids),
            "stage question population is repeated or reordered",
        )
        _int(self.call_upper_bound, "stage call upper bound")
        _require(
            self.hard_prompt_token_cap
            == MAX_PROMPT_TOKENS
            == HARD_PROMPT_TOKEN_CAP,
            "stage hard prompt cap is not 8k",
        )
        _int(self.output_token_reserve, "stage output reserve")
        expected_reserve = (
            MAP_OUTPUT_TOKEN_RESERVE
            if self.stage == "map"
            else SOLVER_OUTPUT_TOKEN_RESERVE
        )
        _require(
            self.output_token_reserve == expected_reserve,
            "stage output reserve changed",
        )
        if self.observed_max_prompt_token_proxy is not None:
            observed = _int(
                self.observed_max_prompt_token_proxy,
                "observed prompt-token proxy",
            )
            _require(
                observed + self.output_token_reserve
                <= self.hard_prompt_token_cap,
                "stage prompt plus output reserve exceeds 8k",
            )
        planned = sum(row.provider_call_planned for row in self.questions)
        if self.availability == "sealed":
            _require(
                self.preflight_sha256 is not None
                and self.plan_identity_sha256 is not None
                and self.prompt_population_sha256 is not None
                and self.exact_required_calls is not None
                and self.prerequisite is None,
                "sealed stage lost an exact population identity",
            )
            exact = _int(self.exact_required_calls, "exact stage calls")
            _require(
                exact == planned == self.call_upper_bound,
                "sealed stage call population changed",
            )
        else:
            _require(
                self.preflight_sha256 is None
                and self.plan_identity_sha256 is None
                and self.prompt_population_sha256 is None
                and self.exact_required_calls is None
                and isinstance(self.prerequisite, str)
                and bool(self.prerequisite),
                "deferred stage must name its missing prerequisite",
            )
            _require(
                planned == self.call_upper_bound,
                "deferred stage upper bound changed",
            )
        _require(
            type(self.retained_request_token_state_bytes) is int
            and self.retained_request_token_state_bytes == 0,
            "stage retained transformer token state",
        )

    def projection(self) -> dict[str, object]:
        return {
            "adapter_population_id": self.adapter_population_id,
            "availability": self.availability,
            "call_upper_bound": self.call_upper_bound,
            "contract_format": self.contract_format,
            "direct_answer_run_sha256": self.direct_answer_run_sha256,
            "exact_required_calls": self.exact_required_calls,
            "hard_prompt_token_cap": self.hard_prompt_token_cap,
            "observed_max_prompt_token_proxy": (
                self.observed_max_prompt_token_proxy
            ),
            "output_token_reserve": self.output_token_reserve,
            "plan_id": self.plan_id,
            "plan_identity_sha256": self.plan_identity_sha256,
            "preflight_sha256": self.preflight_sha256,
            "prerequisite": self.prerequisite,
            "prompt_population_sha256": self.prompt_population_sha256,
            "questions": [row.projection() for row in self.questions],
            "renderer_id": self.renderer_id,
            "retained_request_token_state_bytes": (
                self.retained_request_token_state_bytes
            ),
            "retrieval_sha256": self.retrieval_sha256,
            "snapshot_id": self.snapshot_id,
            "stage": self.stage,
        }

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class ConditionalSourceGateContract:
    """Pinned policy and provider envelope for a conditional fallback."""

    controller_contract_sha256: str
    mapper_contract_sha256: str
    upstream_fact_frontier_contract_sha256: str
    policy: SourceGatePolicy
    candidate_artifacts: tuple[ArtifactRef, ...]
    activation_question_ids: tuple[str, ...]
    mapper_max_prompt_token_proxy: int | None = None
    mapper_output_token_reserve: int | None = None
    augmented_solver_contract_sha256: str | None = None
    augmented_solver_max_prompt_token_proxy: int | None = None
    augmented_solver_output_token_reserve: int | None = None
    retained_transformer_token_state_bytes: Literal[0] = 0

    def __post_init__(self) -> None:
        for value, label in (
            (self.controller_contract_sha256, "source-gate controller contract"),
            (self.mapper_contract_sha256, "source-gate mapper contract"),
            (
                self.upstream_fact_frontier_contract_sha256,
                "upstream fact-frontier contract",
            ),
        ):
            require_sha256(value, label)
        _optional_sha(
            self.augmented_solver_contract_sha256,
            "augmented solver contract",
        )
        _require(
            type(self.policy) is SourceGatePolicy,
            "source-gate policy must be exact",
        )
        _require(
            type(self.candidate_artifacts) is tuple
            and all(type(row) is ArtifactRef for row in self.candidate_artifacts)
            and tuple(row.role for row in self.candidate_artifacts)
            == SOURCE_GATE_INPUT_ROLES,
            "source-gate candidate artifacts changed roles or order",
        )
        _require(
            type(self.activation_question_ids) is tuple
            and all(type(row) is str and bool(row) for row in self.activation_question_ids)
            and len(set(self.activation_question_ids))
            == len(self.activation_question_ids),
            "source-gate activation population must be ordered and unique",
        )
        envelope = (
            self.mapper_max_prompt_token_proxy,
            self.mapper_output_token_reserve,
        )
        _require(
            all(value is None for value in envelope)
            or all(type(value) is int for value in envelope),
            "source-gate mapper envelope must be complete or deferred",
        )
        if self.mapper_max_prompt_token_proxy is not None:
            prompt = _int(
                self.mapper_max_prompt_token_proxy,
                "source-gate mapper prompt upper bound",
            )
            reserve = _int(
                self.mapper_output_token_reserve,
                "source-gate mapper output reserve",
            )
            _require(
                reserve < HARD_PROMPT_TOKEN_CAP
                and prompt + reserve <= HARD_PROMPT_TOKEN_CAP,
                "source-gate mapper prompt plus output reserve exceeds 8k",
            )
        solver_envelope = (
            self.augmented_solver_max_prompt_token_proxy,
            self.augmented_solver_output_token_reserve,
        )
        _require(
            all(value is None for value in solver_envelope)
            or all(type(value) is int for value in solver_envelope),
            "augmented solver envelope must be complete or deferred",
        )
        if self.augmented_solver_max_prompt_token_proxy is not None:
            prompt = _int(
                self.augmented_solver_max_prompt_token_proxy,
                "augmented solver prompt upper bound",
            )
            reserve = _int(
                self.augmented_solver_output_token_reserve,
                "augmented solver output reserve",
            )
            _require(
                reserve == SOLVER_OUTPUT_TOKEN_RESERVE
                and prompt + reserve <= HARD_PROMPT_TOKEN_CAP,
                "augmented solver prompt plus output reserve exceeds or changes 8k envelope",
            )
        _require(
            type(self.retained_transformer_token_state_bytes) is int
            and self.retained_transformer_token_state_bytes == 0,
            "source-gate retained transformer token state",
        )

    @property
    def activation_upper_bound(self) -> int:
        return len(self.activation_question_ids)

    @property
    def physical_map_call_upper_bound(self) -> int:
        return self.activation_upper_bound * self.policy.max_physical_map_calls

    def projection(self) -> dict[str, object]:
        return {
            "activation_question_ids": list(self.activation_question_ids),
            "activation_upper_bound": self.activation_upper_bound,
            "augmented_solver_contract_sha256": (
                self.augmented_solver_contract_sha256
            ),
            "augmented_solver_max_prompt_token_proxy": (
                self.augmented_solver_max_prompt_token_proxy
            ),
            "augmented_solver_output_token_reserve": (
                self.augmented_solver_output_token_reserve
            ),
            "candidate_artifacts": [
                row.projection() for row in self.candidate_artifacts
            ],
            "controller_contract_sha256": self.controller_contract_sha256,
            "controller_format": SOURCE_GATE_FORMAT,
            "hard_prompt_token_cap": HARD_PROMPT_TOKEN_CAP,
            "insertion_point": SOURCE_GATE_INSERTION_POINT,
            "mapper_contract_sha256": self.mapper_contract_sha256,
            "mapper_max_prompt_token_proxy": (
                self.mapper_max_prompt_token_proxy
            ),
            "mapper_output_token_reserve": self.mapper_output_token_reserve,
            "physical_map_call_upper_bound": (
                self.physical_map_call_upper_bound
            ),
            "policy": self.policy.projection(),
            "retained_transformer_token_state_bytes": (
                self.retained_transformer_token_state_bytes
            ),
            "upstream_fact_frontier_contract_sha256": (
                self.upstream_fact_frontier_contract_sha256
            ),
        }

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


@dataclass(frozen=True, slots=True)
class AdaptiveMemoryPreflight:
    map_stage: StagePreflightBinding
    solver_stage: StagePreflightBinding
    source_gate: ConditionalSourceGateContract
    sealed_two_pass_exact_calls: int | None
    baseline_two_pass_call_upper_bound: int
    combined_solver_call_upper_bound: int
    conditional_source_map_call_upper_bound: int
    conditional_total_provider_call_upper_bound: int
    missing_inputs: tuple[str, ...]
    contract_complete: bool
    hard_prompt_token_cap: int = HARD_PROMPT_TOKEN_CAP
    provider_calls_executed: Literal[0] = 0
    retained_transformer_token_state_bytes: Literal[0] = 0

    def __post_init__(self) -> None:
        _require(
            type(self.map_stage) is StagePreflightBinding
            and self.map_stage.stage == "map"
            and self.map_stage.availability == "sealed",
            "adaptive preflight requires one sealed map population",
        )
        _require(
            type(self.solver_stage) is StagePreflightBinding
            and self.solver_stage.stage == "solver",
            "adaptive preflight requires one solver binding",
        )
        _require(
            self.map_stage.questions == self.solver_stage.questions,
            "map and solver question populations differ",
        )
        _require(
            type(self.source_gate) is ConditionalSourceGateContract,
            "adaptive source-gate contract must be exact",
        )
        _int(
            self.baseline_two_pass_call_upper_bound,
            "two-pass call upper bound",
        )
        _int(
            self.combined_solver_call_upper_bound,
            "combined solver call upper bound",
        )
        _int(
            self.conditional_source_map_call_upper_bound,
            "conditional source-map call upper bound",
        )
        _int(
            self.conditional_total_provider_call_upper_bound,
            "conditional total call upper bound",
        )
        expected_baseline = (
            int(self.map_stage.exact_required_calls)
            + self.solver_stage.call_upper_bound
        )
        expected_combined_solver = len(
            {
                row.question_id
                for row in self.solver_stage.questions
                if row.provider_call_planned
            }
            | set(self.source_gate.activation_question_ids)
        )
        _require(
            self.baseline_two_pass_call_upper_bound == expected_baseline
            and self.combined_solver_call_upper_bound
            == expected_combined_solver
            and self.conditional_source_map_call_upper_bound
            == self.source_gate.physical_map_call_upper_bound
            and self.conditional_total_provider_call_upper_bound
            == int(self.map_stage.exact_required_calls)
            + self.source_gate.physical_map_call_upper_bound
            + self.combined_solver_call_upper_bound,
            "adaptive call-bound arithmetic changed",
        )
        if self.solver_stage.availability == "sealed":
            _require(
                self.sealed_two_pass_exact_calls == expected_baseline,
                "sealed two-pass exact call count changed",
            )
        else:
            _require(
                self.sealed_two_pass_exact_calls is None,
                "deferred solver cannot claim an exact sealed total",
            )
        _require(
            type(self.missing_inputs) is tuple
            and len(set(self.missing_inputs)) == len(self.missing_inputs)
            and self.contract_complete == (not self.missing_inputs),
            "adaptive completeness disagrees with missing inputs",
        )
        _require(
            self.hard_prompt_token_cap == HARD_PROMPT_TOKEN_CAP,
            "adaptive hard prompt cap is not 8k",
        )
        _require(
            type(self.provider_calls_executed) is int
            and self.provider_calls_executed == 0
            and type(self.retained_transformer_token_state_bytes) is int
            and self.retained_transformer_token_state_bytes == 0,
            "adaptive preflight executed a provider or retained request state",
        )
        assert_gold_blind(self.projection(), path="adaptive_memory_preflight")

    def projection(self) -> dict[str, object]:
        return {
            "baseline_two_pass_call_upper_bound": (
                self.baseline_two_pass_call_upper_bound
            ),
            "conditional_source_map_call_upper_bound": (
                self.conditional_source_map_call_upper_bound
            ),
            "combined_solver_call_upper_bound": (
                self.combined_solver_call_upper_bound
            ),
            "conditional_total_provider_call_upper_bound": (
                self.conditional_total_provider_call_upper_bound
            ),
            "contract_complete": self.contract_complete,
            "format": FORMAT,
            "hard_prompt_token_cap": self.hard_prompt_token_cap,
            "map_stage_receipt_sha256": self.map_stage.receipt_sha256,
            "missing_inputs": list(self.missing_inputs),
            "provider_calls_executed": self.provider_calls_executed,
            "retained_transformer_token_state_bytes": (
                self.retained_transformer_token_state_bytes
            ),
            "sealed_two_pass_exact_calls": self.sealed_two_pass_exact_calls,
            "solver_insertion_point": SOURCE_GATE_INSERTION_POINT,
            "solver_population_role": (
                "direct_map_baseline_only"
                if self.source_gate.activation_upper_bound
                else "final_no_fallback_population"
            ),
            "solver_stage_receipt_sha256": self.solver_stage.receipt_sha256,
            "source_gate_receipt_sha256": self.source_gate.receipt_sha256,
        }

    @property
    def receipt_sha256(self) -> str:
        return identity_sha256(self.projection())


def _artifact_payload(artifact: SealedArtifact, label: str) -> dict[str, Any]:
    if type(artifact) is not SealedArtifact:
        raise TypeError(f"{label} must be an exact SealedArtifact")
    payload = artifact.payload
    _require(type(payload) is dict, f"{label} payload must be a mapping")
    digest = sha256(canonical_json_bytes(payload)).hexdigest()
    _require(digest == artifact.sha256, f"{label} artifact SHA-256 changed")
    assert_gold_blind(payload, path=f"adaptive_{label}")
    return payload


def _questions(
    payload: dict[str, Any],
    *,
    output_reserve: int,
) -> tuple[QuestionCallBinding, ...]:
    raw = payload.get("ordered_rows")
    _require(type(raw) is list and bool(raw), "stage ordered rows changed")
    questions: list[QuestionCallBinding] = []
    for source in raw:
        _require(type(source) is dict, "stage ordered row changed type")
        assert type(source) is dict
        planned = source.get("provider_call_planned")
        _require(type(planned) is bool, "stage call plan changed type")
        prompt_tokens = source.get("prompt_token_proxy")
        if planned:
            prompt = _int(prompt_tokens, "row prompt-token proxy")
            _require(
                prompt + output_reserve <= HARD_PROMPT_TOKEN_CAP,
                "row prompt plus output reserve exceeds 8k",
            )
        else:
            _require(prompt_tokens is None, "no-op row unexpectedly has a prompt")
        questions.append(
            QuestionCallBinding(
                _int(source.get("ordinal"), "row ordinal"),
                require_text(source.get("question_id"), "row question ID"),
                require_sha256(
                    source.get("question_sha256"), "row question SHA-256"
                ),
                require_sha256(
                    source.get("dated_question_sha256"),
                    "row dated-question SHA-256",
                ),
                planned,
            )
        )
    _require(
        payload.get("question_count") == len(questions),
        "stage question count changed",
    )
    return tuple(questions)


def _sealed_stage(
    artifact: SealedArtifact,
    *,
    stage: Literal["map", "solver"],
    expected_questions: tuple[QuestionCallBinding, ...] | None = None,
) -> StagePreflightBinding:
    payload = _artifact_payload(artifact, f"{stage}_preflight")
    if stage == "map":
        expected_format = MAP_PREFLIGHT_FORMAT
        plan_key, plan_id = "map_plan_id", MAP_PLAN_ID
        renderer_key, renderer_id = "map_renderer_id", MAP_RENDERER_ID
        reserve = MAP_OUTPUT_TOKEN_RESERVE
    else:
        expected_format = SOLVER_PREFLIGHT_FORMAT
        plan_key, plan_id = "answer_plan_id", ANSWER_PLAN_ID
        renderer_key, renderer_id = "solver_renderer_id", SOLVER_RENDERER_ID
        reserve = SOLVER_OUTPUT_TOKEN_RESERVE
    for value, label in (
        (payload.get("snapshot_id"), "stage snapshot"),
        (payload.get("retrieval_sha256"), "stage retrieval"),
        (payload.get("adapter_population_id"), "stage adapter population"),
        (payload.get("direct_answer_run_sha256"), "stage direct-answer run"),
        (payload.get("plan_identity_sha256"), "stage plan identity"),
        (payload.get("prompt_population_sha256"), "prompt population"),
    ):
        require_sha256(value, label)
    _require(
        payload.get("format") == expected_format
        and payload.get("arm_label") == ARM_LABEL
        and payload.get("stage") == stage
        and payload.get(plan_key) == plan_id
        and payload.get(renderer_key) == renderer_id
        and payload.get("gold_loaded") is False,
        f"V2 {stage} contract identity changed",
    )
    cap = _int(payload.get("hard_prompt_token_cap"), "hard prompt cap", 1)
    output = _int(payload.get("output_token_reserve"), "output reserve")
    observed = _int(
        payload.get("observed_max_prompt_token_proxy"),
        "observed prompt-token proxy",
    )
    _require(
        cap == HARD_PROMPT_TOKEN_CAP
        and payload.get("prompt_and_output_token_envelope") == cap
        and output == reserve
        and observed + output <= cap,
        f"V2 {stage} 8k envelope changed",
    )
    _require(
        type(payload.get("provider_calls")) is int
        and payload.get("provider_calls") == 0
        and type(payload.get("retained_request_token_state_bytes")) is int
        and payload.get("retained_request_token_state_bytes") == 0,
        f"V2 {stage} preflight executed or retained provider state",
    )
    required = _int(
        payload.get("required_authorized_provider_calls"),
        "required provider calls",
    )
    _require(
        payload.get("logical_prompt_count") == required
        and payload.get("unique_prompt_count") == required,
        f"V2 {stage} call population changed",
    )
    prompts = payload.get("provider_prompts")
    population = payload.get("prompt_population")
    _require(
        type(prompts) is list
        and len(prompts) == required
        and type(population) is dict
        and population.get("logical_prompt_count") == required
        and population.get("unique_prompt_count") == required
        and population.get("prompt_population_sha256")
        == payload.get("prompt_population_sha256"),
        f"V2 {stage} sealed prompt population changed",
    )
    questions = _questions(payload, output_reserve=reserve)
    _require(
        sum(row.provider_call_planned for row in questions) == required,
        f"V2 {stage} row calls differ from its prompt population",
    )
    if expected_questions is not None:
        _require(
            questions == expected_questions,
            "V2 solver question/call population differs from map",
        )
        _require(
            payload.get("map_run_sha256")
            == payload.get("map_replay_sha256"),
            "V2 solver preflight lacks a terminal map replay",
        )
    return StagePreflightBinding(
        stage,
        "sealed",
        expected_format,
        plan_id,
        renderer_id,
        artifact.sha256,
        payload["plan_identity_sha256"],
        payload["prompt_population_sha256"],
        payload["snapshot_id"],
        payload["retrieval_sha256"],
        payload["adapter_population_id"],
        payload["direct_answer_run_sha256"],
        questions,
        required,
        required,
        cap,
        output,
        observed,
    )


def _deferred_solver(map_stage: StagePreflightBinding) -> StagePreflightBinding:
    return StagePreflightBinding(
        "solver",
        "deferred",
        SOLVER_PREFLIGHT_FORMAT,
        ANSWER_PLAN_ID,
        SOLVER_RENDERER_ID,
        None,
        None,
        None,
        map_stage.snapshot_id,
        map_stage.retrieval_sha256,
        map_stage.adapter_population_id,
        map_stage.direct_answer_run_sha256,
        map_stage.questions,
        None,
        sum(row.provider_call_planned for row in map_stage.questions),
        HARD_PROMPT_TOKEN_CAP,
        SOLVER_OUTPUT_TOKEN_RESERVE,
        None,
        "terminal_v2_map_replay",
    )


def compose_adaptive_memory_preflight(
    map_preflight: SealedArtifact,
    *,
    source_gate: ConditionalSourceGateContract,
    solver_preflight: SealedArtifact | None = None,
) -> AdaptiveMemoryPreflight:
    """Compose exact current populations and conservative fallback bounds.

    A missing solver preflight is expected before the map provider stage has a
    terminal replay.  The result remains useful: its solver population count is
    bounded from the map row dispositions, and the missing artifact is named.
    """

    if type(source_gate) is not ConditionalSourceGateContract:
        raise TypeError("source_gate must be an exact ConditionalSourceGateContract")
    map_stage = _sealed_stage(map_preflight, stage="map")
    solver_stage = (
        _deferred_solver(map_stage)
        if solver_preflight is None
        else _sealed_stage(
            solver_preflight,
            stage="solver",
            expected_questions=map_stage.questions,
        )
    )
    map_question_ids = tuple(row.question_id for row in map_stage.questions)
    activation_ids = source_gate.activation_question_ids
    iterator = iter(map_question_ids)
    _require(
        set(activation_ids) <= set(map_question_ids)
        and all(any(candidate == value for candidate in iterator) for value in activation_ids),
        "source-gate activation population escaped or reordered V2 questions",
    )
    _require(
        solver_stage.snapshot_id == map_stage.snapshot_id
        and solver_stage.retrieval_sha256 == map_stage.retrieval_sha256
        and solver_stage.adapter_population_id == map_stage.adapter_population_id
        and solver_stage.direct_answer_run_sha256
        == map_stage.direct_answer_run_sha256,
        "V2 map/solver parent identities differ",
    )
    missing: list[str] = []
    if solver_stage.availability == "deferred":
        missing.append("terminal_v2_map_replay_and_solver_preflight")
    if source_gate.activation_upper_bound:
        if source_gate.mapper_max_prompt_token_proxy is None:
            missing.append("source_gate_mapper_render_envelope")
        if source_gate.augmented_solver_contract_sha256 is None:
            missing.append("source_gate_combined_solver_adapter")
        if source_gate.augmented_solver_max_prompt_token_proxy is None:
            missing.append("source_gate_combined_solver_render_envelope")
    map_calls = int(map_stage.exact_required_calls)
    baseline = map_calls + solver_stage.call_upper_bound
    source_calls = source_gate.physical_map_call_upper_bound
    combined_solver_calls = len(
        {
            row.question_id
            for row in solver_stage.questions
            if row.provider_call_planned
        }
        | set(source_gate.activation_question_ids)
    )
    return AdaptiveMemoryPreflight(
        map_stage,
        solver_stage,
        source_gate,
        (
            baseline
            if solver_stage.availability == "sealed"
            else None
        ),
        baseline,
        combined_solver_calls,
        source_calls,
        map_calls + source_calls + combined_solver_calls,
        tuple(missing),
        not missing,
    )


__all__ = [
    "FORMAT",
    "HARD_PROMPT_TOKEN_CAP",
    "SOURCE_GATE_INPUT_ROLES",
    "SOURCE_GATE_INSERTION_POINT",
    "AdaptiveMemoryPreflight",
    "AdaptiveMemoryPreflightError",
    "ConditionalSourceGateContract",
    "QuestionCallBinding",
    "StagePreflightBinding",
    "compose_adaptive_memory_preflight",
]
