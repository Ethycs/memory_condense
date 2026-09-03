"""Live, replayable S0-v2 responder execution for the matched-eval spine.

This module is the only bridge from the provider-free matched S0 population to
Terra completions.  It persists no gold data, requires exact call
authorization, and can reconstruct both its answer artifact and runtime ledger
from immutable completion journals before any scorer is allowed to load gold.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
)

from .artifacts import SealedArtifact, publish_sealed_json, read_sealed_json
from .contracts import (
    MatchedEvalContractError,
    StageDisposition,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
)
from .ledger import RuntimeLedgerEntry, build_runtime_ledger
from .population import (
    DEFAULT_MAX_PROMPT_TOKENS,
    EXPECTED_QUESTION_COUNT,
    EXPECTED_RETRIEVAL_SHA256,
    SOURCE_STAGE_ID,
    MatchedS0Population,
    load_s0_population,
    select_s0_population,
)
from .provider_runtime import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_GATEWAY_URL,
    DEFAULT_TERRA_CALLER_MODEL,
    DEFAULT_TERRA_GATEWAY_MODEL,
    make_provider_client,
)
from .renderer import RENDERER_ID, V3_RENDERER_ID, V4_RENDERER_ID


ANSWER_RUN_FORMAT = "memory-condense-matched-s0-v2-answer-run-v1"
ANSWER_PLAN_ID = "matched_s0_control_v2_terra_answer_v1"
ARM_LABEL = "S0_CONTROL_V2"
V3_ANSWER_RUN_FORMAT = "memory-condense-matched-s0-v3-answer-run-v1"
V3_ANSWER_PLAN_ID = "matched_s0_control_v3_terra_answer_v1"
V3_ARM_LABEL = "S0_CONTROL_V3"
V4_ANSWER_RUN_FORMAT = "memory-condense-matched-s0-v4-answer-run-v1"
V4_ANSWER_PLAN_ID = "matched_s0_control_v4_terra_answer_v1"
V4_ARM_LABEL = "S0_CONTROL_V4"
DEFAULT_MAX_NEW_TOKENS = 256

PREFLIGHT_NAME = "s0-v2-preflight.json"
V3_PREFLIGHT_NAME = "s0-v3-preflight.json"
V4_PREFLIGHT_NAME = "s0-v4-preflight.json"
ANSWER_RUN_NAME = "answer-run.json"
ANSWER_REPLAY_NAME = "answer-run-replay.json"
RUNTIME_LEDGER_NAME = "runtime-ledger.json"
RUNTIME_LEDGER_REPLAY_NAME = "runtime-ledger-replay.json"
CHECKPOINT_DIR_NAME = "terra-answer-calls"

_RECORD_DISPOSITION_FIELDS = frozenset({"checkpoint_hit", "physical_call"})
_USAGE_DISPOSITION_FIELDS = frozenset({"checkpoint_hits", "physical_calls"})


@dataclass(frozen=True, slots=True)
class S0ExecutionProfile:
    renderer_id: str
    answer_run_format: str
    answer_plan_id: str
    arm_label: str
    preflight_name: str


V2_EXECUTION_PROFILE = S0ExecutionProfile(
    renderer_id=RENDERER_ID,
    answer_run_format=ANSWER_RUN_FORMAT,
    answer_plan_id=ANSWER_PLAN_ID,
    arm_label=ARM_LABEL,
    preflight_name=PREFLIGHT_NAME,
)
V3_EXECUTION_PROFILE = S0ExecutionProfile(
    renderer_id=V3_RENDERER_ID,
    answer_run_format=V3_ANSWER_RUN_FORMAT,
    answer_plan_id=V3_ANSWER_PLAN_ID,
    arm_label=V3_ARM_LABEL,
    preflight_name=V3_PREFLIGHT_NAME,
)
V4_EXECUTION_PROFILE = S0ExecutionProfile(
    renderer_id=V4_RENDERER_ID,
    answer_run_format=V4_ANSWER_RUN_FORMAT,
    answer_plan_id=V4_ANSWER_PLAN_ID,
    arm_label=V4_ARM_LABEL,
    preflight_name=V4_PREFLIGHT_NAME,
)


def execution_profile(renderer_id: str) -> S0ExecutionProfile:
    if renderer_id == RENDERER_ID:
        return V2_EXECUTION_PROFILE
    if renderer_id == V3_RENDERER_ID:
        return V3_EXECUTION_PROFILE
    if renderer_id == V4_RENDERER_ID:
        return V4_EXECUTION_PROFILE
    raise MatchedEvalContractError(
        f"unsupported S0 execution renderer: {renderer_id!r}"
    )


def _require(condition: object, message: str) -> None:
    if not condition:
        raise MatchedEvalContractError(message)


def _plain_messages(
    messages: Sequence[Mapping[str, str]],
) -> tuple[dict[str, str], ...]:
    return tuple(dict(message) for message in messages)


def _freeze_json(value: Any) -> Any:
    """Return an immutable view of an already validated JSON projection."""

    if type(value) is dict:
        return MappingProxyType(
            {str(key): _freeze_json(child) for key, child in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Copy an immutable JSON view back to exact dict/list containers."""

    if isinstance(value, Mapping):
        return {str(key): _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def _make_provider_client(api_key: str, gateway_url: str) -> Any:
    return make_provider_client(api_key, gateway_url)


def _stable_batch(batch: FastCompletionBatch) -> dict[str, Any]:
    value = batch.model_dump()
    return {
        "logical_completions": value["logical_completions"],
        "unique_records": [
            {
                key: child
                for key, child in row.items()
                if key not in _RECORD_DISPOSITION_FIELDS
            }
            for row in value["unique_records"]
        ],
        "usage": {
            key: child
            for key, child in value["usage"].items()
            if key not in _USAGE_DISPOSITION_FIELDS
        },
        "provenance": value["provenance"],
        "runtime_identity_sha256": value["runtime_identity_sha256"],
        "prompt_population": value["prompt_population"],
    }


def _load_population(
    retrieval_path: str | Path,
    *,
    expected_retrieval_sha256: str | None,
    expected_question_count: int,
    renderer_id: str = RENDERER_ID,
    selected_ordinals: Sequence[int] | None = None,
) -> MatchedS0Population:
    population = load_s0_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
        max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
        renderer_id=renderer_id,
    )
    if selected_ordinals is not None:
        population = select_s0_population(population, selected_ordinals)
    _require(
        population.prompt_population.logical_prompt_count
        == population.prompt_population.unique_prompt_count,
        "matched S0-v2 requires one unique provider prompt per question",
    )
    return population


def _runtime(
    population: MatchedS0Population,
    *,
    checkpoint_dir: str | Path,
    client: Any | None,
    max_concurrency: int,
    preflight_artifact_sha256: str,
    profile: S0ExecutionProfile = V2_EXECUTION_PROFILE,
) -> FastCompletionRuntime:
    require_sha256(preflight_artifact_sha256, "preflight artifact SHA-256")
    return FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=[
            _plain_messages(row.rendered_prompt.messages) for row in population.rows
        ],
        model=DEFAULT_TERRA_GATEWAY_MODEL,
        client=client,
        max_prompt_tokens=population.max_prompt_tokens,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "answer_plan_id": profile.answer_plan_id,
            "arm_label": profile.arm_label,
            "authorized_unique_calls": (
                population.prompt_population.unique_prompt_count
            ),
            "gateway_url": DEFAULT_GATEWAY_URL,
            "gold_loaded": False,
            "matched_population_id": population.population_id,
            "preflight_artifact_sha256": preflight_artifact_sha256,
            "renderer_id": profile.renderer_id,
            "retrieval_sha256": population.retrieval_sha256,
            "snapshot_id": population.snapshot.snapshot_id,
        },
    )


def _answer_artifact(
    population: MatchedS0Population,
    batch: FastCompletionBatch,
    *,
    preflight_artifact_sha256: str,
    profile: S0ExecutionProfile = V2_EXECUTION_PROFILE,
) -> dict[str, Any]:
    _require(
        batch.prompt_population.prompt_population_sha256
        == population.prompt_population.prompt_population_sha256,
        "Terra prompt population changed after matched rendering",
    )
    records = {row.messages_sha256: row for row in batch.unique_records}
    questions: list[dict[str, Any]] = []
    for source, prediction in zip(
        population.rows, batch.logical_completions, strict=True
    ):
        prompt = source.rendered_prompt
        record = records[prompt.messages_sha256]
        prediction_sha256 = quote_sha256(prediction)
        _require(
            prediction_sha256 == record.completion_sha256,
            f"Terra completion changed at ordinal {source.ordinal}",
        )
        body: dict[str, Any] = {
            "call_key_sha256": record.call_key_sha256,
            "dated_question_sha256": source.packet.dated_question_sha256,
            "messages_sha256": prompt.messages_sha256,
            "ordinal": source.ordinal,
            "packet_id": source.packet.packet_id,
            "prediction": prediction,
            "prediction_sha256": prediction_sha256,
            "prompt_id": prompt.prompt_id,
            "prompt_token_proxy": prompt.total_prompt_token_proxy,
            "question_id": source.packet.question_id,
            "question_part_sha256": source.question_part_sha256,
            "question_sha256": source.packet.question_sha256,
            "request_journal_sha256": record.request_journal_sha256,
            "response_journal_sha256": record.response_journal_sha256,
            "source_stage_id": source.packet.stage_id,
            "source_stage_receipt_sha256": source.source_stage_receipt_sha256,
        }
        if profile.renderer_id in (V3_RENDERER_ID, V4_RENDERER_ID):
            body["alias_receipt_sha256"] = identity_sha256(
                [row.projection() for row in prompt.alias_receipt]
            )
        body["source_row_sha256"] = identity_sha256(body)
        questions.append(body)

    result: dict[str, Any] = {
        "arm_label": profile.arm_label,
        "completion_batch": _stable_batch(batch),
        "format": profile.answer_run_format,
        "gold_loaded": False,
        "logical_prediction_count": len(questions),
        "matched_population_id": population.population_id,
        "population_identity_sha256": (
            population.snapshot.population_identity_sha256
        ),
        "preflight_artifact_sha256": preflight_artifact_sha256,
        "preflight_identity_sha256": population.preflight_sha256,
        "prompt_population_sha256": (
            population.prompt_population.prompt_population_sha256
        ),
        "provider_route": {
            "caller_model": DEFAULT_TERRA_CALLER_MODEL,
            "gateway_model": DEFAULT_TERRA_GATEWAY_MODEL,
            "gateway_url": DEFAULT_GATEWAY_URL,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "max_prompt_tokens": population.max_prompt_tokens,
            "retries": 0,
        },
        "question_count": len(questions),
        "questions": questions,
        "renderer_id": profile.renderer_id,
        "retained_request_token_state_bytes": 0,
        "retrieval_sha256": population.retrieval_sha256,
        "snapshot_id": population.snapshot.snapshot_id,
        "unique_provider_prompt_count": (
            batch.prompt_population.unique_prompt_count
        ),
    }
    assert_gold_blind(result, path="s0_v2_answer_run")
    return result


def _runtime_entries(
    population: MatchedS0Population,
    answer_artifact: Mapping[str, Any],
    *,
    profile: S0ExecutionProfile = V2_EXECUTION_PROFILE,
) -> tuple[RuntimeLedgerEntry, ...]:
    raw_questions = answer_artifact.get("questions")
    _require(type(raw_questions) is list, "answer run questions must be an array")
    _require(
        len(raw_questions) == len(population.rows),
        "answer run question count changed",
    )
    total = len(population.rows)
    entries: list[RuntimeLedgerEntry] = []
    for source, raw in zip(population.rows, raw_questions, strict=True):
        _require(type(raw) is dict, "answer run row must be an exact object")
        source_row_sha256 = raw.get("source_row_sha256")
        unsigned_source_row = dict(raw)
        unsigned_source_row.pop("source_row_sha256", None)
        _require(
            source_row_sha256 == identity_sha256(unsigned_source_row),
            f"answer source-row seal changed at ordinal {source.ordinal}",
        )
        prediction = raw.get("prediction")
        prediction_sha256 = raw.get("prediction_sha256")
        _require(
            type(prediction) is str
            and bool(prediction)
            and prediction_sha256 == quote_sha256(prediction),
            f"answer prediction changed at ordinal {source.ordinal}",
        )
        entries.append(
            RuntimeLedgerEntry(
                event_type="answer_observation",
                ordinal=source.ordinal,
                question_id=source.packet.question_id,
                question_sha256=source.packet.question_sha256,
                arm_label=profile.arm_label,
                parent_arm_label=None,
                stage_id={
                    RENDERER_ID: "S0_V2_TERRA_ANSWER",
                    V3_RENDERER_ID: "S0_V3_TERRA_ANSWER",
                    V4_RENDERER_ID: "S0_V4_TERRA_ANSWER",
                }[profile.renderer_id],
                parent_stage_id=SOURCE_STAGE_ID,
                mechanism_id="terra_responder",
                delta_kind="observation",
                renderer_id=profile.renderer_id,
                legacy_renderer=False,
                disposition=StageDisposition.NO_OP,
                provider_calls=1,
                provider_prompt_cap=1,
                provider_prompt_reserved=1,
                global_provider_prompt_cap=total,
                max_final_prompt_tokens=population.max_prompt_tokens,
                prompt_token_proxy=source.rendered_prompt.total_prompt_token_proxy,
                parent_packet_sha256=source.packet.packet_id,
                packet_sha256=source.packet.packet_id,
                prompt_id=source.rendered_prompt.prompt_id,
                prompt_messages_sha256=source.rendered_prompt.messages_sha256,
                prediction=prediction,
                prediction_sha256=prediction_sha256,
                source_row_sha256=source_row_sha256,
                reason={
                    RENDERER_ID: "sealed_terra_s0_v2_prediction",
                    V3_RENDERER_ID: "sealed_terra_s0_v3_prediction",
                    V4_RENDERER_ID: "sealed_terra_s0_v4_prediction",
                }[profile.renderer_id],
            )
        )
    return tuple(entries)


def _runtime_ledger(
    population: MatchedS0Population,
    answer_artifact: Mapping[str, Any],
    *,
    answer_artifact_sha256: str,
    preflight_artifact_sha256: str,
    profile: S0ExecutionProfile = V2_EXECUTION_PROFILE,
) -> dict[str, Any]:
    return build_runtime_ledger(
        snapshot_id=population.snapshot.snapshot_id,
        plan_id=profile.answer_plan_id,
        entries=_runtime_entries(population, answer_artifact, profile=profile),
        source_artifacts=(
            {
                "role": f"{profile.arm_label}:sealed_retrieval",
                "sha256": population.retrieval_sha256,
            },
            {
                "role": f"{profile.arm_label}:preflight",
                "sha256": preflight_artifact_sha256,
            },
            {
                "role": f"{profile.arm_label}:run",
                "sha256": answer_artifact_sha256,
            },
        ),
    )


@dataclass(frozen=True, slots=True)
class S0V2AnswerRunResult:
    answer_artifact: SealedArtifact
    runtime_ledger_artifact: SealedArtifact
    physical_provider_calls: int
    checkpoint_hits: int


@dataclass(frozen=True, slots=True)
class VerifiedS0V2AnswerRow:
    ordinal: int
    question_id: str
    question_sha256: str
    dated_question_sha256: str
    messages_sha256: str
    prediction: str
    prediction_sha256: str
    call_key_sha256: str
    request_journal_sha256: str
    response_journal_sha256: str
    source_row_sha256: str
    runtime_row_id: str
    alias_receipt_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class VerifiedS0V2AnswerPlane:
    run_sha256: str
    replay_sha256: str
    matched_population_id: str
    population_identity_sha256: str
    snapshot_id: str
    renderer_id: str
    runtime_ledger: Mapping[str, Any]
    runtime_ledger_sha256: str
    rows: tuple[VerifiedS0V2AnswerRow, ...]

    @property
    def runtime_ledger_projection(self) -> Mapping[str, Any]:
        """Public name for the deeply immutable runtime-ledger projection."""

        return self.runtime_ledger

    @property
    def ordered_rows(self) -> tuple[VerifiedS0V2AnswerRow, ...]:
        """Public ordered answer rows, including journal provenance."""

        return self.rows

    def runtime_ledger_json(self) -> dict[str, Any]:
        """Return an isolated exact-JSON copy for strict ledger validators."""

        projection = _thaw_json(self.runtime_ledger)
        assert type(projection) is dict
        return projection


def preflight_s0_v2_answers(
    *,
    retrieval_path: str | Path,
    output_root: str | Path,
    expected_retrieval_sha256: str | None = EXPECTED_RETRIEVAL_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
    renderer_id: str = RENDERER_ID,
    selected_ordinals: Sequence[int] | None = None,
) -> SealedArtifact:
    profile = execution_profile(renderer_id)
    population = _load_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
        renderer_id=profile.renderer_id,
        selected_ordinals=selected_ordinals,
    )
    return _preflight_s0_v2_answers_for_population(
        population=population,
        output_root=output_root,
        profile=profile,
    )


def _preflight_s0_v2_answers_for_population(
    *,
    population: MatchedS0Population,
    output_root: str | Path,
    profile: S0ExecutionProfile,
) -> SealedArtifact:
    """Publish preflight for one already verified immutable population."""

    _require(
        population.renderer_id == profile.renderer_id,
        "answer preflight population renderer changed",
    )
    artifact, _created = publish_sealed_json(
        Path(output_root) / profile.preflight_name,
        population.preflight_projection(),
    )
    return artifact


def run_s0_v2_answers(
    *,
    retrieval_path: str | Path,
    output_root: str | Path,
    enable_provider: bool,
    authorized_provider_calls: int,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    max_concurrency: int = 4,
    expected_retrieval_sha256: str | None = EXPECTED_RETRIEVAL_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
    renderer_id: str = RENDERER_ID,
    selected_ordinals: Sequence[int] | None = None,
) -> S0V2AnswerRunResult:
    profile = execution_profile(renderer_id)
    population = _load_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
        renderer_id=profile.renderer_id,
        selected_ordinals=selected_ordinals,
    )
    return _run_s0_v2_answers_for_population(
        population=population,
        output_root=output_root,
        enable_provider=enable_provider,
        authorized_provider_calls=authorized_provider_calls,
        api_key_env=api_key_env,
        max_concurrency=max_concurrency,
        profile=profile,
    )


def _run_s0_v2_answers_for_population(
    *,
    population: MatchedS0Population,
    output_root: str | Path,
    enable_provider: bool,
    authorized_provider_calls: int,
    api_key_env: str,
    max_concurrency: int,
    profile: S0ExecutionProfile,
) -> S0V2AnswerRunResult:
    """Execute or replay answers for one verified immutable population."""

    _require(
        population.renderer_id == profile.renderer_id,
        "answer run population renderer changed",
    )
    required = population.prompt_population.unique_prompt_count
    _require(enable_provider, "S0-v2 answer run requires provider enablement")
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == required,
        f"authorized provider calls must exactly equal {required}",
    )
    output = Path(output_root)
    preflight, _created = publish_sealed_json(
        output / profile.preflight_name, population.preflight_projection()
    )
    existing_run = output / ANSWER_RUN_NAME
    if existing_run.exists():
        # A terminal artifact is a replay boundary, never permission to spend
        # another batch.  This also catches changed concurrency/runtime identity
        # or missing journals before a provider client is constructed.
        source = read_sealed_json(existing_run)
        _replay_s0_v2_answers_for_population(
            population=population,
            source=source,
            output_root=output,
            expected_run_sha256=source.sha256,
            max_concurrency=max_concurrency,
            profile=profile,
        )
        return S0V2AnswerRunResult(
            answer_artifact=source,
            runtime_ledger_artifact=read_sealed_json(
                output / RUNTIME_LEDGER_NAME
            ),
            physical_provider_calls=0,
            checkpoint_hits=required,
        )
    load_dotenv()
    api_key = os.environ.get(api_key_env, "").strip()
    _require(bool(api_key), f"provider API key is empty: {api_key_env}")
    client = _make_provider_client(api_key, DEFAULT_GATEWAY_URL)
    try:
        batch = _runtime(
            population,
            checkpoint_dir=output / CHECKPOINT_DIR_NAME,
            client=client,
            max_concurrency=max_concurrency,
            preflight_artifact_sha256=preflight.sha256,
            profile=profile,
        ).run()
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    _require(
        batch.usage.physical_calls + batch.usage.checkpoint_hits == required,
        "Terra completion journal population changed",
    )
    payload = _answer_artifact(
        population,
        batch,
        preflight_artifact_sha256=preflight.sha256,
        profile=profile,
    )
    answer, _created = publish_sealed_json(output / ANSWER_RUN_NAME, payload)
    ledger = _runtime_ledger(
        population,
        payload,
        answer_artifact_sha256=answer.sha256,
        preflight_artifact_sha256=preflight.sha256,
        profile=profile,
    )
    runtime_ledger, _created = publish_sealed_json(
        output / RUNTIME_LEDGER_NAME, ledger
    )
    return S0V2AnswerRunResult(
        answer_artifact=answer,
        runtime_ledger_artifact=runtime_ledger,
        physical_provider_calls=batch.usage.physical_calls,
        checkpoint_hits=batch.usage.checkpoint_hits,
    )


def replay_s0_v2_answers(
    *,
    retrieval_path: str | Path,
    output_root: str | Path,
    expected_run_sha256: str,
    max_concurrency: int = 4,
    expected_retrieval_sha256: str | None = EXPECTED_RETRIEVAL_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
    renderer_id: str = RENDERER_ID,
    selected_ordinals: Sequence[int] | None = None,
) -> VerifiedS0V2AnswerPlane:
    profile = execution_profile(renderer_id)
    output = Path(output_root)
    source = read_sealed_json(output / ANSWER_RUN_NAME)
    _require(source.sha256 == expected_run_sha256, "answer run SHA-256 changed")
    population = _load_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
        renderer_id=profile.renderer_id,
        selected_ordinals=selected_ordinals,
    )
    return _replay_s0_v2_answers_for_population(
        population=population,
        source=source,
        output_root=output,
        expected_run_sha256=expected_run_sha256,
        max_concurrency=max_concurrency,
        profile=profile,
    )


def _replay_s0_v2_answers_for_population(
    *,
    population: MatchedS0Population,
    source: SealedArtifact,
    output_root: str | Path,
    expected_run_sha256: str,
    max_concurrency: int,
    profile: S0ExecutionProfile,
) -> VerifiedS0V2AnswerPlane:
    """Replay one answer run against an already verified immutable population."""

    output = Path(output_root)
    _require(source.sha256 == expected_run_sha256, "answer run SHA-256 changed")
    _require(
        population.renderer_id == profile.renderer_id,
        "answer replay population renderer changed",
    )
    preflight = read_sealed_json(output / profile.preflight_name)
    _require(
        preflight.payload == population.preflight_projection(),
        "answer preflight changed during replay",
    )
    batch = _runtime(
        population,
        checkpoint_dir=output / CHECKPOINT_DIR_NAME,
        client=None,
        max_concurrency=max_concurrency,
        preflight_artifact_sha256=preflight.sha256,
        profile=profile,
    ).run()
    _require(batch.usage.physical_calls == 0, "answer replay made provider calls")
    _require(
        batch.usage.checkpoint_hits
        == population.prompt_population.unique_prompt_count,
        "answer replay checkpoint population changed",
    )
    expected = _answer_artifact(
        population,
        batch,
        preflight_artifact_sha256=preflight.sha256,
        profile=profile,
    )
    _require(
        canonical_json_bytes(expected) == canonical_json_bytes(source.payload),
        "answer run differs from immutable Terra journals",
    )
    replay, _created = publish_sealed_json(output / ANSWER_REPLAY_NAME, expected)
    expected_ledger = _runtime_ledger(
        population,
        expected,
        answer_artifact_sha256=source.sha256,
        preflight_artifact_sha256=preflight.sha256,
        profile=profile,
    )
    ledger = read_sealed_json(output / RUNTIME_LEDGER_NAME)
    _require(
        canonical_json_bytes(expected_ledger) == canonical_json_bytes(ledger.payload),
        "runtime ledger differs from replayed answer observations",
    )
    publish_sealed_json(output / RUNTIME_LEDGER_REPLAY_NAME, expected_ledger)
    return _verified_plane(
        population=population,
        run=source,
        replay=replay,
        runtime_ledger=ledger,
    )


def _verified_plane(
    *,
    population: MatchedS0Population,
    run: SealedArtifact,
    replay: SealedArtifact,
    runtime_ledger: SealedArtifact,
) -> VerifiedS0V2AnswerPlane:
    ledger_rows = runtime_ledger.payload.get("rows")
    answer_rows = run.payload.get("questions")
    _require(
        type(ledger_rows) is list
        and type(answer_rows) is list
        and len(ledger_rows) == len(answer_rows) == len(population.rows),
        "verified answer plane row count changed",
    )
    rows: list[VerifiedS0V2AnswerRow] = []
    for source, raw, ledger in zip(
        population.rows, answer_rows, ledger_rows, strict=True
    ):
        _require(type(raw) is dict and type(ledger) is dict, "answer row changed")
        _require(
            raw.get("ordinal") == source.ordinal
            and raw.get("question_id") == source.packet.question_id
            and raw.get("question_sha256") == source.packet.question_sha256
            and raw.get("dated_question_sha256")
            == source.packet.dated_question_sha256
            and ledger.get("ordinal") == source.ordinal
            and ledger.get("question_id") == source.packet.question_id
            and ledger.get("prediction_sha256") == raw.get("prediction_sha256"),
            f"answer/runtime row binding changed at ordinal {source.ordinal}",
        )
        for key in (
            "messages_sha256",
            "prediction_sha256",
            "call_key_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
            "source_row_sha256",
        ):
            require_sha256(str(raw.get(key)), f"answer row {source.ordinal} {key}")
        alias_receipt_sha256 = raw.get("alias_receipt_sha256")
        expected_alias_receipt_sha256 = (
            identity_sha256(
                [
                    row.projection()
                    for row in source.rendered_prompt.alias_receipt
                ]
            )
            if population.renderer_id in (V3_RENDERER_ID, V4_RENDERER_ID)
            else None
        )
        _require(
            alias_receipt_sha256 == expected_alias_receipt_sha256,
            f"answer alias receipt changed at ordinal {source.ordinal}",
        )
        require_sha256(
            str(ledger.get("row_id")),
            f"runtime row {source.ordinal} row ID",
        )
        rows.append(
            VerifiedS0V2AnswerRow(
                ordinal=source.ordinal,
                question_id=source.packet.question_id,
                question_sha256=source.packet.question_sha256,
                dated_question_sha256=source.packet.dated_question_sha256,
                messages_sha256=str(raw["messages_sha256"]),
                prediction=str(raw["prediction"]),
                prediction_sha256=str(raw["prediction_sha256"]),
                call_key_sha256=str(raw["call_key_sha256"]),
                request_journal_sha256=str(raw["request_journal_sha256"]),
                response_journal_sha256=str(raw["response_journal_sha256"]),
                source_row_sha256=str(raw["source_row_sha256"]),
                runtime_row_id=str(ledger["row_id"]),
                alias_receipt_sha256=(
                    str(alias_receipt_sha256)
                    if alias_receipt_sha256 is not None
                    else None
                ),
            )
        )
    return VerifiedS0V2AnswerPlane(
        run_sha256=run.sha256,
        replay_sha256=replay.sha256,
        matched_population_id=population.population_id,
        population_identity_sha256=population.snapshot.population_identity_sha256,
        snapshot_id=population.snapshot.snapshot_id,
        renderer_id=population.renderer_id,
        runtime_ledger=_freeze_json(runtime_ledger.payload),
        runtime_ledger_sha256=runtime_ledger.sha256,
        rows=tuple(rows),
    )


def load_verified_s0_v2_answer_plane(
    run_path: str | Path,
    replay_path: str | Path,
    *,
    expected_run_sha256: str,
    retrieval_path: str | Path,
    checkpoint_dir: str | Path | None = None,
    max_concurrency: int = 4,
    expected_retrieval_sha256: str | None = EXPECTED_RETRIEVAL_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
    renderer_id: str = RENDERER_ID,
    selected_ordinals: Sequence[int] | None = None,
) -> VerifiedS0V2AnswerPlane:
    """Verify run, replay, journals, and runtime ledger without loading gold."""

    profile = execution_profile(renderer_id)
    population = _load_population(
        retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_question_count=expected_question_count,
        renderer_id=profile.renderer_id,
        selected_ordinals=selected_ordinals,
    )
    return _load_verified_s0_v2_answer_plane_for_population(
        population=population,
        run_path=run_path,
        replay_path=replay_path,
        expected_run_sha256=expected_run_sha256,
        checkpoint_dir=checkpoint_dir,
        max_concurrency=max_concurrency,
        profile=profile,
    )


def _load_verified_s0_v2_answer_plane_for_population(
    *,
    population: MatchedS0Population,
    run_path: str | Path,
    replay_path: str | Path,
    expected_run_sha256: str,
    checkpoint_dir: str | Path | None,
    max_concurrency: int,
    profile: S0ExecutionProfile,
) -> VerifiedS0V2AnswerPlane:
    """Verify a parent answer plane against one already loaded population."""

    _require(
        population.renderer_id == profile.renderer_id,
        "verified answer population renderer changed",
    )
    run = read_sealed_json(run_path)
    replay = read_sealed_json(replay_path)
    _require(
        run.sha256 == replay.sha256 == expected_run_sha256,
        "answer run/replay SHA-256 binding changed",
    )
    _require(
        canonical_json_bytes(run.payload) == canonical_json_bytes(replay.payload),
        "answer run and replay differ",
    )
    root = Path(run_path).parent
    preflight = read_sealed_json(root / profile.preflight_name)
    _require(
        preflight.payload == population.preflight_projection(),
        "verified answer preflight changed",
    )
    batch = _runtime(
        population,
        checkpoint_dir=checkpoint_dir or root / CHECKPOINT_DIR_NAME,
        client=None,
        max_concurrency=max_concurrency,
        preflight_artifact_sha256=preflight.sha256,
        profile=profile,
    ).run()
    _require(batch.usage.physical_calls == 0, "answer verification made provider calls")
    _require(
        batch.usage.checkpoint_hits
        == population.prompt_population.unique_prompt_count,
        "answer verification checkpoint population changed",
    )
    expected = _answer_artifact(
        population,
        batch,
        preflight_artifact_sha256=preflight.sha256,
        profile=profile,
    )
    _require(
        canonical_json_bytes(run.payload) == canonical_json_bytes(expected),
        "verified answer run differs from Terra journals",
    )
    ledger = read_sealed_json(root / RUNTIME_LEDGER_NAME)
    expected_ledger = _runtime_ledger(
        population,
        expected,
        answer_artifact_sha256=run.sha256,
        preflight_artifact_sha256=preflight.sha256,
        profile=profile,
    )
    _require(
        canonical_json_bytes(ledger.payload) == canonical_json_bytes(expected_ledger),
        "verified runtime ledger differs from answer run",
    )
    ledger_replay = read_sealed_json(root / RUNTIME_LEDGER_REPLAY_NAME)
    _require(
        ledger_replay.sha256 == ledger.sha256
        and canonical_json_bytes(ledger_replay.payload)
        == canonical_json_bytes(ledger.payload),
        "runtime ledger and replay differ",
    )
    return _verified_plane(
        population=population,
        run=run,
        replay=replay,
        runtime_ledger=ledger,
    )


def _v3_arguments(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(kwargs)
    if "renderer_id" in values:
        raise TypeError("v3 wrappers own the renderer identity")
    values["renderer_id"] = V3_RENDERER_ID
    return values


def preflight_s0_v3_answers(**kwargs: Any) -> SealedArtifact:
    return preflight_s0_v2_answers(**_v3_arguments(kwargs))


def run_s0_v3_answers(**kwargs: Any) -> S0V2AnswerRunResult:
    return run_s0_v2_answers(**_v3_arguments(kwargs))


def replay_s0_v3_answers(**kwargs: Any) -> VerifiedS0V2AnswerPlane:
    return replay_s0_v2_answers(**_v3_arguments(kwargs))


def load_verified_s0_v3_answer_plane(
    run_path: str | Path,
    replay_path: str | Path,
    **kwargs: Any,
) -> VerifiedS0V2AnswerPlane:
    return load_verified_s0_v2_answer_plane(
        run_path,
        replay_path,
        **_v3_arguments(kwargs),
    )


def _v4_arguments(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(kwargs)
    if "renderer_id" in values:
        raise TypeError("v4 wrappers own the renderer identity")
    values["renderer_id"] = V4_RENDERER_ID
    return values


def preflight_s0_v4_answers(**kwargs: Any) -> SealedArtifact:
    return preflight_s0_v2_answers(**_v4_arguments(kwargs))


def run_s0_v4_answers(**kwargs: Any) -> S0V2AnswerRunResult:
    return run_s0_v2_answers(**_v4_arguments(kwargs))


def replay_s0_v4_answers(**kwargs: Any) -> VerifiedS0V2AnswerPlane:
    return replay_s0_v2_answers(**_v4_arguments(kwargs))


def load_verified_s0_v4_answer_plane(
    run_path: str | Path,
    replay_path: str | Path,
    **kwargs: Any,
) -> VerifiedS0V2AnswerPlane:
    return load_verified_s0_v2_answer_plane(
        run_path,
        replay_path,
        **_v4_arguments(kwargs),
    )


__all__ = [
    "ANSWER_PLAN_ID",
    "ANSWER_RUN_FORMAT",
    "ANSWER_RUN_NAME",
    "ANSWER_REPLAY_NAME",
    "ARM_LABEL",
    "CHECKPOINT_DIR_NAME",
    "DEFAULT_API_KEY_ENV",
    "DEFAULT_GATEWAY_URL",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_TERRA_CALLER_MODEL",
    "DEFAULT_TERRA_GATEWAY_MODEL",
    "PREFLIGHT_NAME",
    "S0ExecutionProfile",
    "RUNTIME_LEDGER_NAME",
    "RUNTIME_LEDGER_REPLAY_NAME",
    "S0V2AnswerRunResult",
    "V2_EXECUTION_PROFILE",
    "V3_ANSWER_PLAN_ID",
    "V3_ANSWER_RUN_FORMAT",
    "V3_ARM_LABEL",
    "V3_EXECUTION_PROFILE",
    "V3_PREFLIGHT_NAME",
    "V4_ANSWER_PLAN_ID",
    "V4_ANSWER_RUN_FORMAT",
    "V4_ARM_LABEL",
    "V4_EXECUTION_PROFILE",
    "V4_PREFLIGHT_NAME",
    "VerifiedS0V2AnswerPlane",
    "VerifiedS0V2AnswerRow",
    "load_verified_s0_v2_answer_plane",
    "load_verified_s0_v3_answer_plane",
    "load_verified_s0_v4_answer_plane",
    "preflight_s0_v2_answers",
    "preflight_s0_v3_answers",
    "preflight_s0_v4_answers",
    "replay_s0_v2_answers",
    "replay_s0_v3_answers",
    "replay_s0_v4_answers",
    "run_s0_v2_answers",
    "run_s0_v3_answers",
    "run_s0_v4_answers",
]
