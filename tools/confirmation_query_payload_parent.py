#!/usr/bin/env python3
"""Authenticated confirmation parent for the query-payload answer arm.

The frozen confirmation S0 lifecycle uses the registered V4 compact renderer,
while the historical query-payload arm consumes the shared
``VerifiedS0V2AnswerPlane`` carrier (also used by the V3/V4 live wrappers).
This module performs the explicit, provider-free bridge.  It rebuilds and
authenticates the protected confirmation S0 artifact and its sealed Terra
completion, then publishes a canonical answer projection plus a normalized
runtime ledger before exposing the exact shared carrier.

The later query-expansion and query-payload helpers in this module accept only
gold-blind, replayed artifacts.  They never load benchmark labels and never
select rows by ordinal or question ID.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from memory_condense.domain.discourse import quote_sha256
from tools.confirmation_contracts import (
    SealedJson as ConfirmationSealedJson,
    read_sealed_json as read_confirmation_sealed_json,
)
from tools.confirmation_protected_s0_plane import (
    ProtectedS0AnswerPlane,
    build_protected_s0_answer_plane,
)
from tools.confirmation_terra_completion_lifecycle import (
    read_sealed_artifact as read_confirmation_lifecycle_artifact,
)
from tools.confirmation_query_artifacts import VerifiedQueryExpansionArtifacts
from tools.matched_eval import live, query_payload_live
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    StageDisposition,
    assert_gold_blind,
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
)
from tools.matched_eval.ledger import (
    RuntimeLedgerEntry,
    _validated_runtime_ledger,
    build_runtime_ledger,
)
from tools.matched_eval.query_expansion import PLAN_ID as QUERY_EXPANSION_PLAN_ID
from tools.matched_eval.query_fact_adapter import (
    QueryFactAdapterPopulation,
    build_query_fact_population,
)
from tools.matched_eval.query_payload_live import (
    QueryPayloadAnswerPlan,
    QueryPayloadAnswerRunResult,
    QueryPayloadProviderResult,
    VerifiedQueryPayloadAnswerPlane,
    build_query_payload_answer_plan,
    load_query_payload_answer_provider_journals,
    materialize_query_payload_answers,
    preflight_query_payload_answers,
    replay_query_payload_answers,
    run_query_payload_answer_provider,
)
from tools.matched_eval.renderer import V4_RENDERER_ID
from tools.confirmation_canonical import assert_snapshot_unchanged


PARENT_RUN_FORMAT = "memory-condense-confirmation-protected-s0-live-parent-v1"
PARENT_ROW_FORMAT = f"{PARENT_RUN_FORMAT}-row-v1"
PARENT_BRIDGE_FORMAT = f"{PARENT_RUN_FORMAT}-bridge-v1"
PARENT_PLAN_ID = "confirmation_protected_s0_live_parent_v1"
PARENT_STAGE_ID = "confirmation_protected_s0_terra_answer"

PARENT_DIR_NAME = "protected-s0-live-parent"
PARENT_RUN_NAME = "answer-run.json"
PARENT_REPLAY_NAME = "answer-run-replay.json"
PARENT_LEDGER_NAME = "runtime-ledger.json"
PARENT_LEDGER_REPLAY_NAME = "runtime-ledger-replay.json"
PARENT_BRIDGE_NAME = "parent-bridge.json"

PROMPT_FORMAT = "memory-condense-confirmation-query-payload-terra-prompts-v1"
PROMPT_ROW_FORMAT = f"{PROMPT_FORMAT}-row-v1"
PROVIDER_INPUT_FORMAT = f"{PROMPT_FORMAT}-provider-input-v1"
RELEASE_FORMAT = "memory-condense-confirmation-query-payload-provider-release-v1"
PROMPT_NAME = "confirmation-query-payload-terra-prompts-v1.json"
RELEASE_NAME = "confirmation-query-payload-provider-release-v1.json"

_JOURNAL_NAME = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)
_RELEASE_KEYS = frozenset(
    {
        "answer_preflight_sha256",
        "approval_opt_in",
        "checkpoint_snapshot",
        "format",
        "gold_loaded",
        "output_root",
        "output_root_sha256",
        "physical_provider_calls",
        "prompt_plane_sha256",
        "release_identity_sha256",
        "release_status",
        "required_authorized_provider_calls",
        "unsafe_retry_policy",
    }
)
_CHECKPOINT_SNAPSHOT_KEYS = frozenset(
    {
        "authenticated_complete_count",
        "ordered_records",
        "ordered_records_sha256",
    }
)
_CHECKPOINT_RECORD_KEYS = frozenset(
    {
        "messages_sha256",
        "call_key_sha256",
        "request_journal_sha256",
        "response_journal_sha256",
    }
)


class ConfirmationQueryPayloadError(MatchedEvalContractError):
    """A confirmation query-payload parent or lifecycle invariant failed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationQueryPayloadError(message)


@dataclass(frozen=True, slots=True)
class VerifiedProtectedS0Parent:
    """The authenticated source and exact shared parent carrier."""

    source_artifact: ConfirmationSealedJson
    source_plane: ProtectedS0AnswerPlane
    bridge_artifact: SealedArtifact
    run_artifact: SealedArtifact
    replay_artifact: SealedArtifact
    runtime_ledger_artifact: SealedArtifact
    parent_plane: live.VerifiedS0V2AnswerPlane

    def __post_init__(self) -> None:
        _require(
            type(self.parent_plane) is live.VerifiedS0V2AnswerPlane,
            "protected parent is not the exact shared answer-plane type",
        )
        _require(
            self.run_artifact.sha256
            == self.replay_artifact.sha256
            == self.parent_plane.run_sha256
            == self.parent_plane.replay_sha256,
            "protected parent run/replay binding changed",
        )
        _require(
            self.runtime_ledger_artifact.sha256
            == self.parent_plane.runtime_ledger_sha256,
            "protected parent runtime-ledger binding changed",
        )

    @property
    def exact_parent(self) -> live.VerifiedS0V2AnswerPlane:
        return self.parent_plane


def _completion_rows(
    plane: ProtectedS0AnswerPlane,
    *,
    completion_path: str | Path,
    expected_completion_sha256: str,
) -> tuple[dict[str, Any], ...]:
    completion = read_confirmation_lifecycle_artifact(
        completion_path,
        expected_sha256=expected_completion_sha256,
        label="confirmation S0 Terra completion",
    )
    raw_rows = completion.payload.get("ordered_rows")
    _require(
        type(raw_rows) is list
        and all(type(row) is dict for row in raw_rows)
        and len(raw_rows) == len(plane.source_population.rows),
        "confirmation S0 completion population changed after authentication",
    )
    return tuple(dict(row) for row in raw_rows)


def _parent_rows(
    source: ProtectedS0AnswerPlane,
    completion_rows: tuple[dict[str, Any], ...],
) -> tuple[tuple[dict[str, Any], RuntimeLedgerEntry], ...]:
    protected_rows = source.payload.get("ordered_rows")
    _require(
        type(protected_rows) is list
        and len(protected_rows) == len(source.source_population.rows),
        "protected S0 answer rows changed",
    )
    total = len(source.source_population.rows)
    result: list[tuple[dict[str, Any], RuntimeLedgerEntry]] = []
    for matched, raw_protected, completion in zip(
        source.source_population.rows,
        protected_rows,
        completion_rows,
        strict=True,
    ):
        _require(
            type(raw_protected) is dict
            and raw_protected.get("row_index") == matched.ordinal
            and raw_protected.get("question_id") == matched.packet.question_id
            and raw_protected.get("question_sha256")
            == matched.packet.question_sha256
            and raw_protected.get("dated_question_sha256")
            == matched.packet.dated_question_sha256
            and raw_protected.get("messages_sha256")
            == matched.rendered_prompt.messages_sha256,
            f"protected S0 row binding changed at ordinal {matched.ordinal}",
        )
        prediction = completion.get("completion")
        prediction_sha = completion.get("completion_sha256")
        _require(
            type(prediction) is str
            and bool(prediction)
            and prediction_sha == quote_sha256(prediction)
            and raw_protected.get("prediction") == prediction
            and raw_protected.get("prediction_sha256") == prediction_sha
            and raw_protected.get("completion_row_receipt_sha256")
            == completion.get("completion_row_receipt_sha256")
            and completion.get("question_id") == matched.packet.question_id
            and completion.get("messages_sha256")
            == matched.rendered_prompt.messages_sha256,
            f"protected S0 completion binding changed at ordinal {matched.ordinal}",
        )
        for key in (
            "call_key_sha256",
            "request_journal_sha256",
            "response_journal_sha256",
            "completion_row_receipt_sha256",
        ):
            require_sha256(
                str(completion.get(key)),
                f"protected S0 row {matched.ordinal} {key}",
            )
        aliases = [row.projection() for row in matched.rendered_prompt.alias_receipt]
        alias_receipt_sha = identity_sha256(aliases)
        body: dict[str, Any] = {
            "alias_receipt_sha256": alias_receipt_sha,
            "call_key_sha256": completion["call_key_sha256"],
            "completion_row_receipt_sha256": completion[
                "completion_row_receipt_sha256"
            ],
            "dated_question_sha256": matched.packet.dated_question_sha256,
            "format": PARENT_ROW_FORMAT,
            "messages_sha256": matched.rendered_prompt.messages_sha256,
            "ordinal": matched.ordinal,
            "packet_id": matched.packet.packet_id,
            "prediction": prediction,
            "prediction_sha256": prediction_sha,
            "protected_row_receipt_sha256": raw_protected["row_receipt_sha256"],
            "prompt_id": matched.rendered_prompt.prompt_id,
            "question_id": matched.packet.question_id,
            "question_sha256": matched.packet.question_sha256,
            "request_journal_sha256": completion["request_journal_sha256"],
            "response_journal_sha256": completion["response_journal_sha256"],
            "source_stage_receipt_sha256": matched.source_stage_receipt_sha256,
        }
        source_row_sha = identity_sha256(body)
        projected = {**body, "source_row_sha256": source_row_sha}
        entry = RuntimeLedgerEntry(
            event_type="answer_observation",
            ordinal=matched.ordinal,
            question_id=matched.packet.question_id,
            question_sha256=matched.packet.question_sha256,
            arm_label=live.V4_ARM_LABEL,
            parent_arm_label=None,
            stage_id=PARENT_STAGE_ID,
            parent_stage_id=matched.packet.stage_id,
            mechanism_id="sealed_confirmation_terra_s0_v4_projection",
            delta_kind="observation",
            renderer_id=V4_RENDERER_ID,
            legacy_renderer=False,
            disposition=StageDisposition.NO_OP,
            provider_calls=1,
            provider_prompt_cap=1,
            provider_prompt_reserved=1,
            global_provider_prompt_cap=total,
            max_final_prompt_tokens=source.source_population.max_prompt_tokens,
            prompt_token_proxy=matched.rendered_prompt.total_prompt_token_proxy,
            parent_packet_sha256=matched.packet.packet_id,
            packet_sha256=matched.packet.packet_id,
            prompt_id=matched.rendered_prompt.prompt_id,
            prompt_messages_sha256=matched.rendered_prompt.messages_sha256,
            prediction=prediction,
            prediction_sha256=prediction_sha,
            changed_from_parent=False,
            source_row_sha256=source_row_sha,
            reason="authenticated_sealed_confirmation_s0_prediction",
        )
        result.append((projected, entry))
    return tuple(result)


def _build_exact_parent(
    *,
    source_artifact: ConfirmationSealedJson,
    source_plane: ProtectedS0AnswerPlane,
    completion_rows: tuple[dict[str, Any], ...],
    s0_prompt_sha256: str,
    s0_completion_sha256: str,
    output_root: Path,
) -> VerifiedProtectedS0Parent:
    projected = _parent_rows(source_plane, completion_rows)
    answer_rows = [row for row, _entry in projected]
    entries = tuple(entry for _row, entry in projected)
    run_body: dict[str, Any] = {
        "answer_plan_id": live.V4_ANSWER_PLAN_ID,
        "arm_label": live.V4_ARM_LABEL,
        "format": PARENT_RUN_FORMAT,
        "gold_loaded": False,
        "logical_prediction_count": len(answer_rows),
        "matched_population_id": source_plane.source_population.population_id,
        "population_identity_sha256": (
            source_plane.source_population.snapshot.population_identity_sha256
        ),
        "protected_s0_plane_sha256": source_artifact.sha256,
        "provider_calls_during_projection": 0,
        "question_count": len(answer_rows),
        "questions": answer_rows,
        "renderer_id": V4_RENDERER_ID,
        "retrieval_sha256": source_plane.source_population.retrieval_sha256,
        "s0_completion_sha256": s0_completion_sha256,
        "s0_prompt_sha256": s0_prompt_sha256,
        "snapshot_id": source_plane.source_population.snapshot.snapshot_id,
    }
    assert_gold_blind(run_body, path="confirmation_query_payload_parent_run")
    run, _created = publish_sealed_json(output_root / PARENT_RUN_NAME, run_body)
    replay, _created = publish_sealed_json(
        output_root / PARENT_REPLAY_NAME, run_body
    )
    _require(
        run.sha256 == replay.sha256
        and canonical_json_bytes(run.payload) == canonical_json_bytes(replay.payload),
        "protected parent answer run/replay differ",
    )
    ledger_body = build_runtime_ledger(
        snapshot_id=source_plane.source_population.snapshot.snapshot_id,
        plan_id=PARENT_PLAN_ID,
        entries=entries,
        source_artifacts=(
            {"role": "protected_s0_plane", "sha256": source_artifact.sha256},
            {"role": "s0_prompt", "sha256": s0_prompt_sha256},
            {"role": "s0_completion", "sha256": s0_completion_sha256},
            {"role": "parent_answer_run", "sha256": run.sha256},
        ),
    )
    ledger, _created = publish_sealed_json(
        output_root / PARENT_LEDGER_NAME, ledger_body
    )
    ledger_replay, _created = publish_sealed_json(
        output_root / PARENT_LEDGER_REPLAY_NAME, ledger_body
    )
    _require(
        ledger.sha256 == ledger_replay.sha256,
        "protected parent runtime ledger/replay differ",
    )
    _identity, row_ids = _validated_runtime_ledger(ledger.payload)
    _require(
        len(row_ids) == len(answer_rows),
        "protected parent runtime observations are incomplete",
    )
    verified_rows = tuple(
        live.VerifiedS0V2AnswerRow(
            ordinal=matched.ordinal,
            question_id=matched.packet.question_id,
            question_sha256=matched.packet.question_sha256,
            dated_question_sha256=matched.packet.dated_question_sha256,
            messages_sha256=matched.rendered_prompt.messages_sha256,
            prediction=str(answer["prediction"]),
            prediction_sha256=str(answer["prediction_sha256"]),
            call_key_sha256=str(answer["call_key_sha256"]),
            request_journal_sha256=str(answer["request_journal_sha256"]),
            response_journal_sha256=str(answer["response_journal_sha256"]),
            source_row_sha256=str(answer["source_row_sha256"]),
            runtime_row_id=runtime_row_id,
            alias_receipt_sha256=str(answer["alias_receipt_sha256"]),
        )
        for matched, answer, runtime_row_id in zip(
            source_plane.source_population.rows,
            answer_rows,
            row_ids,
            strict=True,
        )
    )
    parent = live.VerifiedS0V2AnswerPlane(
        run_sha256=run.sha256,
        replay_sha256=replay.sha256,
        matched_population_id=source_plane.source_population.population_id,
        population_identity_sha256=(
            source_plane.source_population.snapshot.population_identity_sha256
        ),
        snapshot_id=source_plane.source_population.snapshot.snapshot_id,
        renderer_id=V4_RENDERER_ID,
        runtime_ledger=live._freeze_json(ledger.payload),  # noqa: SLF001
        runtime_ledger_sha256=ledger.sha256,
        rows=verified_rows,
    )
    bridge_body: dict[str, Any] = {
        "format": PARENT_BRIDGE_FORMAT,
        "gold_loaded": False,
        "parent_answer_run_sha256": run.sha256,
        "parent_answer_replay_sha256": replay.sha256,
        "parent_runtime_ledger_sha256": ledger.sha256,
        "physical_provider_calls": 0,
        "protected_s0_plane_sha256": source_artifact.sha256,
        "question_count": len(verified_rows),
        "renderer_id": V4_RENDERER_ID,
        "source_population_id": source_plane.source_population.population_id,
        "status": "verified",
    }
    bridge_body["bridge_identity_sha256"] = identity_sha256(bridge_body)
    assert_gold_blind(bridge_body, path="confirmation_query_payload_parent_bridge")
    bridge, _created = publish_sealed_json(
        output_root / PARENT_BRIDGE_NAME, bridge_body
    )
    return VerifiedProtectedS0Parent(
        source_artifact=source_artifact,
        source_plane=source_plane,
        bridge_artifact=bridge,
        run_artifact=run,
        replay_artifact=replay,
        runtime_ledger_artifact=ledger,
        parent_plane=parent,
    )


def materialize_verified_protected_s0_parent(
    *,
    protected_s0_plane_path: str | Path,
    expected_protected_s0_plane_sha256: str,
    output_root: str | Path,
    **protected_s0_inputs: Any,
) -> VerifiedProtectedS0Parent:
    """Rebuild the protected S0 source and expose its exact V4 live carrier."""

    expected_source = require_sha256(
        expected_protected_s0_plane_sha256,
        "expected protected S0 plane SHA-256",
    )
    source_artifact = read_confirmation_sealed_json(
        protected_s0_plane_path,
        expected_sha256=expected_source,
        label="protected S0 answer plane",
    )
    source_plane = build_protected_s0_answer_plane(**protected_s0_inputs)
    _require(
        source_artifact.payload == source_plane.payload,
        "protected S0 answer plane differs from authenticated reconstruction",
    )
    completion_path = protected_s0_inputs.get("s0_completion_path")
    completion_sha = protected_s0_inputs.get("expected_s0_completion_sha256")
    prompt_sha = protected_s0_inputs.get("expected_s0_prompt_sha256")
    _require(
        isinstance(completion_path, (str, Path)),
        "protected S0 completion path is missing",
    )
    completion_sha = require_sha256(
        str(completion_sha), "expected protected S0 completion SHA-256"
    )
    prompt_sha = require_sha256(
        str(prompt_sha), "expected protected S0 prompt SHA-256"
    )
    completion_rows = _completion_rows(
        source_plane,
        completion_path=completion_path,
        expected_completion_sha256=completion_sha,
    )
    result = _build_exact_parent(
        source_artifact=source_artifact,
        source_plane=source_plane,
        completion_rows=completion_rows,
        s0_prompt_sha256=prompt_sha,
        s0_completion_sha256=completion_sha,
        output_root=Path(output_root),
    )
    assert_snapshot_unchanged(source_artifact.snapshot, "protected S0 answer plane")
    assert_snapshot_unchanged(
        source_artifact.sidecar, "protected S0 answer plane sidecar"
    )
    return result


@dataclass(frozen=True, slots=True)
class ConfirmationQueryPayloadPlan:
    """All authenticated in-memory inputs for the shared query-payload arm."""

    protected_parent: VerifiedProtectedS0Parent
    query_expansion: VerifiedQueryExpansionArtifacts
    query_fact_population: QueryFactAdapterPopulation
    answer_plan: QueryPayloadAnswerPlan

    @property
    def required_calls(self) -> int:
        return self.answer_plan.required_calls


@dataclass(frozen=True, slots=True)
class ConfirmationQueryPayloadPreflight:
    plan: ConfirmationQueryPayloadPlan
    prompt_artifact: SealedArtifact
    answer_preflight_artifact: SealedArtifact


ClientFactory = Callable[[str, str], Any]


def _read_expected_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
    label: str,
) -> SealedArtifact:
    expected = require_sha256(expected_sha256, f"expected {label} SHA-256")
    artifact = read_sealed_json(path)
    _require(artifact.sha256 == expected, f"{label} SHA-256 changed")
    return artifact


def load_verified_query_expansion_artifacts(
    source_parent: VerifiedProtectedS0Parent,
    *,
    query_preflight_path: str | Path,
    expected_query_preflight_sha256: str,
    query_run_path: str | Path,
    query_run_replay_path: str | Path,
    expected_query_run_sha256: str,
    query_runtime_ledger_path: str | Path,
    query_runtime_ledger_replay_path: str | Path,
    expected_query_runtime_ledger_sha256: str,
) -> VerifiedQueryExpansionArtifacts:
    """Authenticate a complete query-expansion replay before downstream use."""

    if type(source_parent) is not VerifiedProtectedS0Parent:
        raise TypeError("source_parent must be an exact VerifiedProtectedS0Parent")
    preflight = _read_expected_artifact(
        query_preflight_path,
        expected_sha256=expected_query_preflight_sha256,
        label="query-expansion preflight",
    )
    run = _read_expected_artifact(
        query_run_path,
        expected_sha256=expected_query_run_sha256,
        label="query-expansion run",
    )
    replay = _read_expected_artifact(
        query_run_replay_path,
        expected_sha256=expected_query_run_sha256,
        label="query-expansion run replay",
    )
    _require(
        run.sha256 == replay.sha256
        and canonical_json_bytes(run.payload) == canonical_json_bytes(replay.payload),
        "query-expansion run/replay differ",
    )
    ledger = _read_expected_artifact(
        query_runtime_ledger_path,
        expected_sha256=expected_query_runtime_ledger_sha256,
        label="query-expansion runtime ledger",
    )
    ledger_replay = _read_expected_artifact(
        query_runtime_ledger_replay_path,
        expected_sha256=expected_query_runtime_ledger_sha256,
        label="query-expansion runtime ledger replay",
    )
    _require(
        ledger.sha256 == ledger_replay.sha256
        and canonical_json_bytes(ledger.payload)
        == canonical_json_bytes(ledger_replay.payload),
        "query-expansion runtime ledger/replay differ",
    )
    _validated_runtime_ledger(ledger.payload)
    population = source_parent.source_plane.source_population
    sources = {
        row.get("role"): row.get("sha256")
        for row in ledger.payload.get("source_artifacts", [])
        if type(row) is dict
    }
    _require(
        ledger.payload.get("plan_id") == QUERY_EXPANSION_PLAN_ID
        and ledger.payload.get("snapshot_id") == population.snapshot.snapshot_id
        and ledger.payload.get("question_count") == population.question_count
        and sources.get("sealed_retrieval") == population.retrieval_sha256
        and sources.get("query_expansion_preflight") == preflight.sha256
        and sources.get("query_expansion_run") == run.sha256,
        "query-expansion runtime ledger lost its source lineage",
    )
    assert_gold_blind(run.payload, path="confirmation_query_expansion_run")
    return VerifiedQueryExpansionArtifacts(
        preflight=preflight,
        run=run,
        run_replay=replay,
        runtime_ledger=ledger,
        runtime_ledger_replay=ledger_replay,
    )


def build_confirmation_query_payload_plan(
    protected_parent: VerifiedProtectedS0Parent,
    *,
    query_preflight_path: str | Path,
    expected_query_preflight_sha256: str,
    query_run_path: str | Path,
    query_run_replay_path: str | Path,
    expected_query_run_sha256: str,
    query_runtime_ledger_path: str | Path,
    query_runtime_ledger_replay_path: str | Path,
    expected_query_runtime_ledger_sha256: str,
    expected_query_population_id: str,
    expected_query_prompt_population_sha256: str,
    max_prompt_tokens: int = query_payload_live.MAX_PROMPT_TOKENS,
    output_token_reserve: int = query_payload_live.OUTPUT_TOKEN_RESERVE,
) -> ConfirmationQueryPayloadPlan:
    """Build the exact query-fact join and the shared direct-answer plan."""

    query = load_verified_query_expansion_artifacts(
        protected_parent,
        query_preflight_path=query_preflight_path,
        expected_query_preflight_sha256=expected_query_preflight_sha256,
        query_run_path=query_run_path,
        query_run_replay_path=query_run_replay_path,
        expected_query_run_sha256=expected_query_run_sha256,
        query_runtime_ledger_path=query_runtime_ledger_path,
        query_runtime_ledger_replay_path=query_runtime_ledger_replay_path,
        expected_query_runtime_ledger_sha256=(
            expected_query_runtime_ledger_sha256
        ),
    )
    source = protected_parent.source_plane.source_population
    facts = build_query_fact_population(
        source,
        query_preflight=query.preflight,
        query_run=query.run,
        expected_retrieval_sha256=source.retrieval_sha256,
        expected_source_population_id=source.population_id,
        expected_query_preflight_sha256=query.preflight.sha256,
        expected_query_run_sha256=query.run.sha256,
        expected_query_population_id=expected_query_population_id,
        expected_query_prompt_population_sha256=(
            expected_query_prompt_population_sha256
        ),
    )
    answer = build_query_payload_answer_plan(
        facts,
        protected_parent.exact_parent,
        max_prompt_tokens=max_prompt_tokens,
        output_token_reserve=output_token_reserve,
    )
    return ConfirmationQueryPayloadPlan(
        protected_parent=protected_parent,
        query_expansion=query,
        query_fact_population=facts,
        answer_plan=answer,
    )


def _plain_messages(row: query_payload_live.QueryPayloadPlanRow) -> list[dict[str, str]]:
    _require(row.messages is not None, "fallback row has no Terra provider input")
    assert row.messages is not None
    messages = [
        {"role": message.role, "content": message.content}
        for message in row.messages
    ]
    _require(
        identity_sha256(messages) == row.messages_sha256,
        f"query-payload provider messages changed at ordinal {row.adapter.source.ordinal}",
    )
    return messages


def compile_confirmation_query_payload_prompt_plane(
    plan: ConfirmationQueryPayloadPlan,
    *,
    answer_preflight_sha256: str,
    max_concurrency: int = 4,
    gateway_url: str = live.DEFAULT_GATEWAY_URL,
) -> dict[str, Any]:
    """Project exact submitted messages into a sealed confirmation prompt plane."""

    if type(plan) is not ConfirmationQueryPayloadPlan:
        raise TypeError("plan must be an exact ConfirmationQueryPayloadPlan")
    preflight_sha = require_sha256(
        answer_preflight_sha256, "query-payload answer preflight SHA-256"
    )
    _require(
        type(max_concurrency) is int and max_concurrency > 0,
        "query-payload max concurrency must be positive",
    )
    rows: list[dict[str, Any]] = []
    for prompt_index, row in enumerate(plan.answer_plan.submitted_rows):
        messages = _plain_messages(row)
        provider_body = {
            "format": PROVIDER_INPUT_FORMAT,
            "messages": messages,
            "messages_sha256": row.messages_sha256,
        }
        provider_input = {
            **provider_body,
            "provider_input_receipt_sha256": identity_sha256(provider_body),
        }
        body: dict[str, Any] = {
            "format": PROMPT_ROW_FORMAT,
            "prompt_index": prompt_index,
            "source_row_index": row.adapter.source.ordinal,
            "question_id": row.adapter.source.packet.question_id,
            "question_sha256": row.adapter.source.packet.question_sha256,
            "query_payload_receipt_sha256": row.receipt_sha256,
            "provider_input": provider_input,
        }
        rows.append({**body, "row_receipt_sha256": identity_sha256(body)})
    source_ids = [
        row.adapter.source.packet.question_id for row in plan.answer_plan.rows
    ]
    body = {
        "answer_preflight_sha256": preflight_sha,
        "answer_plan_identity_sha256": plan.answer_plan.plan_identity_sha256,
        "format": PROMPT_FORMAT,
        "gold_loaded": False,
        "ordered_rows": rows,
        "physical_provider_calls": 0,
        "population": {
            "ordered_source_question_ids_sha256": identity_sha256(source_ids),
            "query_fact_population_id": plan.query_fact_population.population_id,
            "query_population_id": plan.query_fact_population.query_population_id,
            "question_count": len(plan.answer_plan.rows),
            "required_provider_calls": plan.required_calls,
            "submitted_question_ids_sha256": identity_sha256(
                [row["question_id"] for row in rows]
            ),
        },
        "provider_execution_available": False,
        "renderer_id": query_payload_live.RENDERER_ID,
        "runtime": {
            "gateway_url": gateway_url,
            "input_token_cap": plan.answer_plan.max_prompt_tokens,
            "max_concurrency": max_concurrency,
            "model": live.DEFAULT_TERRA_GATEWAY_MODEL,
            "output_token_reserve": plan.answer_plan.output_token_reserve,
            "retries": 0,
        },
        "source_bindings": {
            "parent_bridge_sha256": (
                plan.protected_parent.bridge_artifact.sha256
            ),
            "parent_run_sha256": plan.protected_parent.run_artifact.sha256,
            "query_preflight_sha256": plan.query_expansion.preflight.sha256,
            "query_run_sha256": plan.query_expansion.run.sha256,
            "query_runtime_ledger_sha256": (
                plan.query_expansion.runtime_ledger.sha256
            ),
        },
        "status": "preflighted",
    }
    assert_gold_blind(body, path="confirmation_query_payload_prompt_plane")
    return {**body, "prompt_plane_identity_sha256": identity_sha256(body)}


def publish_confirmation_query_payload_preflight(
    plan: ConfirmationQueryPayloadPlan,
    *,
    output_root: str | Path,
    max_concurrency: int = 4,
    gateway_url: str = live.DEFAULT_GATEWAY_URL,
) -> ConfirmationQueryPayloadPreflight:
    """Publish both the authoritative preflight and exact prompt payload."""

    output = Path(output_root)
    answer_preflight = preflight_query_payload_answers(
        plan.answer_plan, output_root=output
    )
    prompt_payload = compile_confirmation_query_payload_prompt_plane(
        plan,
        answer_preflight_sha256=answer_preflight.sha256,
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
    )
    prompt, _created = publish_sealed_json(output / PROMPT_NAME, prompt_payload)
    return ConfirmationQueryPayloadPreflight(plan, prompt, answer_preflight)


def _verified_preflight(
    preflight: ConfirmationQueryPayloadPreflight,
    *,
    output_root: str | Path,
    expected_prompt_sha256: str,
    expected_answer_preflight_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact]:
    expected_prompt = require_sha256(
        expected_prompt_sha256, "query-payload prompt plane SHA-256"
    )
    expected_answer = require_sha256(
        expected_answer_preflight_sha256,
        "query-payload authoritative answer preflight SHA-256",
    )
    prompt = read_sealed_json(Path(output_root) / PROMPT_NAME)
    answer = read_sealed_json(
        Path(output_root) / query_payload_live.ANSWER_PREFLIGHT_NAME
    )
    _require(
        prompt.sha256 == preflight.prompt_artifact.sha256 == expected_prompt,
        "query-payload prompt plane SHA-256 changed",
    )
    _require(
        answer.sha256
        == preflight.answer_preflight_artifact.sha256
        == expected_answer,
        "query-payload answer preflight SHA-256 changed",
    )
    expected_payload = compile_confirmation_query_payload_prompt_plane(
        preflight.plan,
        answer_preflight_sha256=answer.sha256,
        max_concurrency=prompt.payload["runtime"]["max_concurrency"],
        gateway_url=prompt.payload["runtime"]["gateway_url"],
    )
    _require(prompt.payload == expected_payload, "query-payload prompt plane changed")
    return prompt, answer


def _checkpoint_records(
    preflight: ConfirmationQueryPayloadPreflight,
    *,
    output_root: str | Path,
    answer_preflight_sha256: str,
) -> tuple[dict[str, str], ...]:
    plan = preflight.plan.answer_plan
    checkpoint = Path(output_root) / query_payload_live.CHECKPOINT_DIR_NAME
    if not checkpoint.exists():
        return ()
    _require(
        checkpoint.is_dir() and not checkpoint.is_symlink(),
        "query-payload checkpoint root is absent or unsafe",
    )
    requests: set[str] = set()
    responses: set[str] = set()
    for path in checkpoint.iterdir():
        _require(
            path.is_file() and not path.is_symlink(),
            "query-payload checkpoint contains unsafe state",
        )
        if path.name == ".fast-completion-journal.lock":
            continue
        match = _JOURNAL_NAME.fullmatch(path.name)
        _require(match is not None, "query-payload checkpoint contains foreign state")
        assert match is not None
        (requests if match.group("kind") == "request" else responses).add(
            match.group("key")
        )
    _require(
        requests == responses,
        "query-payload request/response pair is incomplete; unsafe retry forbidden",
    )
    if not requests:
        return ()
    # This is the sole private-runtime seam: authenticate already-complete
    # journal pairs against the exact authoritative plan. Provider execution,
    # materialization, and replay remain owned by query_payload_live.
    runtime = query_payload_live._runtime(  # noqa: SLF001
        plan,
        checkpoint_dir=checkpoint,
        client=None,
        max_concurrency=preflight.prompt_artifact.payload["runtime"][
            "max_concurrency"
        ],
        gateway_url=preflight.prompt_artifact.payload["runtime"]["gateway_url"],
        preflight_sha256=answer_preflight_sha256,
    )
    try:
        with runtime._journal_guard():  # noqa: SLF001
            records = runtime._load_all_records()  # noqa: SLF001
    finally:
        runtime.close()
    ordered: list[dict[str, str]] = []
    assert plan.prompt_population is not None
    seen: set[str] = set()
    for row in plan.prompt_population.ordered_rows:
        record = records.get(row.messages_sha256)
        if record is None or row.messages_sha256 in seen:
            continue
        ordered.append(
            {
                "messages_sha256": record.messages_sha256,
                "call_key_sha256": record.call_key_sha256,
                "request_journal_sha256": record.request_journal_sha256,
                "response_journal_sha256": record.response_journal_sha256,
            }
        )
        seen.add(row.messages_sha256)
    _require(
        len(ordered) == len(requests),
        "query-payload authenticated checkpoint population changed",
    )
    return tuple(ordered)


def approve_confirmation_query_payload_release(
    preflight: ConfirmationQueryPayloadPreflight,
    *,
    output_root: str | Path,
    expected_prompt_sha256: str,
    expected_answer_preflight_sha256: str,
    approve_provider_release: bool,
    authorized_provider_calls: int,
) -> SealedArtifact:
    """Seal exact remaining-call approval after authenticating checkpoints."""

    _require(approve_provider_release is True, "provider release requires approval")
    prompt, answer = _verified_preflight(
        preflight,
        output_root=output_root,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_answer_preflight_sha256=expected_answer_preflight_sha256,
    )
    records = _checkpoint_records(
        preflight,
        output_root=output_root,
        answer_preflight_sha256=answer.sha256,
    )
    remaining = preflight.plan.required_calls - len(records)
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == remaining,
        "query-payload release authorization must equal exact remaining calls",
    )
    root = Path(output_root).resolve().as_posix()
    body = {
        "answer_preflight_sha256": answer.sha256,
        "approval_opt_in": True,
        "checkpoint_snapshot": {
            "authenticated_complete_count": len(records),
            "ordered_records": list(records),
            "ordered_records_sha256": identity_sha256(list(records)),
        },
        "format": RELEASE_FORMAT,
        "gold_loaded": False,
        "output_root": root,
        "output_root_sha256": identity_sha256({"canonical_root": root}),
        "physical_provider_calls": 0,
        "prompt_plane_sha256": prompt.sha256,
        "release_status": "approved_for_provider_execution",
        "required_authorized_provider_calls": remaining,
        "unsafe_retry_policy": "refuse-incomplete-request-response-pair-v1",
    }
    assert_gold_blind(body, path="confirmation_query_payload_release")
    payload = {**body, "release_identity_sha256": identity_sha256(body)}
    release, _created = publish_sealed_json(Path(output_root) / RELEASE_NAME, payload)
    return release


def _verified_release(
    preflight: ConfirmationQueryPayloadPreflight,
    *,
    output_root: str | Path,
    expected_prompt_sha256: str,
    expected_answer_preflight_sha256: str,
    expected_release_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, SealedArtifact, tuple[dict[str, str], ...]]:
    prompt, answer = _verified_preflight(
        preflight,
        output_root=output_root,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_answer_preflight_sha256=expected_answer_preflight_sha256,
    )
    release = _read_expected_artifact(
        Path(output_root) / RELEASE_NAME,
        expected_sha256=expected_release_sha256,
        label="query-payload provider release",
    )
    _require(
        set(release.payload) == _RELEASE_KEYS,
        "query-payload provider release schema changed",
    )
    body = dict(release.payload)
    declared = body.pop("release_identity_sha256", None)
    _require(
        declared == identity_sha256(body),
        "query-payload provider release self-seal changed",
    )
    snapshot = release.payload.get("checkpoint_snapshot")
    _require(
        type(snapshot) is dict and set(snapshot) == _CHECKPOINT_SNAPSHOT_KEYS,
        "query-payload release checkpoint schema changed",
    )
    rows = snapshot.get("ordered_records")
    _require(
        type(rows) is list
        and all(
            type(row) is dict
            and set(row) == _CHECKPOINT_RECORD_KEYS
            for row in rows
        ),
        "query-payload release checkpoint rows changed",
    )
    released = tuple(dict(row) for row in rows)
    for index, row in enumerate(released):
        for key, value in row.items():
            require_sha256(value, f"query-payload release record {index} {key}")
    _require(
        len({row["messages_sha256"] for row in released}) == len(released),
        "query-payload release checkpoint records repeat",
    )
    root = Path(output_root).resolve().as_posix()
    _require(
        release.payload.get("format") == RELEASE_FORMAT
        and release.payload.get("release_status")
        == "approved_for_provider_execution"
        and release.payload.get("approval_opt_in") is True
        and release.payload.get("gold_loaded") is False
        and release.payload.get("prompt_plane_sha256") == prompt.sha256
        and release.payload.get("answer_preflight_sha256") == answer.sha256
        and release.payload.get("output_root") == root
        and release.payload.get("output_root_sha256")
        == identity_sha256({"canonical_root": root})
        and release.payload.get("physical_provider_calls") == 0
        and release.payload.get("unsafe_retry_policy")
        == "refuse-incomplete-request-response-pair-v1"
        and snapshot.get("authenticated_complete_count") == len(released)
        and snapshot.get("ordered_records_sha256")
        == identity_sha256(list(released))
        and release.payload.get("required_authorized_provider_calls")
        == preflight.plan.required_calls - len(released),
        "query-payload provider release bindings changed",
    )
    assert_gold_blind(release.payload, path="confirmation_query_payload_release")
    return prompt, answer, release, released


def _default_client_factory(gateway_url: str, api_key_env: str) -> Any:
    import os

    api_key = os.environ.get(api_key_env, "").strip()
    _require(bool(api_key), f"provider API key is empty: {api_key_env}")
    return live._make_provider_client(api_key, gateway_url)  # noqa: SLF001


def run_confirmation_query_payload_provider(
    preflight: ConfirmationQueryPayloadPreflight,
    *,
    output_root: str | Path,
    expected_prompt_sha256: str,
    expected_answer_preflight_sha256: str,
    expected_release_sha256: str,
    enable_provider: bool,
    authorized_provider_calls: int,
    api_key_env: str = live.DEFAULT_API_KEY_ENV,
    client_factory: ClientFactory = _default_client_factory,
) -> QueryPayloadProviderResult:
    """Fill authoritative journals after every release gate verifies."""

    prompt, answer, release, released = _verified_release(
        preflight,
        output_root=output_root,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_answer_preflight_sha256=expected_answer_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    current = _checkpoint_records(
        preflight,
        output_root=output_root,
        answer_preflight_sha256=answer.sha256,
    )
    current_by_message = {row["messages_sha256"]: row for row in current}
    _require(
        all(current_by_message.get(row["messages_sha256"]) == row for row in released),
        "query-payload checkpoint changed after release",
    )
    remaining = preflight.plan.required_calls - len(current)
    _require(
        enable_provider == bool(preflight.plan.required_calls),
        "query-payload provider opt-in must match the prompt population",
    )
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == remaining,
        "query-payload provider authorization must equal exact remaining calls",
    )
    _require(
        remaining <= release.payload["required_authorized_provider_calls"],
        "query-payload current state exceeds its sealed release budget",
    )
    client = (
        client_factory(prompt.payload["runtime"]["gateway_url"], api_key_env)
        if remaining
        else None
    )
    result = run_query_payload_answer_provider(
        preflight.plan.answer_plan,
        output_root=output_root,
        expected_preflight_sha256=answer.sha256,
        enable_provider=bool(preflight.plan.required_calls),
        authorized_provider_calls=preflight.plan.required_calls,
        client=client,
        max_concurrency=prompt.payload["runtime"]["max_concurrency"],
        gateway_url=prompt.payload["runtime"]["gateway_url"],
    )
    _require(
        result.physical_provider_calls == remaining
        and result.checkpoint_hits == len(current),
        "query-payload provider accounting differs from exact authorization",
    )
    return result


def materialize_confirmation_query_payload_answers(
    preflight: ConfirmationQueryPayloadPreflight,
    *,
    output_root: str | Path,
    expected_prompt_sha256: str,
    expected_answer_preflight_sha256: str,
    expected_release_sha256: str,
) -> QueryPayloadAnswerRunResult:
    """Materialize answers only from a complete client-free journal replay."""

    prompt, answer, _release, _records = _verified_release(
        preflight,
        output_root=output_root,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_answer_preflight_sha256=expected_answer_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    journals = load_query_payload_answer_provider_journals(
        preflight.plan.answer_plan,
        output_root=output_root,
        expected_preflight_sha256=answer.sha256,
        max_concurrency=prompt.payload["runtime"]["max_concurrency"],
        gateway_url=prompt.payload["runtime"]["gateway_url"],
    )
    return materialize_query_payload_answers(
        preflight.plan.answer_plan,
        output_root=output_root,
        expected_preflight_sha256=answer.sha256,
        completion_batch=journals.batch,
        gateway_url=prompt.payload["runtime"]["gateway_url"],
    )


def replay_confirmation_query_payload_answers(
    preflight: ConfirmationQueryPayloadPreflight,
    *,
    output_root: str | Path,
    expected_prompt_sha256: str,
    expected_answer_preflight_sha256: str,
    expected_release_sha256: str,
    expected_run_sha256: str,
) -> VerifiedQueryPayloadAnswerPlane:
    """Return the exact evidence-map parent after client-free replay."""

    prompt, answer, _release, _records = _verified_release(
        preflight,
        output_root=output_root,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_answer_preflight_sha256=expected_answer_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    verified = replay_query_payload_answers(
        preflight.plan.answer_plan,
        output_root=output_root,
        expected_preflight_sha256=answer.sha256,
        expected_run_sha256=expected_run_sha256,
        max_concurrency=prompt.payload["runtime"]["max_concurrency"],
        gateway_url=prompt.payload["runtime"]["gateway_url"],
    )
    _require(
        type(verified) is VerifiedQueryPayloadAnswerPlane,
        "query-payload replay did not produce the exact evidence-map parent",
    )
    return verified


__all__ = [
    "ConfirmationQueryPayloadPlan",
    "ConfirmationQueryPayloadPreflight",
    "ConfirmationQueryPayloadError",
    "PARENT_BRIDGE_FORMAT",
    "PARENT_BRIDGE_NAME",
    "PARENT_DIR_NAME",
    "PARENT_LEDGER_NAME",
    "PARENT_LEDGER_REPLAY_NAME",
    "PARENT_PLAN_ID",
    "PARENT_REPLAY_NAME",
    "PARENT_ROW_FORMAT",
    "PARENT_RUN_FORMAT",
    "PARENT_RUN_NAME",
    "PROMPT_FORMAT",
    "PROMPT_NAME",
    "PROMPT_ROW_FORMAT",
    "PROVIDER_INPUT_FORMAT",
    "RELEASE_FORMAT",
    "RELEASE_NAME",
    "VerifiedQueryExpansionArtifacts",
    "VerifiedProtectedS0Parent",
    "approve_confirmation_query_payload_release",
    "build_confirmation_query_payload_plan",
    "compile_confirmation_query_payload_prompt_plane",
    "load_verified_query_expansion_artifacts",
    "materialize_confirmation_query_payload_answers",
    "materialize_verified_protected_s0_parent",
    "publish_confirmation_query_payload_preflight",
    "replay_confirmation_query_payload_answers",
    "run_confirmation_query_payload_provider",
]
