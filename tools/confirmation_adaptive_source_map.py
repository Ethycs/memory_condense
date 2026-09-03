#!/usr/bin/env python3
"""Gold-blind confirmation adapter for the adaptive source-history base round.

The historical adaptive mapper already owns source selection, namespace-batched
read-only hydration, compact history windowing, prompt rendering, exact-quote
validation, and lane-alias fanout.  This module supplies the confirmation
lifecycle around those mechanisms without importing validation pins:

``source plane -> sealed preflight -> exact release -> native journals ->
store-free materialization -> store-revalidating replay``.

Only the provider-run function can construct a live client.  Materialization
loads the prompt-external work manifest and completed FastCompletion journals;
it never opens a memory store.  Replay rebuilds the base plan from the exact
source population, which deliberately revalidates and rereads every referenced
namespace before accepting byte-identical results.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from memory_condense.eval.fast_completion_runtime import (
    FastCompletionBatch,
    FastCompletionRuntime,
    preflight_fast_completion_prompts,
)
from tools.confirmation_source_streams import ConfirmationSourceStreamsResult
from tools.matched_eval import provider_runtime
from tools.matched_eval.artifacts import (
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    assert_gold_blind,
    identity_sha256,
    require_sha256,
)
from tools.matched_eval.locked_source_gate_adapter import (
    DIRECT_STREAM_PROFILE_V1,
    LockedSourceGateAdapterPopulation,
)
from tools.matched_eval.query_map_source_gate_adapter import (
    CONSOLIDATED_OBLIGATION_MODE,
    QueryMapSourceGateAdapterPlane,
    STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
)
from tools.matched_eval.source_history_mapper_live import (
    HARD_CONTEXT_TOKEN_CAP,
    MAPPER_CONTRACT_SHA256,
    MAX_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    SourceMapperMaterialization,
    WorkDisposition,
)
from tools.run_locked_adaptive_source_map import (
    FastMaterializationQuestionPlan,
    LockedAdaptiveBasePlan,
    build_locked_base_round,
    load_fast_materialization_manifest,
    materialize_fast_question_plans,
    provider_journals_for_question,
    source_gate_policy,
    work_manifest_projection,
)


FORMAT = "memory-condense-confirmation-adaptive-source-map-v1"
PREFLIGHT_FORMAT = f"{FORMAT}-preflight"
RELEASE_FORMAT = f"{FORMAT}-provider-release"
MATERIALIZATION_FORMAT = f"{FORMAT}-materialization"
REPLAY_FORMAT = f"{FORMAT}-replay"

PREFLIGHT_NAME = "confirmation-adaptive-source-map-preflight-v1.json"
WORK_MANIFEST_NAME = "confirmation-adaptive-source-map-work-manifest-v1.json"
RELEASE_NAME = "confirmation-adaptive-source-map-provider-release-v1.json"
MATERIALIZATION_NAME = "confirmation-adaptive-source-map-materialization-v1.json"
REPLAY_NAME = "confirmation-adaptive-source-map-replay-v1.json"
CHECKPOINT_DIR_NAME = "terra-source-history-map-calls"

FROZEN_DIRECT_BASE_CAP = 1
FROZEN_PARTITION_BASE_CAP = 0
FROZEN_GUIDED_BASE_CAP = 1

_JOURNAL_NAME = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)
_CHECKPOINT_RECORD_KEYS = frozenset(
    {
        "call_key_sha256",
        "messages_sha256",
        "physical_work_id",
        "prompt_id",
        "request_journal_sha256",
        "response_journal_sha256",
    }
)
_CHECKPOINT_SNAPSHOT_KEYS = frozenset(
    {"authenticated_complete_count", "ordered_records", "ordered_records_sha256"}
)


class ConfirmationAdaptiveSourceMapError(MatchedEvalContractError):
    """A confirmation source plane, release, journal, or replay changed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationAdaptiveSourceMapError(message)


ClientFactory = Callable[[str, str], Any]


@dataclass(frozen=True, slots=True)
class ConfirmationAdaptiveSourceMapPreflight:
    """Exact in-process plan plus its two prompt-external seals."""

    plan: LockedAdaptiveBasePlan
    preflight_artifact: SealedArtifact
    work_manifest_artifact: SealedArtifact

    def __post_init__(self) -> None:
        _require(
            type(self.plan) is LockedAdaptiveBasePlan,
            "confirmation adaptive preflight changed plan type",
        )
        _require(
            type(self.preflight_artifact) is SealedArtifact
            and type(self.work_manifest_artifact) is SealedArtifact,
            "confirmation adaptive preflight changed artifact type",
        )
        _require(
            self.preflight_artifact.payload.get("work_manifest_sha256")
            == self.work_manifest_artifact.sha256,
            "confirmation adaptive preflight lost its work-manifest binding",
        )

    @property
    def required_provider_calls(self) -> int:
        return self.plan.required_provider_calls


@dataclass(frozen=True, slots=True)
class ConfirmationAdaptiveSourceMapMaterialization:
    """Store-free typed mapper output for the downstream adaptive tail."""

    preflight_artifact: SealedArtifact
    work_manifest_artifact: SealedArtifact
    release_artifact: SealedArtifact
    materialization_artifact: SealedArtifact
    completion_batch: FastCompletionBatch
    questions: tuple[FastMaterializationQuestionPlan, ...]
    materializations: tuple[SourceMapperMaterialization, ...]

    def __post_init__(self) -> None:
        _require(
            type(self.completion_batch) is FastCompletionBatch,
            "confirmation adaptive materialization changed batch type",
        )
        _require(
            type(self.questions) is tuple
            and bool(self.questions)
            and all(
                type(row) is FastMaterializationQuestionPlan
                for row in self.questions
            ),
            "confirmation adaptive materialization changed question plans",
        )
        _require(
            type(self.materializations) is tuple
            and len(self.materializations) == len(self.questions)
            and all(
                type(row) is SourceMapperMaterialization
                for row in self.materializations
            ),
            "confirmation adaptive materialization changed typed results",
        )


@dataclass(frozen=True, slots=True)
class VerifiedConfirmationAdaptiveSourceMapPlane:
    """Store-revalidated, byte-replayed base source-map plane."""

    preflight_artifact: SealedArtifact
    work_manifest_artifact: SealedArtifact
    release_artifact: SealedArtifact
    materialization_artifact: SealedArtifact
    replay_artifact: SealedArtifact
    completion_batch: FastCompletionBatch
    source_population: LockedSourceGateAdapterPopulation
    query_adapter: QueryMapSourceGateAdapterPlane
    questions: tuple[FastMaterializationQuestionPlan, ...]
    materializations: tuple[SourceMapperMaterialization, ...]

    def __post_init__(self) -> None:
        _require(
            type(self.source_population) is LockedSourceGateAdapterPopulation
            and type(self.query_adapter) is QueryMapSourceGateAdapterPlane,
            "verified confirmation base plane changed exact parents",
        )
        _require(
            len(self.questions) == len(self.materializations)
            and all(
                type(row) is SourceMapperMaterialization
                for row in self.materializations
            ),
            "verified confirmation base plane changed typed results",
        )
        _require(
            self.replay_artifact.payload.get("materialization_sha256")
            == self.materialization_artifact.sha256,
            "verified confirmation base plane changed replay binding",
        )


def _frozen_policy_receipt() -> str:
    return source_gate_policy(
        FROZEN_DIRECT_BASE_CAP,
        FROZEN_PARTITION_BASE_CAP,
        FROZEN_GUIDED_BASE_CAP,
    ).receipt_sha256


def _validate_source_parents(
    source_population: LockedSourceGateAdapterPopulation,
    query_adapter: QueryMapSourceGateAdapterPlane,
) -> None:
    if type(source_population) is not LockedSourceGateAdapterPopulation:
        raise TypeError(
            "source_population must be an exact LockedSourceGateAdapterPopulation"
        )
    if type(query_adapter) is not QueryMapSourceGateAdapterPlane:
        raise TypeError("query_adapter must be an exact QueryMapSourceGateAdapterPlane")
    _require(
        source_population.direct_stream_profile == DIRECT_STREAM_PROFILE_V1,
        "confirmation base round requires the frozen direct-stream profile",
    )
    _require(
        query_adapter.obligation_compilation_mode == CONSOLIDATED_OBLIGATION_MODE
        and query_adapter.state_chain_profile
        == STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
        "confirmation base round requires consolidated state-chain authority",
    )
    activated = query_adapter.activated_rows
    questions = source_population.questions
    _require(
        len(activated) == len(questions)
        and tuple((row.ordinal, row.question_id) for row in activated)
        == tuple((row.ordinal, row.plan.question_id) for row in questions),
        "confirmation source population differs from query-map activations",
    )
    policy_receipt = _frozen_policy_receipt()
    for adapted, question in zip(activated, questions, strict=True):
        _require(
            adapted.activation is not None
            and question.plan.activation.receipt_sha256
            == adapted.activation.receipt_sha256
            and question.plan.parent.parent_packet_id == adapted.map_packet_id
            and question.source_packet_id == adapted.source_packet_id
            and question.plan.parent.snapshot_id == query_adapter.snapshot_id
            and question.plan.policy.receipt_sha256 == policy_receipt,
            f"confirmation source/query binding changed at ordinal {adapted.ordinal}",
        )


def _plain_messages(messages: Sequence[Any]) -> tuple[dict[str, str], ...]:
    return tuple({"role": row.role, "content": row.content} for row in messages)


def _provider_prompts(
    plan: LockedAdaptiveBasePlan,
) -> tuple[tuple[dict[str, str], ...], ...]:
    prompts = tuple(
        _plain_messages(row.messages)
        for row in plan.all_prompt_rows
        if row.disposition is WorkDisposition.NEW_CALL
    )
    _require(
        len(prompts)
        == plan.provider_population.logical_prompt_count
        == plan.provider_population.unique_prompt_count
        == plan.required_provider_calls,
        "confirmation mapper provider population changed",
    )
    return prompts


def _preflight_payload(
    plan: LockedAdaptiveBasePlan,
    *,
    work_manifest_sha256: str,
    model: str,
    gateway_url: str,
    max_concurrency: int,
) -> dict[str, Any]:
    adapter = plan.query_adapter
    _require(
        type(adapter) is QueryMapSourceGateAdapterPlane,
        "confirmation adaptive plan lacks its query-map adapter",
    )
    require_sha256(work_manifest_sha256, "confirmation adaptive work manifest")
    _require(type(model) is str and bool(model), "source-map model is empty")
    _require(
        type(gateway_url) is str and bool(gateway_url),
        "source-map gateway URL is empty",
    )
    _require(
        type(max_concurrency) is int and max_concurrency > 0,
        "source-map max concurrency must be positive",
    )
    source_by_id = {
        row.plan.question_id: row for row in plan.source_population.questions
    }
    questions: list[dict[str, Any]] = []
    for row in plan.questions:
        source = source_by_id[row.question_id]
        value = row.projection()
        value.update(
            {
                "activation_receipt_sha256": source.plan.activation.receipt_sha256,
                "parent_identity_sha256": source.plan.parent.identity_sha256,
                "route": source.plan.route.style.value,
                "source_candidate_count": len(source.plan.candidates),
            }
        )
        questions.append(value)
    policy = source_gate_policy(
        FROZEN_DIRECT_BASE_CAP,
        FROZEN_PARTITION_BASE_CAP,
        FROZEN_GUIDED_BASE_CAP,
    )
    payload: dict[str, Any] = {
        "activated_question_count": len(adapter.activated_rows),
        "format": PREFLIGHT_FORMAT,
        "gold_loaded": False,
        "hard_context_token_cap": HARD_CONTEXT_TOKEN_CAP,
        "mapped_activated_question_count": len(plan.questions),
        "mapper_contract_sha256": MAPPER_CONTRACT_SHA256,
        "namespace_hydration_batches": [
            row.projection() for row in plan.hydration_batches
        ],
        "no_op_question_count": len(adapter.no_op_rows),
        "ordered_question_ids_sha256": identity_sha256(
            [row.question_id for row in adapter.rows]
        ),
        "output_token_reserve": OUTPUT_TOKEN_RESERVE,
        "physical_prompt_rows": [
            row.projection(include_messages=True) for row in plan.all_prompt_rows
        ],
        "provider_calls": 0,
        "provider_population": plan.provider_population.model_dump(),
        "query_map_adapter_receipt_sha256": adapter.receipt_sha256,
        "question_count": len(adapter.rows),
        "question_plans": questions,
        "required_authorized_provider_calls": plan.required_provider_calls,
        "retained_transformer_token_state_bytes": 0,
        "route_counts": dict(plan.route_counts),
        "runtime": {
            "gateway_url": gateway_url,
            "max_concurrency": max_concurrency,
            "model": model,
            "retries": 0,
        },
        "source_gate_policy": policy.projection(),
        "source_gate_policy_receipt_sha256": policy.receipt_sha256,
        "source_gate_population_receipt_sha256": (
            plan.source_population.receipt_sha256
        ),
        "source_input_artifacts": [
            row.projection() for row in plan.source_population.source_artifacts
        ],
        "source_mapper_preflight_receipt_sha256s": [
            row.mapper_preflight.receipt_sha256 for row in plan.questions
        ],
        "status": "preflighted",
        "store_reads_during_materialization": 0,
        "work_manifest_name": WORK_MANIFEST_NAME,
        "work_manifest_sha256": work_manifest_sha256,
    }
    assert_gold_blind(payload, path="confirmation_adaptive_source_map_preflight")
    return payload


def publish_confirmation_adaptive_source_map_preflight(
    source_population: LockedSourceGateAdapterPopulation,
    query_adapter: QueryMapSourceGateAdapterPlane,
    *,
    output_root: str | Path,
    model: str = provider_runtime.DEFAULT_TERRA_GATEWAY_MODEL,
    gateway_url: str = provider_runtime.DEFAULT_GATEWAY_URL,
    max_concurrency: int = 4,
) -> ConfirmationAdaptiveSourceMapPreflight:
    """Hydrate once and seal the arbitrary-size confirmation base work."""

    _validate_source_parents(source_population, query_adapter)
    plan = build_locked_base_round(
        source_population,
        query_adapter=query_adapter,
    )
    _require(
        plan.source_population is source_population
        and plan.query_adapter is query_adapter,
        "confirmation base planner changed its exact parents",
    )
    root = Path(output_root)
    work, _created = publish_sealed_json(
        root / WORK_MANIFEST_NAME,
        work_manifest_projection(plan),
    )
    payload = _preflight_payload(
        plan,
        work_manifest_sha256=work.sha256,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
    )
    preflight, _created = publish_sealed_json(root / PREFLIGHT_NAME, payload)
    return ConfirmationAdaptiveSourceMapPreflight(plan, preflight, work)


def publish_confirmation_adaptive_source_map_from_streams(
    source_streams: ConfirmationSourceStreamsResult,
    *,
    output_root: str | Path,
    model: str = provider_runtime.DEFAULT_TERRA_GATEWAY_MODEL,
    gateway_url: str = provider_runtime.DEFAULT_GATEWAY_URL,
    max_concurrency: int = 4,
) -> ConfirmationAdaptiveSourceMapPreflight:
    """Consume the exact provider-free confirmation source-stream result."""

    if type(source_streams) is not ConfirmationSourceStreamsResult:
        raise TypeError(
            "source_streams must be an exact ConfirmationSourceStreamsResult"
        )
    _require(
        source_streams.plane_artifact.payload.get(
            "base_source_population_receipt_sha256"
        )
        == source_streams.base_population.receipt_sha256
        and source_streams.plane_artifact.payload.get(
            "query_map_adapter_receipt_sha256"
        )
        == source_streams.query_map_adapter.receipt_sha256,
        "confirmation source-stream plane changed its base parents",
    )
    return publish_confirmation_adaptive_source_map_preflight(
        source_streams.base_population,
        source_streams.query_map_adapter,
        output_root=output_root,
        model=model,
        gateway_url=gateway_url,
        max_concurrency=max_concurrency,
    )


def _read_expected(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> SealedArtifact:
    expected = require_sha256(expected_sha256, f"expected {label} SHA-256")
    artifact = read_sealed_json(path)
    _require(artifact.sha256 == expected, f"{label} SHA-256 changed")
    return artifact


def _verified_preflight(
    preflight: ConfirmationAdaptiveSourceMapPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_work_manifest_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact]:
    if type(preflight) is not ConfirmationAdaptiveSourceMapPreflight:
        raise TypeError(
            "preflight must be an exact ConfirmationAdaptiveSourceMapPreflight"
        )
    root = Path(output_root)
    artifact = _read_expected(
        root / PREFLIGHT_NAME,
        expected_sha256=expected_preflight_sha256,
        label="confirmation adaptive source-map preflight",
    )
    work = _read_expected(
        root / WORK_MANIFEST_NAME,
        expected_sha256=expected_work_manifest_sha256,
        label="confirmation adaptive source-map work manifest",
    )
    _require(
        artifact.sha256 == preflight.preflight_artifact.sha256
        and work.sha256 == preflight.work_manifest_artifact.sha256
        and artifact.payload.get("work_manifest_sha256") == work.sha256,
        "confirmation adaptive preflight object differs from disk seals",
    )
    runtime = artifact.payload.get("runtime")
    _require(type(runtime) is dict, "confirmation adaptive runtime seal changed")
    expected_payload = _preflight_payload(
        preflight.plan,
        work_manifest_sha256=work.sha256,
        model=runtime.get("model"),
        gateway_url=runtime.get("gateway_url"),
        max_concurrency=runtime.get("max_concurrency"),
    )
    _require(
        artifact.payload == expected_payload
        and work.payload == work_manifest_projection(preflight.plan),
        "confirmation adaptive preflight or work manifest changed",
    )
    return artifact, work


def _runtime(
    preflight: ConfirmationAdaptiveSourceMapPreflight,
    artifact: SealedArtifact,
    *,
    checkpoint_dir: Path,
    client: Any | None,
) -> FastCompletionRuntime:
    runtime = artifact.payload.get("runtime")
    _require(type(runtime) is dict, "confirmation adaptive runtime is absent")
    return FastCompletionRuntime(
        checkpoint_dir=checkpoint_dir,
        prompt_population=_provider_prompts(preflight.plan),
        model=runtime["model"],
        client=client,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_new_tokens=OUTPUT_TOKEN_RESERVE,
        max_concurrency=runtime["max_concurrency"],
        retries=0,
        benchmark_provenance={
            "arm": "confirmation_adaptive_source_map_base_v1",
            "gateway_url": runtime["gateway_url"],
            "gold_loaded": False,
            "mapper_contract_sha256": MAPPER_CONTRACT_SHA256,
            "preflight_artifact_sha256": artifact.sha256,
            "query_map_adapter_receipt_sha256": artifact.payload[
                "query_map_adapter_receipt_sha256"
            ],
            "source_gate_population_receipt_sha256": artifact.payload[
                "source_gate_population_receipt_sha256"
            ],
        },
    )


def _scan_checkpoint_root(checkpoint: Path) -> tuple[set[str], set[str]]:
    if not checkpoint.exists():
        return set(), set()
    _require(
        checkpoint.is_dir() and not checkpoint.is_symlink(),
        "confirmation adaptive checkpoint root is absent or unsafe",
    )
    requests: set[str] = set()
    responses: set[str] = set()
    for path in checkpoint.iterdir():
        _require(
            path.is_file() and not path.is_symlink(),
            "confirmation adaptive checkpoint contains unsafe state",
        )
        if path.name == ".fast-completion-journal.lock":
            continue
        match = _JOURNAL_NAME.fullmatch(path.name)
        _require(
            match is not None,
            "confirmation adaptive checkpoint contains foreign state",
        )
        assert match is not None
        (requests if match.group("kind") == "request" else responses).add(
            match.group("key")
        )
    _require(
        requests == responses,
        "confirmation adaptive request/response pair is incomplete; unsafe retry forbidden",
    )
    return requests, responses


def _checkpoint_records(
    preflight: ConfirmationAdaptiveSourceMapPreflight,
    artifact: SealedArtifact,
    *,
    output_root: str | Path,
) -> tuple[dict[str, str], ...]:
    checkpoint = Path(output_root) / CHECKPOINT_DIR_NAME
    requests, _responses = _scan_checkpoint_root(checkpoint)
    if not requests:
        return ()
    runtime = _runtime(
        preflight,
        artifact,
        checkpoint_dir=checkpoint,
        client=None,
    )
    try:
        with runtime._journal_guard():  # noqa: SLF001 - native auth seam
            records = runtime._load_all_records()  # noqa: SLF001
    finally:
        runtime.close()
    prompt_by_messages = {
        row.messages_sha256: row
        for row in preflight.plan.submitted_prompt_rows
    }
    ordered: list[dict[str, str]] = []
    seen: set[str] = set()
    for prompt_row in preflight.plan.provider_population.ordered_rows:
        if prompt_row.messages_sha256 in seen:
            continue
        record = records.get(prompt_row.messages_sha256)
        if record is None:
            continue
        prompt = prompt_by_messages[prompt_row.messages_sha256]
        ordered.append(
            {
                "call_key_sha256": record.call_key_sha256,
                "messages_sha256": record.messages_sha256,
                "physical_work_id": prompt.physical_work_id,
                "prompt_id": prompt.prompt_id,
                "request_journal_sha256": record.request_journal_sha256,
                "response_journal_sha256": record.response_journal_sha256,
            }
        )
        seen.add(prompt_row.messages_sha256)
    _require(
        len(ordered) == len(requests),
        "confirmation adaptive checkpoint population changed",
    )
    return tuple(ordered)


def approve_confirmation_adaptive_source_map_release(
    preflight: ConfirmationAdaptiveSourceMapPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_work_manifest_sha256: str,
    approve_provider_release: bool,
    authorized_provider_calls: int,
) -> SealedArtifact:
    """Seal approval for exactly the currently missing native journal pairs."""

    _require(approve_provider_release is True, "source-map release requires approval")
    artifact, work = _verified_preflight(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_work_manifest_sha256=expected_work_manifest_sha256,
    )
    records = _checkpoint_records(
        preflight,
        artifact,
        output_root=output_root,
    )
    remaining = preflight.required_provider_calls - len(records)
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == remaining,
        "source-map release authorization must equal exact remaining calls",
    )
    canonical_root = Path(output_root).resolve().as_posix()
    body: dict[str, Any] = {
        "approval_opt_in": True,
        "checkpoint_snapshot": {
            "authenticated_complete_count": len(records),
            "ordered_records": list(records),
            "ordered_records_sha256": identity_sha256(list(records)),
        },
        "format": RELEASE_FORMAT,
        "gold_loaded": False,
        "output_root": canonical_root,
        "output_root_sha256": identity_sha256(
            {"canonical_root": canonical_root}
        ),
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "release_status": "approved_for_provider_execution",
        "required_authorized_provider_calls": remaining,
        "unsafe_retry_policy": "refuse-incomplete-request-response-pair-v1",
        "work_manifest_sha256": work.sha256,
    }
    assert_gold_blind(body, path="confirmation_adaptive_source_map_release")
    payload = {**body, "release_identity_sha256": identity_sha256(body)}
    release, _created = publish_sealed_json(
        Path(output_root) / RELEASE_NAME,
        payload,
    )
    return release


def _verified_release(
    preflight: ConfirmationAdaptiveSourceMapPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_work_manifest_sha256: str,
    expected_release_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, SealedArtifact, tuple[dict[str, str], ...]]:
    artifact, work = _verified_preflight(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_work_manifest_sha256=expected_work_manifest_sha256,
    )
    release = _read_expected(
        Path(output_root) / RELEASE_NAME,
        expected_sha256=expected_release_sha256,
        label="confirmation adaptive source-map provider release",
    )
    body = dict(release.payload)
    declared = body.pop("release_identity_sha256", None)
    _require(
        declared == identity_sha256(body),
        "confirmation adaptive provider release self-seal changed",
    )
    snapshot = release.payload.get("checkpoint_snapshot")
    _require(
        type(snapshot) is dict and set(snapshot) == _CHECKPOINT_SNAPSHOT_KEYS,
        "confirmation adaptive release checkpoint schema changed",
    )
    rows = snapshot.get("ordered_records")
    _require(
        type(rows) is list
        and all(type(row) is dict and set(row) == _CHECKPOINT_RECORD_KEYS for row in rows),
        "confirmation adaptive release checkpoint rows changed",
    )
    released = tuple(dict(row) for row in rows)
    for index, row in enumerate(released):
        for key, value in row.items():
            require_sha256(value, f"confirmation adaptive checkpoint {index} {key}")
    _require(
        len({row["messages_sha256"] for row in released}) == len(released),
        "confirmation adaptive release checkpoint rows repeat",
    )
    root = Path(output_root).resolve().as_posix()
    _require(
        release.payload.get("format") == RELEASE_FORMAT
        and release.payload.get("release_status")
        == "approved_for_provider_execution"
        and release.payload.get("approval_opt_in") is True
        and release.payload.get("gold_loaded") is False
        and release.payload.get("preflight_sha256") == artifact.sha256
        and release.payload.get("work_manifest_sha256") == work.sha256
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
        == preflight.required_provider_calls - len(released),
        "confirmation adaptive provider release bindings changed",
    )
    assert_gold_blind(release.payload, path="confirmation_adaptive_source_map_release")
    return artifact, work, release, released


def _default_client_factory(gateway_url: str, api_key_env: str) -> Any:
    import os

    api_key = os.environ.get(api_key_env, "").strip()
    _require(bool(api_key), f"provider API key is empty: {api_key_env}")
    return provider_runtime.make_provider_client(api_key, gateway_url)


def run_confirmation_adaptive_source_map_provider(
    preflight: ConfirmationAdaptiveSourceMapPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_work_manifest_sha256: str,
    expected_release_sha256: str,
    enable_provider: bool,
    authorized_provider_calls: int,
    api_key_env: str = provider_runtime.DEFAULT_API_KEY_ENV,
    client_factory: ClientFactory = _default_client_factory,
) -> FastCompletionBatch:
    """Fill only missing source-history journals under an exact release."""

    artifact, _work, release, released = _verified_release(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_work_manifest_sha256=expected_work_manifest_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    current = _checkpoint_records(
        preflight,
        artifact,
        output_root=output_root,
    )
    current_by_messages = {row["messages_sha256"]: row for row in current}
    _require(
        all(
            current_by_messages.get(row["messages_sha256"]) == row
            for row in released
        ),
        "confirmation adaptive checkpoint changed after release",
    )
    remaining = preflight.required_provider_calls - len(current)
    _require(enable_provider is True, "source-map provider execution is not enabled")
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == remaining,
        "source-map provider authorization must equal exact remaining calls",
    )
    _require(
        remaining <= release.payload["required_authorized_provider_calls"],
        "source-map checkpoint state exceeds its sealed release budget",
    )
    runtime_config = artifact.payload["runtime"]
    client = (
        client_factory(runtime_config["gateway_url"], api_key_env)
        if remaining
        else None
    )
    runtime = _runtime(
        preflight,
        artifact,
        checkpoint_dir=Path(output_root) / CHECKPOINT_DIR_NAME,
        client=client,
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    _require(
        batch.usage.physical_calls == remaining
        and batch.usage.checkpoint_hits == len(current)
        and batch.usage.unique_calls == preflight.required_provider_calls,
        "source-map provider accounting differs from exact authorization",
    )
    return batch


def _completed_batch(
    preflight: ConfirmationAdaptiveSourceMapPreflight,
    artifact: SealedArtifact,
    *,
    output_root: str | Path,
) -> FastCompletionBatch:
    runtime = _runtime(
        preflight,
        artifact,
        checkpoint_dir=Path(output_root) / CHECKPOINT_DIR_NAME,
        client=None,
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    _require(
        batch.usage.physical_calls == 0
        and batch.usage.checkpoint_hits == preflight.required_provider_calls,
        "source-map materialization requires complete checkpoint-only journals",
    )
    return batch


def _materialization_payload(
    *,
    preflight_artifact: SealedArtifact,
    work_manifest_artifact: SealedArtifact,
    release_artifact: SealedArtifact,
    batch: FastCompletionBatch,
    questions: tuple[FastMaterializationQuestionPlan, ...],
    materializations: tuple[SourceMapperMaterialization, ...],
) -> dict[str, Any]:
    _require(
        len(questions) == len(materializations),
        "source-map materialization population changed",
    )
    journals = tuple(
        journal
        for question in questions
        for journal in provider_journals_for_question(
            question.mapper_preflight,
            batch,
        )
    )
    payload: dict[str, Any] = {
        "accepted_before_post_map_dedup_count": sum(
            row.accepted_before_post_map_dedup_count
            for result in materializations
            for row in result.work_results
        ),
        "format": MATERIALIZATION_FORMAT,
        "gold_loaded": False,
        "materializations": [row.projection() for row in materializations],
        "post_map_dedup_performed": False,
        "preflight_sha256": preflight_artifact.sha256,
        "provider_calls_during_materialization": 0,
        "provider_journal_receipt_sha256s": [
            row.receipt_sha256 for row in journals
        ],
        "question_count": len(materializations),
        "rejected_item_count": sum(
            row.rejected_item_count
            for result in materializations
            for row in result.work_results
        ),
        "release_sha256": release_artifact.sha256,
        "retained_transformer_token_state_bytes": 0,
        "runtime_identity_sha256": batch.runtime_identity_sha256,
        "source_gate_population_receipt_sha256": preflight_artifact.payload[
            "source_gate_population_receipt_sha256"
        ],
        "source_mapper_materialization_receipt_sha256s": [
            row.receipt_sha256 for row in materializations
        ],
        "store_reads_during_materialization": 0,
        "work_manifest_sha256": work_manifest_artifact.sha256,
    }
    assert_gold_blind(payload, path="confirmation_adaptive_source_map_materialization")
    return payload


def materialize_confirmation_adaptive_source_map(
    preflight: ConfirmationAdaptiveSourceMapPreflight,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_work_manifest_sha256: str,
    expected_release_sha256: str,
) -> ConfirmationAdaptiveSourceMapMaterialization:
    """Materialize exact mapper objects without opening source stores."""

    artifact, work, release, _released = _verified_release(
        preflight,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_work_manifest_sha256=expected_work_manifest_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    batch = _completed_batch(
        preflight,
        artifact,
        output_root=output_root,
    )
    questions = load_fast_materialization_manifest(
        work,
        expected_source_population_receipt_sha256=require_sha256(
            artifact.payload.get("source_gate_population_receipt_sha256"),
            "confirmation source-gate population",
        ),
    )
    _require(
        [
            row.mapper_preflight.receipt_sha256 for row in questions
        ]
        == artifact.payload.get("source_mapper_preflight_receipt_sha256s")
        and [
            prompt.projection(include_messages=True)
            for row in questions
            for prompt in row.mapper_preflight.prompt_rows
        ]
        == artifact.payload.get("physical_prompt_rows"),
        "source-map work manifest differs from sealed prompts",
    )
    materializations = materialize_fast_question_plans(questions, batch)
    payload = _materialization_payload(
        preflight_artifact=artifact,
        work_manifest_artifact=work,
        release_artifact=release,
        batch=batch,
        questions=questions,
        materializations=materializations,
    )
    terminal, _created = publish_sealed_json(
        Path(output_root) / MATERIALIZATION_NAME,
        payload,
    )
    return ConfirmationAdaptiveSourceMapMaterialization(
        artifact,
        work,
        release,
        terminal,
        batch,
        questions,
        materializations,
    )


def replay_confirmation_adaptive_source_map(
    source_population: LockedSourceGateAdapterPopulation,
    query_adapter: QueryMapSourceGateAdapterPlane,
    *,
    output_root: str | Path,
    expected_preflight_sha256: str,
    expected_work_manifest_sha256: str,
    expected_release_sha256: str,
    expected_materialization_sha256: str,
) -> VerifiedConfirmationAdaptiveSourceMapPlane:
    """Rehydrate stores, rebuild the plan, and require byte-identical output."""

    expected_preflight = require_sha256(
        expected_preflight_sha256, "expected confirmation source-map preflight"
    )
    expected_work = require_sha256(
        expected_work_manifest_sha256,
        "expected confirmation source-map work manifest",
    )
    expected_terminal = require_sha256(
        expected_materialization_sha256,
        "expected confirmation source-map materialization",
    )
    existing = read_sealed_json(Path(output_root) / PREFLIGHT_NAME)
    runtime = existing.payload.get("runtime")
    _require(type(runtime) is dict, "confirmation replay runtime seal changed")
    rebuilt = publish_confirmation_adaptive_source_map_preflight(
        source_population,
        query_adapter,
        output_root=output_root,
        model=runtime["model"],
        gateway_url=runtime["gateway_url"],
        max_concurrency=runtime["max_concurrency"],
    )
    _require(
        rebuilt.preflight_artifact.sha256 == expected_preflight
        and rebuilt.work_manifest_artifact.sha256 == expected_work,
        "confirmation adaptive replay changed preflight bytes",
    )
    materialized = materialize_confirmation_adaptive_source_map(
        rebuilt,
        output_root=output_root,
        expected_preflight_sha256=expected_preflight,
        expected_work_manifest_sha256=expected_work,
        expected_release_sha256=expected_release_sha256,
    )
    _require(
        materialized.materialization_artifact.sha256 == expected_terminal,
        "confirmation adaptive materialization replay changed bytes",
    )
    replay_payload: dict[str, Any] = {
        "byte_identical": True,
        "format": REPLAY_FORMAT,
        "gold_loaded": False,
        "materialization_sha256": expected_terminal,
        "preflight_sha256": expected_preflight,
        "provider_calls_during_replay": 0,
        "query_map_adapter_receipt_sha256": query_adapter.receipt_sha256,
        "retained_transformer_token_state_bytes": 0,
        "source_gate_population_receipt_sha256": source_population.receipt_sha256,
        "source_mapper_materialization_receipt_sha256s": [
            row.receipt_sha256 for row in materialized.materializations
        ],
        "stores_revalidated_during_replay": True,
        "work_manifest_sha256": expected_work,
    }
    assert_gold_blind(replay_payload, path="confirmation_adaptive_source_map_replay")
    replay, _created = publish_sealed_json(
        Path(output_root) / REPLAY_NAME,
        replay_payload,
    )
    return VerifiedConfirmationAdaptiveSourceMapPlane(
        materialized.preflight_artifact,
        materialized.work_manifest_artifact,
        materialized.release_artifact,
        materialized.materialization_artifact,
        replay,
        materialized.completion_batch,
        source_population,
        query_adapter,
        materialized.questions,
        materialized.materializations,
    )


__all__ = [
    "CHECKPOINT_DIR_NAME",
    "ConfirmationAdaptiveSourceMapError",
    "ConfirmationAdaptiveSourceMapMaterialization",
    "ConfirmationAdaptiveSourceMapPreflight",
    "MATERIALIZATION_NAME",
    "PREFLIGHT_NAME",
    "RELEASE_NAME",
    "REPLAY_NAME",
    "VerifiedConfirmationAdaptiveSourceMapPlane",
    "WORK_MANIFEST_NAME",
    "approve_confirmation_adaptive_source_map_release",
    "materialize_confirmation_adaptive_source_map",
    "publish_confirmation_adaptive_source_map_preflight",
    "publish_confirmation_adaptive_source_map_from_streams",
    "replay_confirmation_adaptive_source_map",
    "run_confirmation_adaptive_source_map_provider",
]
