#!/usr/bin/env python3
"""Authenticated confirmation adapter for the frozen V2 evidence-map stage.

This module is intentionally a thin boundary around
``query_evidence_map_solver_v2_live``.  It joins an exact confirmation
query-payload plan to its replayed direct-answer plane, exports the exact map
messages, and seals approval for only the still-missing native journal pairs.
Provider execution, map parsing/materialization, runtime-ledger production,
and replay remain owned by the authoritative matched-eval implementation.

The adapter is population-size neutral and gold-blind.  It does not load a
benchmark, inspect reference answers, select ordinals, or invoke the rejected
standalone evidence solver.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.confirmation_query_payload_parent import ConfirmationQueryPayloadPlan
from tools.matched_eval import live
from tools.matched_eval import query_evidence_map_solver_v2_live as evidence_map_live
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
from tools.matched_eval.query_evidence_map_solver_v2_live import (
    EvidenceMapPlan,
    EvidenceMapRunResult,
    TwoPassProviderResult,
    VerifiedEvidenceMapPlane,
    build_evidence_map_plan,
    load_map_provider_journals,
    load_map_provider_population,
    materialize_evidence_map,
    preflight_evidence_map,
    replay_evidence_map,
    run_sealed_two_pass_provider,
)
from tools.matched_eval.query_payload_live import VerifiedQueryPayloadAnswerPlane


PROMPT_FORMAT = "memory-condense-confirmation-evidence-map-terra-prompts-v1"
PROMPT_ROW_FORMAT = f"{PROMPT_FORMAT}-row-v1"
PROVIDER_INPUT_FORMAT = f"{PROMPT_FORMAT}-provider-input-v1"
RELEASE_FORMAT = "memory-condense-confirmation-evidence-map-provider-release-v1"
PROMPT_NAME = "confirmation-evidence-map-terra-prompts-v1.json"
RELEASE_NAME = "confirmation-evidence-map-provider-release-v1.json"

_JOURNAL_NAME = re.compile(
    r"^(?P<key>[0-9a-f]{64})\.(?P<kind>request|response)\.json$"
)
_RELEASE_KEYS = frozenset(
    {
        "approval_opt_in",
        "checkpoint_namespace",
        "checkpoint_snapshot",
        "format",
        "gold_loaded",
        "map_preflight_sha256",
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
        "call_key_sha256",
        "messages_sha256",
        "request_journal_sha256",
        "response_journal_sha256",
    }
)


class ConfirmationEvidenceMapError(MatchedEvalContractError):
    """A confirmation evidence-map boundary invariant failed."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ConfirmationEvidenceMapError(message)


@dataclass(frozen=True, slots=True)
class ConfirmationEvidenceMapPlan:
    """Exact confirmation parent objects joined to the authoritative map plan."""

    query_payload: ConfirmationQueryPayloadPlan
    direct_plane: VerifiedQueryPayloadAnswerPlane
    map_plan: EvidenceMapPlan

    def __post_init__(self) -> None:
        _require(
            type(self.query_payload) is ConfirmationQueryPayloadPlan,
            "query-payload plan is not the exact confirmation type",
        )
        _require(
            type(self.direct_plane) is VerifiedQueryPayloadAnswerPlane,
            "direct plane is not an exact replayed query-payload plane",
        )
        _require(
            type(self.map_plan) is EvidenceMapPlan
            and self.map_plan.direct_plan is self.query_payload.answer_plan
            and self.map_plan.direct_plane is self.direct_plane,
            "authoritative evidence-map plan lost its exact parents",
        )

    @property
    def required_calls(self) -> int:
        return self.map_plan.required_calls


@dataclass(frozen=True, slots=True)
class ConfirmationEvidenceMapPreflight:
    """Sealed native preflight plus the exact submitted-message export."""

    plan: ConfirmationEvidenceMapPlan
    prompt_artifact: SealedArtifact
    map_preflight_artifact: SealedArtifact


ClientFactory = Callable[[str, str], Any]


def build_confirmation_evidence_map_plan(
    query_payload: ConfirmationQueryPayloadPlan,
    direct_plane: VerifiedQueryPayloadAnswerPlane,
    *,
    max_prompt_tokens: int = evidence_map_live.MAX_PROMPT_TOKENS,
    output_token_reserve: int = evidence_map_live.MAP_OUTPUT_TOKEN_RESERVE,
) -> ConfirmationEvidenceMapPlan:
    """Build the exact authoritative V2 map plan from replayed parents."""

    if type(query_payload) is not ConfirmationQueryPayloadPlan:
        raise TypeError("query_payload must be an exact ConfirmationQueryPayloadPlan")
    if type(direct_plane) is not VerifiedQueryPayloadAnswerPlane:
        raise TypeError(
            "direct_plane must be an exact VerifiedQueryPayloadAnswerPlane"
        )
    plan = build_evidence_map_plan(
        query_payload.answer_plan,
        direct_plane,
        max_prompt_tokens=max_prompt_tokens,
        output_token_reserve=output_token_reserve,
    )
    return ConfirmationEvidenceMapPlan(query_payload, direct_plane, plan)


def _plain_messages(row: evidence_map_live.EvidenceMapPlanRow) -> list[dict[str, str]]:
    _require(row.messages is not None, "preserved map row has no Terra prompt")
    assert row.messages is not None
    messages = [
        {"role": message.role, "content": message.content}
        for message in row.messages
    ]
    _require(
        identity_sha256(messages) == row.messages_sha256,
        f"evidence-map messages changed at ordinal {row.ordinal}",
    )
    return messages


def compile_confirmation_evidence_map_prompt_plane(
    plan: ConfirmationEvidenceMapPlan,
    *,
    map_preflight_sha256: str,
    max_concurrency: int = 4,
    gateway_url: str = live.DEFAULT_GATEWAY_URL,
) -> dict[str, Any]:
    """Compile the exact messages submitted by eligible V2 map rows."""

    if type(plan) is not ConfirmationEvidenceMapPlan:
        raise TypeError("plan must be an exact ConfirmationEvidenceMapPlan")
    preflight_sha = require_sha256(
        map_preflight_sha256, "evidence-map preflight SHA-256"
    )
    _require(
        type(max_concurrency) is int and max_concurrency > 0,
        "evidence-map max concurrency must be positive",
    )
    _require(
        type(gateway_url) is str and bool(gateway_url.strip()),
        "evidence-map gateway URL must be non-empty",
    )
    rows: list[dict[str, Any]] = []
    for prompt_index, row in enumerate(plan.map_plan.submitted_rows):
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
            "alias_receipt_sha256": row.alias_receipt_sha256,
            "format": PROMPT_ROW_FORMAT,
            "map_plan_row_receipt_sha256": row.receipt_sha256,
            "prompt_index": prompt_index,
            "provider_input": provider_input,
            "question_id": row.direct_plan_row.adapter.source.packet.question_id,
            "question_sha256": (
                row.direct_plan_row.adapter.source.packet.question_sha256
            ),
            "source_row_index": row.ordinal,
        }
        rows.append({**body, "row_receipt_sha256": identity_sha256(body)})
    submitted_question_ids = [row["question_id"] for row in rows]
    population = plan.map_plan.prompt_population
    if population is None:
        _require(not rows and plan.required_calls == 0, "empty map prompts changed")
        prompt_population_sha256 = None
    else:
        _require(
            len(rows)
            == plan.required_calls
            == population.logical_prompt_count
            == population.unique_prompt_count,
            "evidence-map submitted prompt population changed",
        )
        prompt_population_sha256 = population.prompt_population_sha256
    body = {
        "format": PROMPT_FORMAT,
        "gold_loaded": False,
        "map_plan_identity_sha256": plan.map_plan.plan_identity_sha256,
        "map_preflight_sha256": preflight_sha,
        "ordered_rows": rows,
        "physical_provider_calls": 0,
        "population": {
            "question_count": len(plan.map_plan.rows),
            "required_provider_calls": plan.required_calls,
            "submitted_question_ids_sha256": identity_sha256(
                submitted_question_ids
            ),
        },
        "prompt_population_sha256": prompt_population_sha256,
        "provider_execution_available": False,
        "renderer_id": evidence_map_live.MAP_RENDERER_ID,
        "runtime": {
            "gateway_url": gateway_url,
            "max_concurrency": max_concurrency,
            "max_prompt_tokens": plan.map_plan.max_prompt_tokens,
            "model": live.DEFAULT_TERRA_GATEWAY_MODEL,
            "output_token_reserve": plan.map_plan.output_token_reserve,
            "retries": 0,
        },
        "source_bindings": {
            "direct_answer_run_sha256": plan.direct_plane.run_sha256,
            "direct_answer_runtime_ledger_sha256": (
                plan.direct_plane.runtime_ledger_sha256
            ),
            "query_payload_plan_identity_sha256": (
                plan.query_payload.answer_plan.plan_identity_sha256
            ),
        },
        "status": "preflighted",
    }
    assert_gold_blind(body, path="confirmation_evidence_map_prompt_plane")
    return {**body, "prompt_plane_identity_sha256": identity_sha256(body)}


def publish_confirmation_evidence_map_preflight(
    plan: ConfirmationEvidenceMapPlan,
    *,
    output_root: str | Path,
    max_concurrency: int = 4,
    gateway_url: str = live.DEFAULT_GATEWAY_URL,
) -> ConfirmationEvidenceMapPreflight:
    """Publish the authoritative V2 preflight and its exact prompt export."""

    output = Path(output_root)
    native = preflight_evidence_map(plan.map_plan, output_root=output)
    prompt_payload = compile_confirmation_evidence_map_prompt_plane(
        plan,
        map_preflight_sha256=native.sha256,
        max_concurrency=max_concurrency,
        gateway_url=gateway_url,
    )
    native_prompts = native.payload.get("provider_prompts")
    _require(
        native_prompts
        == [
            row["provider_input"]["messages"]
            for row in prompt_payload["ordered_rows"]
        ],
        "exported map prompts differ from the authoritative preflight",
    )
    prompt, _created = publish_sealed_json(output / PROMPT_NAME, prompt_payload)
    return ConfirmationEvidenceMapPreflight(plan, prompt, native)


def _verified_preflight(
    preflight: ConfirmationEvidenceMapPreflight,
    *,
    output_root: str | Path,
    expected_prompt_sha256: str,
    expected_map_preflight_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact]:
    if type(preflight) is not ConfirmationEvidenceMapPreflight:
        raise TypeError(
            "preflight must be an exact ConfirmationEvidenceMapPreflight"
        )
    expected_prompt = require_sha256(
        expected_prompt_sha256, "evidence-map prompt plane SHA-256"
    )
    expected_native = require_sha256(
        expected_map_preflight_sha256, "authoritative evidence-map preflight SHA-256"
    )
    output = Path(output_root)
    prompt = read_sealed_json(output / PROMPT_NAME)
    native = read_sealed_json(output / evidence_map_live.MAP_PREFLIGHT_NAME)
    _require(
        prompt.sha256 == preflight.prompt_artifact.sha256 == expected_prompt,
        "evidence-map prompt plane SHA-256 changed",
    )
    _require(
        native.sha256 == preflight.map_preflight_artifact.sha256 == expected_native,
        "authoritative evidence-map preflight SHA-256 changed",
    )
    # Public no-clobber preflight recomputes the entire native payload from the
    # in-memory plan.  A resealed-but-different artifact is therefore refused.
    rebuilt = preflight_evidence_map(preflight.plan.map_plan, output_root=output)
    _require(
        rebuilt.sha256 == native.sha256 and rebuilt.payload == native.payload,
        "authoritative evidence-map preflight population changed",
    )
    expected_payload = compile_confirmation_evidence_map_prompt_plane(
        preflight.plan,
        map_preflight_sha256=native.sha256,
        max_concurrency=prompt.payload["runtime"]["max_concurrency"],
        gateway_url=prompt.payload["runtime"]["gateway_url"],
    )
    _require(
        prompt.payload == expected_payload,
        "evidence-map prompt plane changed",
    )
    return prompt, native


def _checkpoint_records(
    preflight: ConfirmationEvidenceMapPreflight,
    *,
    output_root: str | Path,
    map_preflight_sha256: str,
) -> tuple[dict[str, str], ...]:
    checkpoint = Path(output_root) / evidence_map_live.MAP_CHECKPOINT_DIR_NAME
    if not checkpoint.exists():
        return ()
    _require(
        checkpoint.is_dir() and not checkpoint.is_symlink(),
        "evidence-map checkpoint root is absent or unsafe",
    )
    requests: set[str] = set()
    responses: set[str] = set()
    for path in checkpoint.iterdir():
        _require(
            path.is_file() and not path.is_symlink(),
            "evidence-map checkpoint contains unsafe state",
        )
        if path.name == ".fast-completion-journal.lock":
            continue
        match = _JOURNAL_NAME.fullmatch(path.name)
        _require(match is not None, "evidence-map checkpoint contains foreign state")
        assert match is not None
        (requests if match.group("kind") == "request" else responses).add(
            match.group("key")
        )
    _require(
        requests == responses,
        "evidence-map request/response pair is incomplete; unsafe retry forbidden",
    )
    if not requests:
        return ()
    population = load_map_provider_population(
        output_root=output_root,
        expected_preflight_sha256=map_preflight_sha256,
    )
    # Private use is read-only and limited to authenticating already-complete
    # journals against the native runtime identity and call-key population.
    runtime = evidence_map_live._provider_runtime(  # noqa: SLF001
        population,
        client=None,
        max_concurrency=preflight.prompt_artifact.payload["runtime"][
            "max_concurrency"
        ],
        gateway_url=preflight.prompt_artifact.payload["runtime"]["gateway_url"],
    )
    try:
        with runtime._journal_guard():  # noqa: SLF001
            records = runtime._load_all_records()  # noqa: SLF001
    finally:
        runtime.close()
    _require(
        len(records) == len(requests),
        "evidence-map authenticated checkpoint population changed",
    )
    assert population.prompt_population is not None
    ordered: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in population.prompt_population.ordered_rows:
        record = records.get(row.messages_sha256)
        if record is None or row.messages_sha256 in seen:
            continue
        ordered.append(
            {
                "call_key_sha256": record.call_key_sha256,
                "messages_sha256": record.messages_sha256,
                "request_journal_sha256": record.request_journal_sha256,
                "response_journal_sha256": record.response_journal_sha256,
            }
        )
        seen.add(row.messages_sha256)
    _require(
        len(ordered) == len(requests),
        "evidence-map checkpoint order differs from the prompt population",
    )
    return tuple(ordered)


def approve_confirmation_evidence_map_release(
    preflight: ConfirmationEvidenceMapPreflight,
    *,
    output_root: str | Path,
    expected_prompt_sha256: str,
    expected_map_preflight_sha256: str,
    approve_provider_release: bool,
    authorized_provider_calls: int,
) -> SealedArtifact:
    """Seal explicit approval for exactly the missing native map journals."""

    _require(approve_provider_release is True, "provider release requires approval")
    prompt, native = _verified_preflight(
        preflight,
        output_root=output_root,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_map_preflight_sha256=expected_map_preflight_sha256,
    )
    records = _checkpoint_records(
        preflight,
        output_root=output_root,
        map_preflight_sha256=native.sha256,
    )
    remaining = preflight.plan.required_calls - len(records)
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == remaining,
        "evidence-map release authorization must equal exact remaining calls",
    )
    root = Path(output_root).resolve().as_posix()
    body = {
        "approval_opt_in": True,
        "checkpoint_namespace": evidence_map_live.MAP_CHECKPOINT_DIR_NAME,
        "checkpoint_snapshot": {
            "authenticated_complete_count": len(records),
            "ordered_records": list(records),
            "ordered_records_sha256": identity_sha256(list(records)),
        },
        "format": RELEASE_FORMAT,
        "gold_loaded": False,
        "map_preflight_sha256": native.sha256,
        "output_root": root,
        "output_root_sha256": identity_sha256({"canonical_root": root}),
        "physical_provider_calls": 0,
        "prompt_plane_sha256": prompt.sha256,
        "release_status": "approved_for_provider_execution",
        "required_authorized_provider_calls": remaining,
        "unsafe_retry_policy": "refuse-incomplete-request-response-pair-v1",
    }
    assert_gold_blind(body, path="confirmation_evidence_map_release")
    payload = {**body, "release_identity_sha256": identity_sha256(body)}
    release, _created = publish_sealed_json(Path(output_root) / RELEASE_NAME, payload)
    return release


def _verified_release(
    preflight: ConfirmationEvidenceMapPreflight,
    *,
    output_root: str | Path,
    expected_prompt_sha256: str,
    expected_map_preflight_sha256: str,
    expected_release_sha256: str,
) -> tuple[
    SealedArtifact,
    SealedArtifact,
    SealedArtifact,
    tuple[dict[str, str], ...],
]:
    prompt, native = _verified_preflight(
        preflight,
        output_root=output_root,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_map_preflight_sha256=expected_map_preflight_sha256,
    )
    release = read_sealed_json(Path(output_root) / RELEASE_NAME)
    expected_release = require_sha256(
        expected_release_sha256, "evidence-map provider release SHA-256"
    )
    _require(
        release.sha256 == expected_release,
        "evidence-map provider release SHA-256 changed",
    )
    _require(
        set(release.payload) == _RELEASE_KEYS,
        "evidence-map provider release schema changed",
    )
    body = dict(release.payload)
    declared = body.pop("release_identity_sha256", None)
    _require(
        declared == identity_sha256(body),
        "evidence-map provider release self-seal changed",
    )
    snapshot = release.payload.get("checkpoint_snapshot")
    _require(
        type(snapshot) is dict and set(snapshot) == _CHECKPOINT_SNAPSHOT_KEYS,
        "evidence-map release checkpoint schema changed",
    )
    rows = snapshot.get("ordered_records")
    _require(
        type(rows) is list
        and all(
            type(row) is dict and set(row) == _CHECKPOINT_RECORD_KEYS
            for row in rows
        ),
        "evidence-map release checkpoint rows changed",
    )
    released = tuple(dict(row) for row in rows)
    for index, row in enumerate(released):
        for key, value in row.items():
            require_sha256(value, f"evidence-map release record {index} {key}")
    _require(
        len({row["messages_sha256"] for row in released}) == len(released),
        "evidence-map release checkpoint records repeat",
    )
    root = Path(output_root).resolve().as_posix()
    _require(
        release.payload.get("format") == RELEASE_FORMAT
        and release.payload.get("release_status")
        == "approved_for_provider_execution"
        and release.payload.get("approval_opt_in") is True
        and release.payload.get("gold_loaded") is False
        and release.payload.get("checkpoint_namespace")
        == evidence_map_live.MAP_CHECKPOINT_DIR_NAME
        and release.payload.get("prompt_plane_sha256") == prompt.sha256
        and release.payload.get("map_preflight_sha256") == native.sha256
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
        "evidence-map provider release bindings changed",
    )
    current = _checkpoint_records(
        preflight,
        output_root=output_root,
        map_preflight_sha256=native.sha256,
    )
    current_by_message = {row["messages_sha256"]: row for row in current}
    _require(
        all(current_by_message.get(row["messages_sha256"]) == row for row in released),
        "evidence-map released checkpoint snapshot is not present",
    )
    assert_gold_blind(release.payload, path="confirmation_evidence_map_release")
    return prompt, native, release, released


def _default_client_factory(gateway_url: str, api_key_env: str) -> Any:
    api_key = os.environ.get(api_key_env, "").strip()
    _require(bool(api_key), f"provider API key is empty: {api_key_env}")
    return live._make_provider_client(api_key, gateway_url)  # noqa: SLF001


def run_confirmation_evidence_map_provider(
    preflight: ConfirmationEvidenceMapPreflight,
    *,
    output_root: str | Path,
    expected_prompt_sha256: str,
    expected_map_preflight_sha256: str,
    expected_release_sha256: str,
    enable_provider: bool,
    authorized_provider_calls: int,
    api_key_env: str = live.DEFAULT_API_KEY_ENV,
    client_factory: ClientFactory = _default_client_factory,
) -> TwoPassProviderResult:
    """Execute only missing calls through the native V2 map runtime."""

    prompt, native, release, released = _verified_release(
        preflight,
        output_root=output_root,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_map_preflight_sha256=expected_map_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    current = _checkpoint_records(
        preflight,
        output_root=output_root,
        map_preflight_sha256=native.sha256,
    )
    current_by_message = {row["messages_sha256"]: row for row in current}
    _require(
        all(current_by_message.get(row["messages_sha256"]) == row for row in released),
        "evidence-map checkpoint changed after release",
    )
    remaining = preflight.plan.required_calls - len(current)
    _require(
        enable_provider == bool(preflight.plan.required_calls),
        "evidence-map provider opt-in must match the prompt population",
    )
    _require(
        type(authorized_provider_calls) is int
        and authorized_provider_calls == remaining,
        "evidence-map provider authorization must equal exact remaining calls",
    )
    _require(
        remaining <= release.payload["required_authorized_provider_calls"],
        "evidence-map current state exceeds its sealed release budget",
    )
    population = load_map_provider_population(
        output_root=output_root,
        expected_preflight_sha256=native.sha256,
    )
    if remaining == 0 and preflight.plan.required_calls:
        result = load_map_provider_journals(
            preflight.plan.map_plan,
            output_root=output_root,
            expected_preflight_sha256=native.sha256,
            max_concurrency=prompt.payload["runtime"]["max_concurrency"],
            gateway_url=prompt.payload["runtime"]["gateway_url"],
        )
    else:
        client = (
            client_factory(prompt.payload["runtime"]["gateway_url"], api_key_env)
            if remaining
            else None
        )
        result = run_sealed_two_pass_provider(
            population,
            enable_provider=bool(preflight.plan.required_calls),
            authorized_provider_calls=preflight.plan.required_calls,
            client=client,
            max_concurrency=prompt.payload["runtime"]["max_concurrency"],
            gateway_url=prompt.payload["runtime"]["gateway_url"],
        )
    _require(
        result.physical_provider_calls == remaining
        and result.checkpoint_hits == len(current),
        "native evidence-map provider accounting differs from exact authorization",
    )
    return result


def materialize_confirmation_evidence_map(
    preflight: ConfirmationEvidenceMapPreflight,
    *,
    output_root: str | Path,
    expected_prompt_sha256: str,
    expected_map_preflight_sha256: str,
    expected_release_sha256: str,
) -> EvidenceMapRunResult:
    """Materialize only from a complete client-free native journal replay."""

    prompt, native, _release, _released = _verified_release(
        preflight,
        output_root=output_root,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_map_preflight_sha256=expected_map_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    journals = load_map_provider_journals(
        preflight.plan.map_plan,
        output_root=output_root,
        expected_preflight_sha256=native.sha256,
        max_concurrency=prompt.payload["runtime"]["max_concurrency"],
        gateway_url=prompt.payload["runtime"]["gateway_url"],
    )
    return materialize_evidence_map(
        preflight.plan.map_plan,
        output_root=output_root,
        expected_preflight_sha256=native.sha256,
        completion_batch=journals.batch,
        gateway_url=prompt.payload["runtime"]["gateway_url"],
    )


def replay_confirmation_evidence_map(
    preflight: ConfirmationEvidenceMapPreflight,
    *,
    output_root: str | Path,
    expected_prompt_sha256: str,
    expected_map_preflight_sha256: str,
    expected_release_sha256: str,
    expected_run_sha256: str,
) -> VerifiedEvidenceMapPlane:
    """Replay and return the exact downstream ``VerifiedEvidenceMapPlane``."""

    prompt, native, _release, _released = _verified_release(
        preflight,
        output_root=output_root,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_map_preflight_sha256=expected_map_preflight_sha256,
        expected_release_sha256=expected_release_sha256,
    )
    verified = replay_evidence_map(
        preflight.plan.map_plan,
        output_root=output_root,
        expected_preflight_sha256=native.sha256,
        expected_run_sha256=expected_run_sha256,
        max_concurrency=prompt.payload["runtime"]["max_concurrency"],
        gateway_url=prompt.payload["runtime"]["gateway_url"],
    )
    _require(
        type(verified) is VerifiedEvidenceMapPlane,
        "evidence-map replay did not return the exact downstream plane",
    )
    return verified


__all__ = [
    "ConfirmationEvidenceMapError",
    "ConfirmationEvidenceMapPlan",
    "ConfirmationEvidenceMapPreflight",
    "PROMPT_FORMAT",
    "PROMPT_NAME",
    "PROMPT_ROW_FORMAT",
    "PROVIDER_INPUT_FORMAT",
    "RELEASE_FORMAT",
    "RELEASE_NAME",
    "approve_confirmation_evidence_map_release",
    "build_confirmation_evidence_map_plan",
    "compile_confirmation_evidence_map_prompt_plane",
    "materialize_confirmation_evidence_map",
    "publish_confirmation_evidence_map_preflight",
    "replay_confirmation_evidence_map",
    "run_confirmation_evidence_map_provider",
]
