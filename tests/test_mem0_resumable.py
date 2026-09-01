from __future__ import annotations

import hashlib
import json
import os
import shutil
import copy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.mem0_eval.resumable import (
    AppendOnlyResumeJournal,
    RESUME_CHECKPOINT_GC_FORMAT,
    RESUME_CLEANUP_FORMAT,
    RESUME_PUBLICATION_FORMAT,
    RESUME_STATE_REMOVAL_FORMAT,
    ResumeAmbiguityError,
    ResumePlan,
    ResumableShardError,
    append_active_state_removed,
    append_checkpoint_gc,
    append_cleanup_closed,
    append_commit,
    append_intent,
    append_prefix_sealed,
    append_presend_rollback,
    append_send_attempt,
    append_terminal_search,
    append_terminal_published,
    canonical_json_sha256,
    create_immutable_state_snapshot,
    checkpoint_authority_from_replay,
    deterministic_user_scope,
    prefix_close_receipt,
    prefix_zero_restore_receipt,
    publish_sealed_json,
    read_journal,
    read_sealed_json,
    replay_journal,
    reconcile_published_checkpoint,
    rehydration_material,
    remove_verified_snapshot_root,
    restore_snapshot_to_fresh_state,
    state_tree_receipt,
    suffix_counter_seed,
    verify_completed_add_receipts,
    verify_immutable_state_snapshot,
)
from tools.mem0_eval import resumable_runner
from tools.mem0_eval.policy import (
    MEM0_EXTRACTION_GATEWAY_URL,
    MEM0_EXTRACTION_MODEL,
    MEM0_EXTRACTION_PROVIDER,
    MEM0_EXTRACTION_REVISION,
    MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
)
from tools.mem0_eval.run_shard import RetrievalStageAuthorization


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _plan(*, adds: int = 2, searches: int = 1) -> ResumePlan:
    authorization = _sha("authorization")
    return ResumePlan(
        authorization_sha256=authorization,
        mem0_policy_sha256=_sha("policy"),
        source_validation_policy_sha256=_sha("source-policy"),
        source_implementation_sha256=_sha("source-code"),
        source_environment_lock_sha256=_sha("source-lock"),
        mem0_tool_implementation_sha256=_sha("tool-code"),
        mem0_environment_lock_sha256=_sha("tool-lock"),
        sample_offset=0,
        sample_sha256=_sha("sample"),
        raw_history_bundle_sha256=_sha("history"),
        ordered_batch_sha256s=tuple(_sha(f"batch-{index}") for index in range(adds)),
        authorized_add_operations=adds,
        authorized_extraction_calls=adds,
        authorized_search_operations=searches,
        user_scope=deterministic_user_scope(authorization),
    )


def _new_journal(tmp_path: Path, plan: ResumePlan) -> tuple[AppendOnlyResumeJournal, object]:
    journal = AppendOnlyResumeJournal(tmp_path / "resume.jsonl", plan)
    state = journal.create(
        owned_state_path="state/working",
        snapshot_root_path="state/snapshots",
    )
    return journal, state


def _fake_owned_state(tmp_path: Path, name: str = "working") -> Path:
    state = tmp_path / "state" / name
    state.mkdir(parents=True)
    (state / ".memory-condense-owned-state").write_text("a" * 32, encoding="utf-8")
    (state / "history.sqlite").write_bytes(b"history-v1")
    collection = state / "qdrant" / "collection"
    collection.mkdir(parents=True)
    (collection / "storage.sqlite").write_bytes(b"qdrant-v1")
    return state


def _source_ref(ordinal: int, *, turn_count: int | None = None) -> dict[str, object]:
    count = turn_count if turn_count is not None else (1 if ordinal % 2 == 0 else 2)
    return {
        "sample_id": "sample-0",
        "source": f"session-{ordinal}",
        "session": f"Session {ordinal}",
        "session_index": ordinal,
        "original_session_index": ordinal,
        "batch_index": ordinal,
        "date": f"2026/08/{ordinal + 1:02d}",
        "turn_start": ordinal * 2,
        "turn_count": count,
        "roles": ["user", "assistant"][:count],
    }


def _unique_refs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in rows:
        if row not in result:
            result.append(row)
    return result


def _stats(
    state: object,
    *,
    ids: tuple[str, ...],
    latency: float,
    tokens: int,
) -> dict[str, object]:
    prior = rehydration_material(state)
    previous = prior["adapter_stats"]
    if previous is None:
        previous = {
            "add_calls": 0,
            "add_attempted_calls": 0,
            "add_completed_calls": 0,
            "add_failed_calls": 0,
            "search_calls": 0,
            "add_latency_s": 0.0,
            "search_latency_s": 0.0,
            "add_raw_message_tokens": 0,
            "search_query_tokens": 0,
            "search_raw_memory_tokens": 0,
            "search_context_tokens": 0,
            "search_prompt_token_proxy": 0,
            "search_prompt_tokens": 0,
            "add_returned_memories": 0,
            "unique_ledger_memories": 0,
            "search_returned_memories": 0,
            "search_packed_memories": 0,
            "released_scopes": 0,
            "provider_prompt_tokens": None,
            "provider_completion_tokens": None,
            "provider_usage_status": "unavailable_from_mem0_oss_public_api",
            "token_counter_identity": "test-counter:v1",
            "token_counter_identity_verified": True,
        }
    row = dict(previous)
    row.update(
        add_calls=int(row["add_calls"]) + 1,
        add_attempted_calls=int(row["add_attempted_calls"]) + 1,
        add_completed_calls=int(row["add_completed_calls"]) + 1,
        add_latency_s=float(row["add_latency_s"]) + latency,
        add_raw_message_tokens=int(row["add_raw_message_tokens"]) + tokens,
        add_returned_memories=int(row["add_returned_memories"]) + len(ids),
        unique_ledger_memories=len(
            {entry["memory_id"] for entry in prior["ledger_projection"]} | set(ids)
        ),
    )
    return row


def _extraction_request_identity_sha256() -> str:
    return canonical_json_sha256(
        {
            "format": "memory-condense-mem0-extraction-request-v1",
            "route_identity_sha256": MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 2_000,
            "sampling_parameters": "omitted",
            "sdk_retries": 0,
            "http_transport_retries": 0,
            "follow_redirects": False,
            "trust_env": False,
            "timeout_seconds": 600.0,
            "connect_timeout_seconds": 30.0,
            "cap_boundary": "httpx.BaseTransport.handle_request",
        }
    )


def _segment_transport_receipt(
    *, authorized: int, completed: int
) -> dict[str, object]:
    input_tokens = completed * 11
    output_tokens = completed * 3
    return {
        "kind": "local_transport_send_cap",
        "role": "extraction",
        "authorized": authorized,
        "attempted": completed,
        "completed": completed,
        "failed": 0,
        "rejected": 0,
        "retries_authorized": 0,
        "provider_usage_status": "provider_reported_exact",
        "provider_usage_records": completed,
        "provider_input_tokens": input_tokens,
        "provider_output_tokens": output_tokens,
        "provider_total_tokens": input_tokens + output_tokens,
        "provider_latency_s": completed * 0.01,
        "production_eligible": True,
        "provider": MEM0_EXTRACTION_PROVIDER,
        "model": MEM0_EXTRACTION_MODEL,
        "revision": MEM0_EXTRACTION_REVISION,
        "route_identity_sha256": MEM0_EXTRACTION_ROUTE_IDENTITY_SHA256,
        "request_identity_sha256": _extraction_request_identity_sha256(),
        "gateway_url": MEM0_EXTRACTION_GATEWAY_URL,
        "max_completion_tokens": 2_000,
        "sampling_parameters_omitted": True,
        "sdk_retries": 0,
        "http_transport_retries": 0,
        "follow_redirects": False,
        "trust_env": False,
        "cap_boundary": "httpx.BaseTransport.handle_request",
        "external_http_attempts_certified": True,
        "external_provider_persistence_certified": False,
    }


def _counter_receipts(
    state: object, total: int
) -> tuple[dict[str, object], dict[str, object]]:
    plan = state.plan
    seeded_prefix = state.sealed_prefix
    segment_authorized = min(
        resumable_runner.DEFAULT_SEGMENT_ADDS,
        plan.authorized_extraction_calls - seeded_prefix,
    )
    segment_completed = total - seeded_prefix
    logical_segment = {
        "kind": "mem0_memory_llm_generate_response_logical_calls",
        "boundary": "Memory.llm.generate_response",
        "count_semantics": "local_logical_wrapper_calls_not_http_attempts",
        "external_http_attempts_certified": False,
        "authorized_local_wrapper_retries": 0,
        "external_retry_attempts_certified": False,
        "authorized": segment_authorized,
        "attempted": segment_completed,
        "completed": segment_completed,
        "failed": 0,
        "rejected": 0,
        "infer_true_adds_started": segment_completed,
        "infer_true_adds_exactly_one_call": segment_completed,
        "one_logical_call_per_infer_true_add_certified": (
            segment_completed == segment_authorized
        ),
        "persisted_request_token_state": False,
        "retained_request_token_state_bytes": 0,
        "request_token_state_evidence_kind": (
            "local_injected_request_token_state_contract"
        ),
        "external_provider_persistence_certified": False,
    }
    logical = resumable_runner._cumulative_receipt(
        logical_segment,
        prefix=seeded_prefix,
        full_authorized=plan.authorized_extraction_calls,
        logical=True,
    )
    segment_transport = _segment_transport_receipt(
        authorized=segment_authorized,
        completed=segment_completed,
    )
    prior_transport = (
        state.commits[seeded_prefix - 1]["transport_receipt"]
        if seeded_prefix
        else None
    )
    transport = resumable_runner._cumulative_receipt(
        segment_transport,
        prefix=seeded_prefix,
        full_authorized=plan.authorized_extraction_calls,
        logical=False,
        prior_cumulative=prior_transport,
    )
    return logical, transport


def _append_one(
    journal: AppendOnlyResumeJournal,
    state: object,
    *,
    ordinal: int,
    ids: tuple[str, ...] | None = None,
    turn_count: int | None = None,
) -> object:
    state = append_intent(journal, state, ordinal=ordinal, session_sha256=_sha("session"))
    state = append_send_attempt(journal, state, request_sha256=_sha(f"request-{ordinal}"))
    source = _source_ref(ordinal, turn_count=turn_count)
    prior = rehydration_material(state)
    window = _unique_refs(list(prior["request_window_deque"]))
    if source not in window:
        window.append(source)
    ids = ids if ids is not None else (f"memory-{ordinal}",)
    latency = 0.25 + ordinal
    tokens = 10 + ordinal
    logical, transport = _counter_receipts(state, ordinal + 1)
    return append_commit(
        journal,
        state,
        response_sha256=_sha(f"response-{ordinal}"),
        returned_memory_ids=ids,
        source_ref=source,
        request_window_refs=window,
        adapter_stats=_stats(
            state, ids=ids, latency=latency, tokens=tokens
        ),
        logical_meter_receipt=logical,
        transport_receipt=transport,
        add_latency_s=latency,
        raw_message_tokens=tokens,
        scope_protocol=True,
    )


def _seal(
    tmp_path: Path,
    journal: AppendOnlyResumeJournal,
    state: object,
    owned_state: Path,
) -> object:
    snapshot = _snapshot(tmp_path, journal, state, owned_state)
    return append_prefix_sealed(journal, state, snapshot_receipt=snapshot)


def _reseal_receipt(value: dict[str, object], field: str) -> None:
    body = copy.deepcopy(value)
    body.pop(field, None)
    value[field] = canonical_json_sha256(body)


def _reseal_journal_entry(value: dict[str, object]) -> None:
    _reseal_receipt(value, "entry_sha256")


def _write_usage_material(
    state: object,
    owned_state: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    latest = state.commits[-1]
    transport = latest["transport_receipt"]
    closure_body = {
        "format": "memory-condense-mem0-resumable-transport-closure-v1",
        "segment_authorized_calls": transport["segment_authorized"],
        "transport_closed": True,
        "budget_closed_exactly": True,
        "provider_usage_complete": True,
        "sdk_retries": 0,
        "http_transport_retries": 0,
        "transport_receipt": transport["segment_receipt"],
        "transport_receipt_sha256": transport["segment_receipt_sha256"],
    }
    closure = {
        **closure_body,
        "receipt_sha256": canonical_json_sha256(closure_body),
    }
    prefix_before = state.sealed_prefix
    segment_adds = state.committed_prefix - prefix_before
    launch_authority = {
        "format": "memory-condense-mem0-live-launch-authority-v1",
        "preflight_sha256": _sha("live-preflight"),
        "launch_manifest_sha256": _sha("live-manifest"),
        "shard_launch_sha256": _sha("live-shard-launch"),
        "shard_launch_payload_sha256": _sha("live-shard-payload"),
        "plan_sha256": state.plan.sha256,
        "authorization_sha256": state.plan.authorization_sha256,
        "journal_path_sha256": state.entries[0]["journal_path_sha256"],
        "sample_offset": state.plan.sample_offset,
        "namespace": state.plan.user_scope,
        "namespace_sha256": hashlib.sha256(
            state.plan.user_scope.encode("utf-8")
        ).hexdigest(),
        "mem0_policy_sha256": state.plan.mem0_policy_sha256,
        "mem0_tool_implementation_sha256": (
            state.plan.mem0_tool_implementation_sha256
        ),
        "mem0_environment_lock_sha256": state.plan.mem0_environment_lock_sha256,
        "retained_transformer_token_state_bytes": 0,
    }
    authorization_body = {
        "format": "memory-condense-mem0-one-use-segment-authorization-v1",
        "plan_sha256": state.plan.sha256,
        "authorization_sha256": state.plan.authorization_sha256,
        "journal_path_sha256": state.entries[0]["journal_path_sha256"],
        "prefix_before": prefix_before,
        "prefix_after": state.committed_prefix,
        "generation": state.generation,
        "prior_checkpoint_authority_sha256": state.checkpoint_authority_sha256,
        "authorized_provider_calls": segment_adds,
        "authorized_add_operations": segment_adds,
        "provider_retries": 0,
        "namespace": state.plan.user_scope,
        "retained_transformer_token_state_bytes": 0,
        "live_launch_authority": launch_authority,
        "live_launch_authority_sha256": canonical_json_sha256(
            launch_authority
        ),
    }
    authorization_receipt = {
        **authorization_body,
        "receipt_sha256": canonical_json_sha256(authorization_body),
    }
    activity_body = {
        "format": "memory-condense-mem0-resumable-write-activity-v1",
        "embedding_attempted": segment_adds,
        "embedding_completed": segment_adds,
        "embedding_failed": 0,
        "embedding_input_token_proxy": segment_adds * 7,
        "embedding_latency_s": segment_adds * 0.02,
        "storage_attempted": segment_adds,
        "storage_completed": segment_adds,
        "storage_failed": 0,
        "storage_latency_s": segment_adds * 0.03,
        "wrappers_installed": True,
        "wrappers_restored": True,
    }
    activity_receipt = {
        **activity_body,
        "receipt_sha256": canonical_json_sha256(activity_body),
    }
    owned_tree = state_tree_receipt(owned_state)
    usage = resumable_runner._complete_write_usage_attestation(
        state=state,
        segment_authorization_receipt=authorization_receipt,
        segment_write_activity_receipt=activity_receipt,
        suspended={
            "transport_closure": closure,
            "transport_closure_sha256": closure["receipt_sha256"],
            "owned_state_tree": owned_tree,
            "namespace_persisted_memory_count": latest["adapter_stats"][
                "unique_ledger_memories"
            ],
        },
    )
    return closure, usage, owned_tree


def _snapshot(
    tmp_path: Path,
    journal: AppendOnlyResumeJournal,
    state: object,
    owned_state: Path,
) -> dict[str, object]:
    closure, usage, owned_tree = _write_usage_material(state, owned_state)
    closed = prefix_close_receipt(
        state,
        history_sqlite_closed=True,
        qdrant_local_collections_closed=2,
        qdrant_clients_closed=1,
        transport_closed=True,
        transport_closure_receipt=closure,
        write_usage_attestation=usage,
        expected_storage_bytes=owned_tree["total_bytes"],
    )
    authority = checkpoint_authority_from_replay(
        state, handles_closed_receipt=closed
    )
    snapshot = create_immutable_state_snapshot(
        journal_path=journal.path,
        owned_state_dir=owned_state,
        snapshot_root=tmp_path / "state" / "snapshots",
        committed_prefix=state.committed_prefix,
        checkpoint_authority=authority,
    )
    return snapshot


def test_journal_round_trip_and_every_commit_is_reverified(tmp_path: Path) -> None:
    plan = _plan(adds=1)
    journal, state = _new_journal(tmp_path, plan)
    state = _append_one(journal, state, ordinal=0)
    owned = _fake_owned_state(tmp_path)
    state = _seal(tmp_path, journal, state, owned)

    replay = replay_journal(read_journal(journal.path), expected_plan=plan)
    assert replay.committed_prefix == replay.sealed_prefix == 1
    assert replay.resume_safe is True
    verify_completed_add_receipts(
        replay, ordered_batch_sha256s=plan.ordered_batch_sha256s
    )


@pytest.mark.parametrize(
    ("field", "mutated"),
    (("authorized", 3), ("seeded_prefix", 1)),
)
def test_replay_rejects_resealed_cumulative_transport_authority_tamper(
    tmp_path: Path, field: str, mutated: int
) -> None:
    plan = _plan(adds=2)
    journal, state = _new_journal(tmp_path, plan)
    state = _append_one(journal, state, ordinal=0)
    rows = [copy.deepcopy(dict(row)) for row in state.entries]
    commit = rows[-1]
    transport = commit["transport_receipt"]
    transport[field] = mutated
    commit["transport_receipt_sha256"] = canonical_json_sha256(transport)
    _reseal_journal_entry(commit)

    with pytest.raises(ResumableShardError, match="cumulative transport changed"):
        replay_journal(rows, expected_plan=plan)


@pytest.mark.parametrize("tamper", ("authorization", "observed"))
def test_replay_rejects_coherently_resealed_write_attestation_tamper(
    tmp_path: Path, tamper: str
) -> None:
    plan = _plan(adds=1)
    journal, state = _new_journal(tmp_path, plan)
    state = _append_one(journal, state, ordinal=0)
    state = _seal(tmp_path, journal, state, _fake_owned_state(tmp_path))
    rows = [copy.deepcopy(dict(row)) for row in state.entries]
    seal = rows[-1]
    usage = seal["write_usage_attestation"]
    if tamper == "authorization":
        authorization = usage["segment_authorization_receipt"]
        authorization["namespace"] = "forged-namespace"
        _reseal_receipt(authorization, "receipt_sha256")
        usage["segment_authorization_receipt_sha256"] = authorization[
            "receipt_sha256"
        ]
    else:
        observed = usage["observed"]
        observed["persisted_memory_count"] = (
            observed["returned_memory_count"] + 1
        )
        usage["observed_sha256"] = canonical_json_sha256(observed)
    _reseal_receipt(usage, "receipt_sha256")
    seal["write_usage_attestation_sha256"] = usage["receipt_sha256"]
    _reseal_journal_entry(seal)

    with pytest.raises(ResumableShardError):
        replay_journal(rows, expected_plan=plan)


def test_nonfinal_short_segment_cannot_publish_a_write_attestation(
    tmp_path: Path,
) -> None:
    plan = _plan(adds=257)
    journal, state = _new_journal(tmp_path, plan)
    state = _append_one(journal, state, ordinal=0)
    with pytest.raises(
        ResumableShardError, match="segment authorization binding changed"
    ):
        _snapshot(tmp_path, journal, state, _fake_owned_state(tmp_path))


def _synthetic_cumulative_stats(prefix: int) -> dict[str, object]:
    return {
        "add_calls": prefix,
        "add_attempted_calls": prefix,
        "add_completed_calls": prefix,
        "add_failed_calls": 0,
        "search_calls": 0,
        "add_latency_s": prefix * 0.25,
        "search_latency_s": 0.0,
        "add_raw_message_tokens": prefix * 10,
        "search_query_tokens": 0,
        "search_raw_memory_tokens": 0,
        "search_context_tokens": 0,
        "search_prompt_token_proxy": 0,
        "search_prompt_tokens": 0,
        "add_returned_memories": prefix,
        "unique_ledger_memories": prefix,
        "search_returned_memories": 0,
        "search_packed_memories": 0,
        "released_scopes": 0,
        "provider_prompt_tokens": None,
        "provider_completion_tokens": None,
        "provider_usage_status": "unavailable_from_mem0_oss_public_api",
        "token_counter_identity": "test-counter:v1",
        "token_counter_identity_verified": True,
    }


def test_exact_256_authority_then_final_tail_attestation_validates(
    tmp_path: Path,
) -> None:
    plan = _plan(adds=257)
    _journal, empty = _new_journal(tmp_path, plan)
    owned = _fake_owned_state(tmp_path)
    segment_256 = _segment_transport_receipt(authorized=256, completed=256)
    transport_256 = resumable_runner._cumulative_receipt(
        segment_256,
        prefix=0,
        full_authorized=257,
        logical=False,
    )
    state_256 = replace(
        empty,
        committed_prefix=256,
        commits=(
            {
                "adapter_stats": _synthetic_cumulative_stats(256),
                "transport_receipt": transport_256,
            },
        ),
    )
    _closure_256, usage_256, _tree = _write_usage_material(
        state_256, owned
    )
    assert usage_256["segment_authorization_receipt"][
        "authorized_provider_calls"
    ] == 256

    seal_256 = {
        "kind": "prefix_sealed",
        "committed_prefix": 256,
        "generation": 0,
        "snapshot_authority_sha256": _sha("prefix-256-authority"),
        "write_usage_attestation": usage_256,
        "write_usage_attestation_sha256": usage_256["receipt_sha256"],
    }
    segment_tail = _segment_transport_receipt(authorized=1, completed=1)
    transport_257 = resumable_runner._cumulative_receipt(
        segment_tail,
        prefix=256,
        full_authorized=257,
        logical=False,
        prior_cumulative=transport_256,
    )
    state_257 = replace(
        empty,
        entries=(empty.entries[0], seal_256),
        committed_prefix=257,
        sealed_prefix=256,
        latest_prefix_seal=seal_256,
        commits=(
            {
                "adapter_stats": _synthetic_cumulative_stats(257),
                "transport_receipt": transport_257,
            },
        ),
    )
    _closure_tail, usage_tail, _tree = _write_usage_material(
        state_257, owned
    )
    assert usage_tail["prior_write_usage_attestation_sha256"] == usage_256[
        "receipt_sha256"
    ]
    assert usage_tail["segment_authorization_receipt"][
        "authorized_provider_calls"
    ] == 1


def test_final_tail_attestation_survives_full_journal_replay(
    tmp_path: Path,
) -> None:
    plan = _plan(adds=2)
    journal, state = _new_journal(tmp_path, plan)
    owned = _fake_owned_state(tmp_path)
    state = _append_one(journal, state, ordinal=0)
    state = _append_one(journal, state, ordinal=1)
    state = _seal(tmp_path, journal, state, owned)
    replay = replay_journal(read_journal(journal.path), expected_plan=plan)
    authorization = replay.latest_prefix_seal["write_usage_attestation"][
        "segment_authorization_receipt"
    ]
    assert authorization["prefix_before"] == 0
    assert authorization["prefix_after"] == 2
    assert replay.resume_safe is True


def test_presend_intent_can_roll_back_to_latest_prefix_authority(
    tmp_path: Path,
) -> None:
    plan = _plan()
    journal, state = _new_journal(tmp_path, plan)
    state = append_intent(
        journal, state, ordinal=0, session_sha256=_sha("session-0")
    )
    assert state.requires_rollback is True

    working = tmp_path / "state" / "working"
    restore = prefix_zero_restore_receipt(
        state, destination_state_dir=working
    )
    state = append_presend_rollback(journal, state, restore_receipt=restore)
    assert state.generation == 1
    assert state.committed_prefix == state.sealed_prefix == 0
    assert state.resume_safe is True


def test_send_without_commit_is_externally_ambiguous_and_cannot_roll_back(
    tmp_path: Path,
) -> None:
    plan = _plan()
    journal, state = _new_journal(tmp_path, plan)
    state = append_intent(
        journal, state, ordinal=0, session_sha256=_sha("session-0")
    )
    state = append_send_attempt(
        journal, state, request_sha256=_sha("request-0")
    )

    assert state.externally_ambiguous is True
    with pytest.raises(ResumeAmbiguityError):
        state.require_resumable()
    with pytest.raises(ResumeAmbiguityError):
            append_presend_rollback(journal, state, restore_receipt={})


def test_commit_without_prefix_seal_is_not_resumable(tmp_path: Path) -> None:
    plan = _plan()
    journal, state = _new_journal(tmp_path, plan)
    state = _append_one(journal, state, ordinal=0)

    assert state.externally_ambiguous is True
    with pytest.raises(ResumeAmbiguityError):
        state.require_resumable()


def test_provider_free_launch_replay_rejects_valid_unsealed_commit(
    tmp_path: Path,
) -> None:
    from tools.mem0_eval import resumable_launch

    plan = _plan()
    journal = AppendOnlyResumeJournal(tmp_path / "resume.jsonl", plan)
    state = journal.create(
        owned_state_path="owned-state", snapshot_root_path="snapshots"
    )
    _append_one(journal, state, ordinal=0)
    payload = {
        "paths": {
            "journal_run_root_relative": "resume.jsonl",
            "runner_paths_relative_to_journal_parent": {
                "owned_state": "owned-state",
                "snapshot_root": "snapshots",
                "terminal_stage": "terminal-stage.json",
                "retrieval_artifact": "retrieval.json",
                "retrieval_trace": "retrieval.trace.json",
            },
        }
    }
    binding = SimpleNamespace(sample_offset=0, plan=plan)
    with pytest.raises(ResumeAmbiguityError):
        resumable_launch._journal_status(
            tmp_path,
            payload,
            binding,
            expected_live_launch_authority={},
        )


@pytest.mark.parametrize("mutation", ["truncate", "reorder", "corrupt_digest"])
def test_journal_corruption_fails_closed(tmp_path: Path, mutation: str) -> None:
    plan = _plan()
    journal, state = _new_journal(tmp_path, plan)
    state = append_intent(journal, state, ordinal=0, session_sha256=_sha("session"))
    rows = journal.path.read_bytes().splitlines(keepends=True)
    if mutation == "truncate":
        journal.path.write_bytes(b"".join(rows)[:-1])
    elif mutation == "reorder":
        journal.path.write_bytes(rows[1] + rows[0])
    else:
        row = json.loads(rows[-1])
        row["entry_sha256"] = "0" * 64
        rows[-1] = (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        journal.path.write_bytes(b"".join(rows))

    with pytest.raises(ResumableShardError):
        read_journal(journal.path) if mutation == "truncate" else replay_journal(
            read_journal(journal.path), expected_plan=plan
        )


def test_snapshot_is_immutable_hash_verified_and_restorable(tmp_path: Path) -> None:
    plan = _plan(adds=1)
    journal, replay = _new_journal(tmp_path, plan)
    replay = _append_one(journal, replay, ordinal=0)
    state = _fake_owned_state(tmp_path)
    snapshot = _snapshot(tmp_path, journal, replay, state)
    snapshot_path = tmp_path / snapshot["snapshot_path"]
    verified = verify_immutable_state_snapshot(
        snapshot_path,
        expected_authority_sha256=snapshot["snapshot_authority_sha256"],
        expected_manifest_sha256=snapshot["snapshot_manifest_sha256"],
        expected_tree_sha256=snapshot["snapshot_tree_sha256"],
        expected_ownership_token_sha256=snapshot["ownership_token_sha256"],
    )
    restored = restore_snapshot_to_fresh_state(
        snapshot_path=snapshot_path,
        destination_state_dir=tmp_path / "restored",
        expected_authority_sha256=snapshot["snapshot_authority_sha256"],
        expected_manifest_sha256=snapshot["snapshot_manifest_sha256"],
        expected_tree_sha256=snapshot["snapshot_tree_sha256"],
        expected_ownership_token_sha256=snapshot["ownership_token_sha256"],
    )
    assert (
        restored["restored_snapshot_tree_sha256"]
        == verified["snapshot_tree_sha256"]
    )


def test_snapshot_corruption_is_rejected(tmp_path: Path) -> None:
    plan = _plan(adds=1)
    journal, replay = _new_journal(tmp_path, plan)
    replay = _append_one(journal, replay, ordinal=0)
    state = _fake_owned_state(tmp_path)
    snapshot = _snapshot(tmp_path, journal, replay, state)
    snapshot_path = tmp_path / snapshot["snapshot_path"]
    (snapshot_path / "state" / "history.sqlite").write_bytes(b"tampered")
    with pytest.raises(ResumableShardError, match="manifest"):
        verify_immutable_state_snapshot(
            snapshot_path,
            expected_authority_sha256=snapshot["snapshot_authority_sha256"],
            expected_manifest_sha256=snapshot["snapshot_manifest_sha256"],
            expected_tree_sha256=snapshot["snapshot_tree_sha256"],
            expected_ownership_token_sha256=snapshot["ownership_token_sha256"],
        )


def test_wrong_ownership_marker_is_rejected(tmp_path: Path) -> None:
    state = _fake_owned_state(tmp_path)
    (state / ".memory-condense-owned-state").write_text("not-a-token", encoding="utf-8")
    with pytest.raises(ResumableShardError, match="ownership token"):
        state_tree_receipt(state)


def test_early_search_and_early_cleanup_are_forbidden(tmp_path: Path) -> None:
    plan = _plan()
    journal, state = _new_journal(tmp_path, plan)
    state = _append_one(journal, state, ordinal=0)
    with pytest.raises(ResumableShardError, match="before every add"):
        append_terminal_search(
            journal,
            state,
            terminal_stage_path="eval/terminal-stage.json",
            terminal_stage_sha256=_sha("terminal-stage"),
            terminal_result_sha256=_sha("terminal-result"),
            terminal_trace_sha256=_sha("terminal-trace"),
            completed_search_operations=1,
        )
    with pytest.raises(ResumableShardError, match="checkpoint GC"):
        append_cleanup_closed(
            journal,
            state,
            cleanup_receipt={},
        )


def test_terminal_search_then_cleanup_is_the_only_terminal_order(tmp_path: Path) -> None:
    plan = _plan()
    journal, state = _new_journal(tmp_path, plan)
    owned = _fake_owned_state(tmp_path)
    state = _append_one(journal, state, ordinal=0)
    (owned / "history.sqlite").write_bytes(b"history-after-one")
    state = _append_one(journal, state, ordinal=1)
    state = _seal(tmp_path, journal, state, owned)
    state = append_terminal_search(
        journal,
        state,
        terminal_stage_path="eval/terminal-stage.json",
        terminal_stage_sha256=_sha("terminal-stage"),
        terminal_result_sha256=_sha("terminal-result"),
        terminal_trace_sha256=_sha("terminal-trace"),
        completed_search_operations=1,
    )
    state = append_active_state_removed(
        journal,
        state,
        removal_receipt={
            "format": RESUME_STATE_REMOVAL_FORMAT,
            "plan_sha256": plan.sha256,
            "terminal_stage_sha256": _sha("terminal-stage"),
            "owned_state_path_sha256": _sha("owned-state-path"),
            "owned_state_removed": True,
            "snapshots_retained": True,
        },
    )
    publication = {
        "format": RESUME_PUBLICATION_FORMAT,
        "terminal_stage_sha256": _sha("terminal-stage"),
        "official_artifact_path": "eval/terminal.json",
        "official_artifact_sha256": _sha("terminal-result"),
        "official_trace_path": "eval/terminal.trace.json",
        "official_trace_sha256": _sha("terminal-trace"),
        "outputs_verified": True,
    }
    state = append_terminal_published(
        journal, state, publication_receipt=publication
    )
    gc_receipt = {
        "format": RESUME_CHECKPOINT_GC_FORMAT,
        "publication_receipt_sha256": canonical_json_sha256(publication),
        "snapshots_removed": True,
        "terminal_stage_removed": True,
    }
    state = append_checkpoint_gc(
        journal, state, checkpoint_gc_receipt=gc_receipt
    )
    state = append_cleanup_closed(
        journal,
        state,
        cleanup_receipt={
            "format": RESUME_CLEANUP_FORMAT,
            "checkpoint_gc_receipt_sha256": canonical_json_sha256(gc_receipt),
            "owned_state_removed": True,
            "snapshots_removed": True,
            "terminal_stage_removed": True,
            "official_outputs_retained": True,
        },
    )
    assert state.cleanup_closed is not None
    with pytest.raises(ResumableShardError, match="nothing may follow"):
        journal.append(
            "cleanup_closed",
            generation=state.generation,
            checkpoint_gc_entry_sha256=state.checkpoint_gc["entry_sha256"],
            cleanup_receipt_sha256=_sha("again"),
            owned_state_removed=True,
            snapshots_removed=True,
            terminal_stage_removed=True,
            official_outputs_retained=True,
        )


def test_changed_rebuilt_corpus_is_rejected(tmp_path: Path) -> None:
    plan = _plan()
    journal, state = _new_journal(tmp_path, plan)
    state = _append_one(journal, state, ordinal=0)
    with pytest.raises(ResumableShardError, match="population changed"):
        verify_completed_add_receipts(
            state,
            ordered_batch_sha256s=(_sha("changed"), plan.ordered_batch_sha256s[1]),
        )


def test_plan_rejects_non_one_to_one_add_extraction_budget() -> None:
    plan = _plan()
    values = plan.__dict__ if hasattr(plan, "__dict__") else {
        field: getattr(plan, field) for field in plan.__dataclass_fields__
    }
    values = dict(values)
    values["authorized_extraction_calls"] = plan.authorized_extraction_calls + 1
    with pytest.raises(ResumableShardError, match="one infer=True"):
        ResumePlan(**values)


def test_snapshot_path_cannot_overlap_working_state(tmp_path: Path) -> None:
    journal = AppendOnlyResumeJournal(tmp_path / "resume.jsonl", _plan())
    with pytest.raises(ResumableShardError, match="overlap"):
        journal.create(
            owned_state_path="state/working",
            snapshot_root_path="state/working/snapshots",
        )


def test_header_is_an_authenticated_empty_prefix_and_first_intent_rolls_back(
    tmp_path: Path,
) -> None:
    plan = _plan()
    journal, state = _new_journal(tmp_path, plan)
    assert state.resume_safe is True
    assert state.committed_prefix == state.sealed_prefix == 0
    seed = suffix_counter_seed(state)
    assert seed.remaining == plan.authorized_add_operations
    state = append_intent(
        journal, state, ordinal=0, session_sha256=_sha("session-zero")
    )
    restore = prefix_zero_restore_receipt(
        state, destination_state_dir=tmp_path / "state" / "working"
    )
    state = append_presend_rollback(
        journal, state, restore_receipt=restore
    )
    assert state.generation == 1
    assert state.resume_safe is True


def test_prefix_zero_rollback_rejects_dirty_factory_state(tmp_path: Path) -> None:
    plan = _plan()
    journal, state = _new_journal(tmp_path, plan)
    state = append_intent(
        journal, state, ordinal=0, session_sha256=_sha("session-zero")
    )
    _fake_owned_state(tmp_path)
    with pytest.raises(ResumableShardError, match="requires verified absence"):
        prefix_zero_restore_receipt(
            state, destination_state_dir=tmp_path / "state" / "working"
        )


def test_atomic_send_record_survives_torn_jsonl_and_stays_ambiguous(
    tmp_path: Path,
) -> None:
    journal, state = _new_journal(tmp_path, _plan())
    state = append_intent(
        journal, state, ordinal=0, session_sha256=_sha("session")
    )
    state = append_send_attempt(
        journal, state, request_sha256=_sha("request")
    )
    journal.path.write_bytes(journal.path.read_bytes()[:-13])
    recovered = journal.replay()
    assert recovered.pending_send_attempt is not None
    assert recovered.externally_ambiguous is True
    with pytest.raises(ResumeAmbiguityError):
        recovered.require_resumable()
    evidence = list((journal.path.with_name(journal.path.name + ".records")).glob(
        "projection-recovery-*.bin"
    ))
    assert len(evidence) == 1


def test_published_checkpoint_is_authority_before_journal_ack(tmp_path: Path) -> None:
    plan = _plan(adds=1)
    journal, state = _new_journal(tmp_path, plan)
    state = _append_one(journal, state, ordinal=0)
    snapshot = _snapshot(tmp_path, journal, state, _fake_owned_state(tmp_path))
    assert state.externally_ambiguous is True
    reconciled = reconcile_published_checkpoint(journal, state)
    assert reconciled.committed_prefix == reconciled.sealed_prefix == 1
    assert reconciled.resume_safe is True
    assert (
        reconciled.latest_prefix_seal["snapshot_receipt_sha256"]
        == snapshot["snapshot_receipt_sha256"]
    )


def test_commit_without_published_checkpoint_cannot_be_reconciled(
    tmp_path: Path,
) -> None:
    journal, state = _new_journal(tmp_path, _plan())
    state = _append_one(journal, state, ordinal=0)
    with pytest.raises(ResumeAmbiguityError, match="no valid published"):
        reconcile_published_checkpoint(journal, state)


def test_duplicate_returned_ids_preserve_order_and_dedupe_only_ledger_key(
    tmp_path: Path,
) -> None:
    journal, state = _new_journal(tmp_path, _plan())
    state = _append_one(
        journal,
        state,
        ordinal=0,
        ids=("same", "same", "other", "same"),
    )
    commit = state.commits[0]
    assert commit["returned_memory_ids"] == ["same", "same", "other", "same"]
    material = rehydration_material(state)
    assert material["ordered_response_ids_by_add"] == [
        ["same", "same", "other", "same"]
    ]
    assert [row["memory_id"] for row in material["ledger_projection"]] == [
        "other",
        "same",
    ]


def test_irregular_batches_reconstruct_exact_ten_message_deque(tmp_path: Path) -> None:
    plan = _plan(adds=8)
    journal, state = _new_journal(tmp_path, plan)
    counts = (2, 1, 2, 2, 1, 2, 1, 2)
    expanded: list[dict[str, object]] = []
    for ordinal, count in enumerate(counts):
        state = _append_one(
            journal, state, ordinal=ordinal, turn_count=count
        )
        expanded.extend([_source_ref(ordinal, turn_count=count)] * count)
    material = rehydration_material(state)
    assert material["request_window_deque"] == expanded[-10:]
    assert len(material["request_window_deque"]) == 10
    assert material["adapter_stats"]["add_calls"] == 8


def test_full_prefix_seed_denies_further_sends_but_remains_searchable(
    tmp_path: Path,
) -> None:
    plan = _plan(adds=1)
    journal, state = _new_journal(tmp_path, plan)
    state = _append_one(journal, state, ordinal=0)
    state = _seal(tmp_path, journal, state, _fake_owned_state(tmp_path))
    seed = suffix_counter_seed(state)
    assert seed.remaining == 0
    assert seed.zero_remaining_send_denied is True
    assert seed.http_completed == seed.logical_completed == 1
    with pytest.raises(ResumableShardError):
        append_intent(journal, state, ordinal=1, session_sha256=_sha("extra"))


def test_event_schema_rejects_unknown_fields_even_with_valid_digest(
    tmp_path: Path,
) -> None:
    journal, state = _new_journal(tmp_path, _plan())
    state = append_intent(
        journal, state, ordinal=0, session_sha256=_sha("session")
    )
    rows = [dict(row) for row in state.entries]
    mutated = dict(rows[-1])
    mutated["unexpected"] = "field"
    body = dict(mutated)
    body.pop("entry_sha256")
    mutated["entry_sha256"] = canonical_json_sha256(body)
    rows[-1] = mutated
    with pytest.raises(ResumableShardError, match="fields mismatch"):
        replay_journal(rows, expected_plan=state.plan)


@pytest.mark.parametrize(
    "unsafe",
    ["../escape", "C:drive-relative", "safe\\backslash", "CON/file", "x:y"],
)
def test_windows_unsafe_relative_paths_are_rejected(
    tmp_path: Path, unsafe: str
) -> None:
    journal = AppendOnlyResumeJournal(tmp_path / "resume.jsonl", _plan())
    with pytest.raises(ResumableShardError):
        journal.create(
            owned_state_path=unsafe,
            snapshot_root_path="state/snapshots",
        )


def test_state_manifest_hashes_empty_directories(tmp_path: Path) -> None:
    state = _fake_owned_state(tmp_path)
    before = state_tree_receipt(state)
    (state / "new-empty-directory").mkdir()
    after = state_tree_receipt(state)
    assert before["snapshot_manifest_sha256"] != after["snapshot_manifest_sha256"]
    assert after["manifest"]["new-empty-directory"] == {"type": "directory"}


def test_state_manifest_rejects_hardlinked_files(tmp_path: Path) -> None:
    state = _fake_owned_state(tmp_path)
    try:
        os.link(state / "history.sqlite", state / "history-copy.sqlite")
    except OSError as exc:  # pragma: no cover - filesystem capability varies.
        pytest.skip(f"hardlinks unavailable: {exc}")
    with pytest.raises(ResumableShardError, match="hard-linked"):
        state_tree_receipt(state)


def test_snapshot_root_marker_tamper_blocks_checkpoint(tmp_path: Path) -> None:
    journal, state = _new_journal(tmp_path, _plan(adds=1))
    state = _append_one(journal, state, ordinal=0)
    owned = _fake_owned_state(tmp_path)
    marker = tmp_path / "state" / "snapshots" / ".memory-condense-mem0-snapshot-root"
    marker.write_bytes(b"tampered\n")
    with pytest.raises(ResumableShardError, match="marker"):
        _snapshot(tmp_path, journal, state, owned)


def test_terminal_stage_retains_full_checkpoint_until_outputs_publish(
    tmp_path: Path,
) -> None:
    plan = _plan(adds=1)
    journal, state = _new_journal(tmp_path, plan)
    state = _append_one(journal, state, ordinal=0)
    state = _seal(tmp_path, journal, state, _fake_owned_state(tmp_path))
    state = append_terminal_search(
        journal,
        state,
        terminal_stage_path="eval/stage.json",
        terminal_stage_sha256=_sha("stage"),
        terminal_result_sha256=_sha("result"),
        terminal_trace_sha256=_sha("trace"),
        completed_search_operations=1,
    )
    state = append_active_state_removed(
        journal,
        state,
        removal_receipt={
            "format": RESUME_STATE_REMOVAL_FORMAT,
            "plan_sha256": plan.sha256,
            "terminal_stage_sha256": _sha("stage"),
            "owned_state_path_sha256": _sha("state-path"),
            "owned_state_removed": True,
            "snapshots_retained": True,
        },
    )
    assert state.terminal_published is None
    assert state.checkpoint_gc is None
    assert state.latest_prefix_seal is not None
    with pytest.raises(ResumableShardError, match="checkpoint GC"):
        append_cleanup_closed(journal, state, cleanup_receipt={})


def test_sealed_json_repairs_only_an_exact_missing_sidecar(tmp_path: Path) -> None:
    target = tmp_path / "stage.json"
    receipt = publish_sealed_json(target, {"format": "stage", "value": 1})
    sidecar = target.with_name(target.name + ".sha256")
    sidecar.unlink()
    repaired = publish_sealed_json(target, {"format": "stage", "value": 1})
    assert repaired["created"] is False
    assert repaired["sha256"] == receipt["sha256"]
    assert read_sealed_json(target)["payload"]["value"] == 1
    sidecar.unlink()
    with pytest.raises(ResumableShardError, match="different"):
        publish_sealed_json(target, {"format": "stage", "value": 2})


def test_verified_snapshot_gc_rejects_extra_entry_then_is_restart_safe(
    tmp_path: Path,
) -> None:
    plan = _plan(adds=1)
    journal, state = _new_journal(tmp_path, plan)
    state = _append_one(journal, state, ordinal=0)
    state = _seal(tmp_path, journal, state, _fake_owned_state(tmp_path))
    state = append_terminal_search(
        journal,
        state,
        terminal_stage_path="eval/stage.json",
        terminal_stage_sha256=_sha("stage"),
        terminal_result_sha256=_sha("result"),
        terminal_trace_sha256=_sha("trace"),
        completed_search_operations=1,
    )
    state = append_active_state_removed(
        journal,
        state,
        removal_receipt={
            "format": RESUME_STATE_REMOVAL_FORMAT,
            "plan_sha256": plan.sha256,
            "terminal_stage_sha256": _sha("stage"),
            "owned_state_path_sha256": _sha("state-path"),
            "owned_state_removed": True,
            "snapshots_retained": True,
        },
    )
    publication = {
        "format": RESUME_PUBLICATION_FORMAT,
        "terminal_stage_sha256": _sha("stage"),
        "official_artifact_path": "eval/artifact.json",
        "official_artifact_sha256": _sha("artifact"),
        "official_trace_path": "eval/trace.json",
        "official_trace_sha256": _sha("trace"),
        "outputs_verified": True,
    }
    state = append_terminal_published(
        journal, state, publication_receipt=publication
    )
    root = tmp_path / "state" / "snapshots"
    (root / "unowned.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ResumableShardError, match="unowned"):
        remove_verified_snapshot_root(journal, state)
    (root / "unowned.txt").unlink()
    receipt = remove_verified_snapshot_root(journal, state)
    assert receipt["snapshots_removed"] is True
    assert root.exists() is False
    replay = remove_verified_snapshot_root(journal, state)
    assert replay["already_absent"] is True


def _terminal_lifecycle_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], list[str]]:
    lock = tmp_path / "mem0.lock"
    lock.write_bytes(b"locked-environment")
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    base = _plan(adds=1, searches=1)
    authorization = RetrievalStageAuthorization(
        sample_offset=base.sample_offset,
        sample_sha256=base.sample_sha256,
        raw_history_bundle_sha256=base.raw_history_bundle_sha256,
        question_ids=("q-0",),
        authorized_add_operations=1,
        authorized_extraction_calls=1,
        authorized_search_operations=1,
        source_validation_policy_sha256=base.source_validation_policy_sha256,
        source_implementation_sha256=base.source_implementation_sha256,
        source_environment_lock_sha256=base.source_environment_lock_sha256,
        mem0_policy_sha256=base.mem0_policy_sha256,
        mem0_tool_implementation_sha256=base.mem0_tool_implementation_sha256,
        mem0_environment_lock_sha256=lock_sha,
        mem0_stable_config_sha256=_sha("stable-config"),
        source_evaluation_identity={},
        mem0_stable_payload={},
        extraction_model_identity={},
        extraction_model_identity_sha256=_sha("extraction-identity"),
        embedder_model_identity={},
        embedder_model_identity_sha256=_sha("embedder-identity"),
        mem0_provider_retries=0,
    )
    values = {
        field: getattr(base, field) for field in base.__dataclass_fields__
    }
    values["authorization_sha256"] = canonical_json_sha256(
        asdict(authorization)
    )
    values["mem0_environment_lock_sha256"] = lock_sha
    plan = ResumePlan(**values)
    journal, state = _new_journal(tmp_path, plan)
    state = _append_one(journal, state, ordinal=0)
    state = _seal(tmp_path, journal, state, _fake_owned_state(tmp_path))

    ref = SimpleNamespace(**{**_source_ref(0), "roles": ("user",)})
    batch = SimpleNamespace(ref=ref, messages=(("user", "sealed"),))
    monkeypatch.setattr(
        resumable_runner,
        "build_adapter_prepared_corpus",
        lambda _shard: SimpleNamespace(batches=(batch,)),
    )
    monkeypatch.setattr(
        resumable_runner, "_validate_retrieval_authorization", lambda *a, **k: None
    )
    monkeypatch.setattr(
        resumable_runner, "_validate_plan_against_inputs", lambda *a, **k: None
    )
    search_calls: list[str] = []

    def fake_search(**kwargs: object) -> dict[str, object]:
        state_path = Path(kwargs["state_path"])
        assert state_path.is_dir()
        search_calls.append("search")
        return {"fake": True}

    monkeypatch.setattr(
        resumable_runner, "_perform_terminal_search", fake_search
    )

    def fake_stage(**kwargs: object) -> dict[str, object]:
        replay = kwargs["state"]
        seal = replay.latest_prefix_seal
        assert seal is not None
        write_usage = dict(seal["write_usage_attestation"])
        write_usage_sha256 = seal["write_usage_attestation_sha256"]
        transport_closure_sha256 = seal[
            "transport_closure_receipt_sha256"
        ]
        result = {
            "format": resumable_runner.RESUMABLE_TERMINAL_RESULT_FORMAT,
            "sample_sha256": plan.sample_sha256,
            "resumable_closure": {
                "checkpoint_authority_sha256": (
                    replay.checkpoint_authority_sha256
                ),
                "write_usage_attestation_sha256": write_usage_sha256,
            },
            "write_usage_attestation": write_usage,
        }
        trace = {
            "format": resumable_runner.RESUMABLE_TERMINAL_TRACE_FORMAT,
            "checkpoint_authority_sha256": replay.checkpoint_authority_sha256,
            "events": [],
            "transport_closure_receipt_sha256": (
                transport_closure_sha256
            ),
            "write_usage_attestation_sha256": write_usage_sha256,
        }
        return {
            "format": resumable_runner.RESUME_TERMINAL_FORMAT,
            "plan_sha256": plan.sha256,
            "authorization_sha256": plan.authorization_sha256,
            "committed_prefix": 1,
            "full_checkpoint_authority_sha256": (
                replay.checkpoint_authority_sha256
            ),
            "completed_search_operations": 1,
            "extraction_calls_closed": True,
            "provider_retries": 0,
            "transport_closure_receipt_sha256": (
                transport_closure_sha256
            ),
            "write_usage_attestation_sha256": write_usage_sha256,
            "terminal_result_sha256": canonical_json_sha256(result),
            "terminal_trace_sha256": canonical_json_sha256(trace),
            "result": result,
            "trace": trace,
        }

    monkeypatch.setattr(resumable_runner, "_terminal_stage_payload", fake_stage)
    monkeypatch.setattr(
        resumable_runner,
        "_official_terminal_payloads",
        lambda **_kwargs: (
            b'{"kind":"artifact"}\n',
            b'{"kind":"trace"}\n',
            {"kind": "artifact"},
            {"kind": "trace"},
        ),
    )
    shard = SimpleNamespace(
        sample_offset=0,
        sample_sha256=plan.sample_sha256,
        raw_history_bundle_sha256=plan.raw_history_bundle_sha256,
    )
    kwargs: dict[str, object] = {
        "shard": shard,
        "authorization": authorization,
        "plan": plan,
        "journal_path": journal.path,
        "owned_state_relative": "state/working",
        "snapshot_root_relative": "state/snapshots",
        "terminal_stage_relative": "eval/stage.json",
        "artifact_relative": "eval/artifact.json",
        "trace_relative": "eval/trace.json",
        "mem0_environment_lock_path": lock,
    }
    return kwargs, search_calls


def test_terminal_lifecycle_publishes_then_replays_without_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs, search_calls = _terminal_lifecycle_fixture(tmp_path, monkeypatch)
    first = resumable_runner.run_resumable_terminal_stage(**kwargs)
    assert first.action == "terminal_published_and_cleaned"
    assert search_calls == ["search"]
    assert not (tmp_path / "state" / "working").exists()
    assert not (tmp_path / "state" / "snapshots").exists()
    assert not (tmp_path / "eval" / "stage.json").exists()
    assert (tmp_path / "eval" / "artifact.json").is_file()
    second = resumable_runner.run_resumable_terminal_stage(**kwargs)
    assert second.artifact_sha256 == first.artifact_sha256
    assert second.journal_tail_sha256 == first.journal_tail_sha256
    assert search_calls == ["search"]


@pytest.mark.parametrize(
    "boundary",
    [
        "append_terminal_search",
        "append_active_state_removed",
        "append_terminal_published",
        "append_checkpoint_gc",
        "append_cleanup_closed",
    ],
)
def test_terminal_lifecycle_recovers_after_each_durable_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    kwargs, search_calls = _terminal_lifecycle_fixture(tmp_path, monkeypatch)
    original = getattr(resumable_runner, boundary)
    failed = False

    def fail_once(*args: object, **call_kwargs: object) -> object:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError(f"crash at {boundary}")
        return original(*args, **call_kwargs)

    monkeypatch.setattr(resumable_runner, boundary, fail_once)
    with pytest.raises(RuntimeError, match="crash at"):
        resumable_runner.run_resumable_terminal_stage(**kwargs)
    monkeypatch.setattr(resumable_runner, boundary, original)
    result = resumable_runner.run_resumable_terminal_stage(**kwargs)
    assert result.action == "terminal_published_and_cleaned"
    assert search_calls == ["search"]
    assert not (tmp_path / "state" / "working").exists()
    assert not (tmp_path / "state" / "snapshots").exists()
    assert not (tmp_path / "eval" / "stage.json").exists()
