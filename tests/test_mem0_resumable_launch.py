from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.mem0_eval import production_binding, resumable_cli, resumable_launch
from tools.mem0_eval.preflight import SourceValidationPlan
from tools.mem0_eval.resumable import (
    AppendOnlyResumeJournal,
    JournalLease,
    ResumePlan,
    ResumeJournalLocked,
    ResumableShardError,
    append_intent,
    deterministic_user_scope,
    publish_sealed_json,
    read_sealed_json,
    _path_identity_sha256,
)
from tools.mem0_eval.resumable_launch import (
    LOCKED_ADD_COUNTS,
    LOCKED_SAMPLE_OFFSETS,
    MANIFEST_NAME,
    PREFLIGHT_NAME,
    LockedLaunchContext,
    LockedLaunchInputs,
    Mem0ResumableLaunchError,
    ShardLaunchBinding,
    WRITE_METERING_MISSING_FIELDS,
    build_preflight_payload,
    materialize_launch,
    replay_launch,
    run_locked_live_segment,
)
from tools.mem0_eval.resumable_runner import (
    RESUMABLE_LIVE_LAUNCH_AUTHORITY_FORMAT,
    ResumableSegmentResult,
    _OneUseSegmentAuthorizationIssuer,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def launch_context() -> LockedLaunchContext:
    policy_sha = _sha("policy-v3")
    tool_sha = _sha("tool-v3")
    lock_sha = _sha("mem0-lock")
    plan = SourceValidationPlan(
        dataset_sha256=_sha("dataset"),
        split_manifest_sha256=_sha("split"),
        policy_manifest_sha256=_sha("source-policy"),
        implementation_sha256=_sha("source-code"),
        environment_lock_sha256=_sha("source-lock"),
        sample_offsets=LOCKED_SAMPLE_OFFSETS,
        target_tokens=1_000_000,
        questions_per_shard=10,
        evaluation_identity={
            "provider_retries": 0,
            "stress_context_tokens": 1_000_000,
            "stress_questions": 10,
            "stress_question_offset": 0,
            "max_samples": 1,
            "min_target_questions": 100,
            "sample_offsets": list(LOCKED_SAMPLE_OFFSETS),
        },
    )
    bindings = []
    for offset, add_count in zip(
        LOCKED_SAMPLE_OFFSETS, LOCKED_ADD_COUNTS, strict=True
    ):
        authorization_sha = _sha(f"authorization-{offset}")
        resume_plan = ResumePlan(
            authorization_sha256=authorization_sha,
            mem0_policy_sha256=policy_sha,
            source_validation_policy_sha256=plan.policy_manifest_sha256,
            source_implementation_sha256=plan.implementation_sha256,
            source_environment_lock_sha256=plan.environment_lock_sha256,
            mem0_tool_implementation_sha256=tool_sha,
            mem0_environment_lock_sha256=lock_sha,
            sample_offset=offset,
            sample_sha256=_sha(f"sample-{offset}"),
            raw_history_bundle_sha256=_sha(f"raw-{offset}"),
            ordered_batch_sha256s=(_sha(f"batch-{offset}"),) * add_count,
            authorized_add_operations=add_count,
            authorized_extraction_calls=add_count,
            authorized_search_operations=10,
            user_scope=deterministic_user_scope(authorization_sha),
        )
        bindings.append(
            ShardLaunchBinding(
                sample_offset=offset,
                sample_id=f"stress-{offset:03d}",
                sample_sha256=resume_plan.sample_sha256,
                raw_history_bundle_sha256=(
                    resume_plan.raw_history_bundle_sha256
                ),
                question_ids=tuple(
                    f"question-{offset:03d}-{index}" for index in range(10)
                ),
                authorization_sha256=authorization_sha,
                plan=resume_plan,
            )
        )
    return LockedLaunchContext(
        source_plan=plan,
        mem0_policy_sha256=policy_sha,
        mem0_tool_implementation_sha256=tool_sha,
        mem0_environment_lock_sha256=lock_sha,
        shards=tuple(bindings),
    )


def _sealed_preflight(
    tmp_path: Path, context: LockedLaunchContext, payload: dict | None = None
) -> tuple[Path, str]:
    path = tmp_path / PREFLIGHT_NAME
    receipt = publish_sealed_json(
        path, payload if payload is not None else build_preflight_payload(context)
    )
    return path, receipt["sha256"]


def _materialized(
    tmp_path: Path, context: LockedLaunchContext
) -> tuple[Path, str, Path, str]:
    preflight_path, preflight_sha = _sealed_preflight(tmp_path, context)
    root = tmp_path / "run"
    materialize_launch(
        context=context,
        preflight_path=preflight_path,
        expected_preflight_sha256=preflight_sha,
        run_root=root,
    )
    launch = read_sealed_json(root / MANIFEST_NAME)
    return root / PREFLIGHT_NAME, preflight_sha, root, launch["sha256"]


def test_preflight_binds_full100_common_parent_and_zero_call_contract(
    launch_context: LockedLaunchContext,
) -> None:
    from tools.mem0_eval.common_parent_contract import (
        COMPARISON_SEMANTICS,
        EXACT_ACCOUNTING,
    )

    payload = build_preflight_payload(launch_context)
    assert payload["source"]["sample_offsets"] == list(LOCKED_SAMPLE_OFFSETS)
    assert payload["source"]["target_tokens_per_shard"] == 1_000_000
    assert payload["population"] == {
        "questions": 100,
        "question_ids_sha256": payload["population"]["question_ids_sha256"],
        "add_operations": 24_923,
        "logical_extraction_calls": 24_923,
        "search_operations": 100,
    }
    answer = payload["common_parent_request_budget"]["answer"]
    judge = payload["common_parent_request_budget"]["judge"]
    assert payload["common_parent_request_budget"]["comparison_semantics"] == (
        COMPARISON_SEMANTICS
    )
    assert payload["common_parent_request_budget"][
        "exact_accounting_sha256"
    ] == resumable_launch.canonical_json_sha256(EXACT_ACCOUNTING)
    assert answer["max_prompt_tokens"] == 7_232
    assert answer["output_token_reserve"] == 768
    assert answer["complete_request_token_cap"] == 8_000
    assert answer["model"] == EXACT_ACCOUNTING["responder_model"]
    assert judge["max_prompt_tokens"] == 8_000
    assert judge["output_token_reserve"] == 1_024
    assert judge["complete_request_token_cap"] == 9_024
    assert judge["model"] == EXACT_ACCOUNTING["judge_model"]
    assert payload["provider_call_authorization"] == {
        **payload["provider_call_authorization"],
        "authorization_granted": False,
        "physical_provider_calls_performed": 0,
        "prospective_hard_call_ceiling": 25_123,
        "sdk_retries": 0,
    }
    assert payload["retained_transformer_token_state_bytes"] == 0
    assert payload["gold_handling"] == {
        "references_loaded_for_source_validation": True,
        "references_persisted_in_launch_artifacts": False,
        "references_exposed_to_provider": False,
    }
    assert len({row["namespace"] for row in payload["shards"]}) == 10
    assert all(
        row["cross_namespace_reads_authorized"] is False
        for row in payload["shards"]
    )


@pytest.mark.parametrize(
    "target,field,replacement",
    [
        (
            "tools.mem0_eval.common_parent_contract",
            "EXACT_ACCOUNTING",
            {"answer_max_prompt_tokens": 7_233},
        ),
        (
            "tools.mem0_eval.common_parent_contract",
            "COMPARISON_SEMANTICS",
            "standalone",
        ),
    ],
    ids=("authoritative-accounting-drift", "comparison-semantics-drift"),
)
def test_common_parent_authority_drift_is_rejected(
    launch_context: LockedLaunchContext,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    field: str,
    replacement: object,
) -> None:
    module = __import__(target, fromlist=[field])
    if field == "EXACT_ACCOUNTING":
        replacement = {**getattr(module, field), **replacement}
    monkeypatch.setattr(module, field, replacement)
    with pytest.raises(Mem0ResumableLaunchError, match="authoritative common-parent"):
        build_preflight_payload(launch_context)


def test_policy_header_rejects_v2_and_wrong_expected_hash(tmp_path: Path) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text('{"format":"memory-condense-mem0-comparison-policy-v2"}')
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    with pytest.raises(Mem0ResumableLaunchError, match="exact policy format"):
        resumable_launch._verify_policy_header_and_digest(policy, digest)
    policy.write_text('{"format":"memory-condense-mem0-comparison-policy-v3"}')
    with pytest.raises(Mem0ResumableLaunchError, match="byte SHA-256"):
        resumable_launch._verify_policy_header_and_digest(policy, digest)


def test_unverified_direct_cli_runtime_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="bootstrap.py"):
        resumable_cli._require_bootstrap_envelope()


def test_network_enabled_bootstrap_receipt_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resumable_cli.sys, "flags", SimpleNamespace(isolated=True))
    values = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "MEM0_TELEMETRY": "false",
        "CUSTOM_TIKTOKEN_CACHE_DIR": "verified-cache",
        "TIKTOKEN_CACHE_DIR": "verified-cache",
        "MEM0_VERIFIED_BOOTSTRAP_SOURCE_SHA256": "1" * 64,
        "MEM0_VERIFIED_BOOTSTRAP_TOOL_SHA256": "2" * 64,
        "MEM0_VERIFIED_BOOTSTRAP_NETWORK_DENIED": "0",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match="receipt is absent"):
        resumable_cli._require_bootstrap_envelope()


def test_live_segment_rejects_provider_free_bootstrap_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resumable_cli.sys, "flags", SimpleNamespace(isolated=True))
    values = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "MEM0_TELEMETRY": "false",
        "CUSTOM_TIKTOKEN_CACHE_DIR": "verified-cache",
        "TIKTOKEN_CACHE_DIR": "verified-cache",
        "MEM0_VERIFIED_BOOTSTRAP_SOURCE_SHA256": "1" * 64,
        "MEM0_VERIFIED_BOOTSTRAP_TOOL_SHA256": "2" * 64,
        "MEM0_VERIFIED_BOOTSTRAP_NETWORK_DENIED": "1",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match="receipt is absent"):
        resumable_cli._require_bootstrap_envelope(allow_network=True)


def test_hashed_source_must_also_be_the_imported_source(tmp_path: Path) -> None:
    with pytest.raises(Mem0ResumableLaunchError, match="package is not the source"):
        resumable_launch._verify_source_import_origin(tmp_path)


def test_journal_lease_rejects_same_process_contention_and_releases_on_error(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "shard" / "resume.jsonl"
    first = JournalLease(journal)
    with pytest.raises(RuntimeError, match="crash window"):
        with first:
            assert first.held_for(journal) is True
            with pytest.raises(ResumeJournalLocked, match="already held"):
                with JournalLease(journal):
                    pytest.fail("contending lease entered")
            with pytest.raises(ResumeJournalLocked, match="re-entered"):
                first.__enter__()
            raise RuntimeError("crash window")
    with JournalLease(journal) as recovered:
        assert recovered.held_for(journal) is True


def test_journal_lease_rejects_precreated_link_without_touching_target(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "resume.jsonl"
    lock = tmp_path / "resume.jsonl.lock"
    target = tmp_path / "outside.txt"
    target.write_bytes(b"untouched")
    try:
        os.symlink(target, lock)
    except (OSError, NotImplementedError):
        pytest.skip("filesystem does not permit an unprivileged symlink")
    with pytest.raises(ResumableShardError, match="link/reparse"):
        with JournalLease(journal):
            pytest.fail("linked lock entered")
    assert target.read_bytes() == b"untouched"


def test_segment_issuer_requires_exact_budget_and_is_one_use(
    tmp_path: Path, launch_context: LockedLaunchContext
) -> None:
    plan = launch_context.shards[0].plan
    journal_path = tmp_path / "resume.jsonl"
    lease = JournalLease(journal_path)
    authority = {
        "format": RESUMABLE_LIVE_LAUNCH_AUTHORITY_FORMAT,
        "preflight_sha256": _sha("preflight"),
        "launch_manifest_sha256": _sha("manifest"),
        "shard_launch_sha256": _sha("shard"),
        "shard_launch_payload_sha256": _sha("shard-payload"),
        "plan_sha256": plan.sha256,
        "authorization_sha256": plan.authorization_sha256,
        "journal_path_sha256": _path_identity_sha256(journal_path),
        "sample_offset": plan.sample_offset,
        "namespace": plan.user_scope,
        "namespace_sha256": hashlib.sha256(
            plan.user_scope.encode("utf-8")
        ).hexdigest(),
        "mem0_policy_sha256": plan.mem0_policy_sha256,
        "mem0_tool_implementation_sha256": plan.mem0_tool_implementation_sha256,
        "mem0_environment_lock_sha256": plan.mem0_environment_lock_sha256,
        "retained_transformer_token_state_bytes": 0,
    }
    with lease:
        state = AppendOnlyResumeJournal(journal_path, plan).create(
            owned_state_path="owned-state", snapshot_root_path="snapshots"
        )
        issuer = _OneUseSegmentAuthorizationIssuer(
            plan=plan,
            journal_path=journal_path,
            lease=lease,
            live_launch_authority=authority,
        )
        with pytest.raises(ResumableShardError, match="exact next segment"):
            issuer.issue(state=state, authorized_provider_calls=255)
        grant = issuer.issue(state=state, authorized_provider_calls=256)
        with pytest.raises(ResumableShardError, match="one-use"):
            issuer.issue(state=state, authorized_provider_calls=256)
    with pytest.raises(ResumableShardError, match="lost its journal lease"):
        grant.consume(
            state=state, journal_path=journal_path, segment_adds=256
        )


def test_locked_live_segment_derives_and_consumes_one_sealed_grant_without_io(
    tmp_path: Path,
    launch_context: LockedLaunchContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_path, preflight_sha, root, manifest_sha = _materialized(
        tmp_path, launch_context
    )
    inputs = LockedLaunchInputs(
        benchmark_file=tmp_path / "benchmark.json",
        split_manifest=tmp_path / "split.json",
        source_policy_manifest=tmp_path / "source-policy.json",
        source_repository_root=tmp_path / "source",
        mem0_policy_manifest=tmp_path / "mem0-policy.json",
        expected_mem0_policy_sha256=launch_context.mem0_policy_sha256,
        mem0_environment_lock=tmp_path / "mem0.lock",
        tool_root=Path(resumable_launch.__file__).resolve().parent,
    )
    monkeypatch.setattr(
        resumable_launch,
        "load_locked_launch_context",
        lambda _inputs: launch_context,
    )
    monkeypatch.setattr(
        resumable_launch,
        "recheck_locked_launch_inputs",
        lambda _inputs, _context: None,
    )
    raw_shards = tuple(
        SimpleNamespace(sample_offset=offset) for offset in LOCKED_SAMPLE_OFFSETS
    )
    monkeypatch.setattr(
        resumable_launch,
        "build_raw_stress_shards",
        lambda **_kwargs: raw_shards,
    )

    class FakePolicy:
        def retrieval_authorization(self, shard):
            return SimpleNamespace(sample_offset=shard.sample_offset)

        def recheck(self) -> None:
            return None

    policy = FakePolicy()
    monkeypatch.setattr(
        resumable_launch,
        "load_mem0_comparison_policy",
        lambda *_args, **_kwargs: policy,
    )
    monkeypatch.setattr(
        resumable_launch,
        "_binding_from_policy",
        lambda _policy, shard: next(
            row
            for row in launch_context.shards
            if row.sample_offset == shard.sample_offset
        ),
    )
    observed: dict[str, object] = {}

    def fake_segment(**kwargs):
        state = kwargs["state"]
        receipt = kwargs["segment_authorization"].consume(
            state=state,
            journal_path=kwargs["journal_path"],
            segment_adds=256,
        )
        observed.update(receipt)
        with pytest.raises(ResumableShardError, match="already consumed"):
            kwargs["segment_authorization"].consume(
                state=state,
                journal_path=kwargs["journal_path"],
                segment_adds=256,
            )
        return ResumableSegmentResult(
            action="provider_free_test_double",
            prefix_before=0,
            prefix_after=0,
            segment_adds=0,
            checkpoint_authority_sha256=state.checkpoint_authority_sha256,
            journal_tail_sha256=state.entries[-1]["entry_sha256"],
            state_tree_sha256=None,
            receipt_sha256=_sha("test-result"),
        )

    monkeypatch.setattr(
        resumable_launch, "_run_resumable_ingest_segment_locked", fake_segment
    )
    result = run_locked_live_segment(
        inputs=inputs,
        preflight_path=preflight_path,
        expected_preflight_sha256=preflight_sha,
        launch_manifest_path=root / MANIFEST_NAME,
        expected_launch_manifest_sha256=manifest_sha,
        run_root=root,
        sample_offset=0,
        authorized_provider_calls=256,
    )
    assert result.action == "provider_free_test_double"
    assert observed["authorized_provider_calls"] == 256
    assert observed["namespace"] == launch_context.shards[0].plan.user_scope
    authority = observed["live_launch_authority"]
    assert authority["preflight_sha256"] == preflight_sha
    assert authority["launch_manifest_sha256"] == manifest_sha
    assert authority["sample_offset"] == 0
    assert authority["retained_transformer_token_state_bytes"] == 0


def test_materialize_and_replay_are_provider_free(
    tmp_path: Path,
    launch_context: LockedLaunchContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_transport(*_args, **_kwargs):
        raise AssertionError("provider transport constructed")

    monkeypatch.setattr(
        production_binding,
        "LiteLLMTerraExtractionTransport",
        forbidden_transport,
    )
    preflight_path, preflight_sha, root, launch_sha = _materialized(
        tmp_path, launch_context
    )
    replay = replay_launch(
        context=launch_context,
        preflight_path=preflight_path,
        expected_preflight_sha256=preflight_sha,
        launch_manifest_path=root / MANIFEST_NAME,
        expected_launch_manifest_sha256=launch_sha,
        run_root=root,
        dry_run=True,
    )
    assert replay["physical_provider_calls"] == 0
    assert replay["provider_call_authorization_granted"] is False
    assert replay["retained_transformer_token_state_bytes"] == 0
    assert {row["status"] for row in replay["shards"]} == {"not_started"}
    assert (root / PREFLIGHT_NAME).is_file()


def test_no_provider_cli_dry_run_writes_nothing(
    tmp_path: Path,
    launch_context: LockedLaunchContext,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        resumable_cli,
        "load_locked_launch_context",
        lambda _inputs: launch_context,
    )
    monkeypatch.setattr(
        resumable_cli, "_require_bootstrap_envelope", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        resumable_cli,
        "_require_verified_bootstrap",
        lambda _context, **_kwargs: None,
    )
    monkeypatch.setattr(
        resumable_cli,
        "recheck_locked_launch_inputs",
        lambda _inputs, _context: None,
    )
    monkeypatch.setattr(
        production_binding,
        "LiteLLMTerraExtractionTransport",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider transport constructed")
        ),
    )
    output = tmp_path / "must-not-exist.json"
    arguments = [
        "preflight",
        "--benchmark-file",
        str(tmp_path / "dataset.json"),
        "--split-manifest",
        str(tmp_path / "split.json"),
        "--source-policy-manifest",
        str(tmp_path / "source.json"),
        "--source-repository-root",
        str(tmp_path),
        "--mem0-policy-manifest",
        str(tmp_path / "mem0.json"),
        "--expected-mem0-policy-sha256",
        "0" * 64,
        "--mem0-environment-lock",
        str(tmp_path / "pixi.lock"),
        "--output",
        str(output),
        "--dry-run",
    ]
    assert resumable_cli.main(arguments) == 0
    result = __import__("json").loads(capsys.readouterr().out)
    assert result["physical_provider_calls"] == 0
    assert result["provider_call_authorization_granted"] is False
    assert not output.exists()


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda payload: payload["common_parent_request_budget"]["answer"].__setitem__(
                "max_prompt_tokens", 7_233
            ),
            "not the current reconstructed contract",
        ),
        (
            lambda payload: payload["cost_accounting"].__setitem__(
                "missing_write_metering_fields",
                list(WRITE_METERING_MISSING_FIELDS[:-1]),
            ),
            "not the current reconstructed contract",
        ),
    ],
    ids=("over-budget-answer-prompt", "incomplete-write-metering"),
)
def test_materialize_rejects_resealed_contract_tampering(
    tmp_path: Path,
    launch_context: LockedLaunchContext,
    mutate,
    match: str,
) -> None:
    payload = build_preflight_payload(launch_context)
    mutate(payload)
    path, digest = _sealed_preflight(tmp_path, launch_context, payload)
    with pytest.raises(Mem0ResumableLaunchError, match=match):
        materialize_launch(
            context=launch_context,
            preflight_path=path,
            expected_preflight_sha256=digest,
            run_root=tmp_path / "run",
        )


def test_duplicate_namespace_is_rejected(
    launch_context: LockedLaunchContext,
) -> None:
    first, second, *remaining = launch_context.shards
    leaked_plan = replace(second.plan, user_scope=first.plan.user_scope)
    leaked = replace(second, plan=leaked_plan)
    context = replace(
        launch_context, shards=(first, leaked, *remaining)
    )
    with pytest.raises(Mem0ResumableLaunchError, match="not unique"):
        build_preflight_payload(context)


def test_unique_namespace_not_derived_from_authorization_is_rejected(
    launch_context: LockedLaunchContext,
) -> None:
    first, *remaining = launch_context.shards
    forged_scope = deterministic_user_scope(_sha("forged-authorization"))
    assert forged_scope not in {row.plan.user_scope for row in launch_context.shards}
    context = replace(
        launch_context,
        shards=(
            replace(first, plan=replace(first.plan, user_scope=forged_scope)),
            *remaining,
        ),
    )
    with pytest.raises(Mem0ResumableLaunchError, match="not derived"):
        build_preflight_payload(context)


def test_stale_tool_hash_in_resume_plan_is_rejected(
    launch_context: LockedLaunchContext,
) -> None:
    first, *remaining = launch_context.shards
    stale_plan = replace(
        first.plan, mem0_tool_implementation_sha256=_sha("stale-tool")
    )
    context = replace(
        launch_context,
        shards=(replace(first, plan=stale_plan), *remaining),
    )
    with pytest.raises(Mem0ResumableLaunchError, match="tool_implementation"):
        build_preflight_payload(context)


def test_resume_journal_tampering_fails_closed(
    tmp_path: Path, launch_context: LockedLaunchContext
) -> None:
    preflight_path, preflight_sha, root, launch_sha = _materialized(
        tmp_path, launch_context
    )
    first = launch_context.shards[0]
    journal_path = root / "shard-000" / "resume.jsonl"
    journal = AppendOnlyResumeJournal(journal_path, first.plan)
    journal.create(
        owned_state_path="owned-state",
        snapshot_root_path="snapshots",
    )
    with journal_path.open("ab") as stream:
        stream.write(b"{}\n")
    with pytest.raises(Mem0ResumableLaunchError, match="strict replay"):
        replay_launch(
            context=launch_context,
            preflight_path=preflight_path,
            expected_preflight_sha256=preflight_sha,
            launch_manifest_path=root / MANIFEST_NAME,
            expected_launch_manifest_sha256=launch_sha,
            run_root=root,
            dry_run=True,
        )


def test_replay_rejects_projection_without_atomic_records(
    tmp_path: Path, launch_context: LockedLaunchContext
) -> None:
    preflight_path, preflight_sha, root, launch_sha = _materialized(
        tmp_path, launch_context
    )
    first = launch_context.shards[0]
    journal_path = root / "shard-000" / "resume.jsonl"
    AppendOnlyResumeJournal(journal_path, first.plan).create(
        owned_state_path="owned-state", snapshot_root_path="snapshots"
    )
    shutil.rmtree(journal_path.with_name(journal_path.name + ".records"))
    with pytest.raises(Mem0ResumableLaunchError, match="atomic record root"):
        replay_launch(
            context=launch_context,
            preflight_path=preflight_path,
            expected_preflight_sha256=preflight_sha,
            launch_manifest_path=root / MANIFEST_NAME,
            expected_launch_manifest_sha256=launch_sha,
            run_root=root,
            dry_run=True,
        )


def test_replay_reports_missing_projection_as_repair_required(
    tmp_path: Path, launch_context: LockedLaunchContext
) -> None:
    preflight_path, preflight_sha, root, launch_sha = _materialized(
        tmp_path, launch_context
    )
    first = launch_context.shards[0]
    journal_path = root / "shard-000" / "resume.jsonl"
    AppendOnlyResumeJournal(journal_path, first.plan).create(
        owned_state_path="owned-state", snapshot_root_path="snapshots"
    )
    journal_path.unlink()
    replay = replay_launch(
        context=launch_context,
        preflight_path=preflight_path,
        expected_preflight_sha256=preflight_sha,
        launch_manifest_path=root / MANIFEST_NAME,
        expected_launch_manifest_sha256=launch_sha,
        run_root=root,
        dry_run=True,
    )
    assert replay["shards"][0]["status"] == (
        "journal_projection_repair_required"
    )


def test_replay_rejects_orphan_state_without_journal(
    tmp_path: Path, launch_context: LockedLaunchContext
) -> None:
    preflight_path, preflight_sha, root, launch_sha = _materialized(
        tmp_path, launch_context
    )
    orphan = root / "shard-000" / "owned-state"
    orphan.mkdir()
    with pytest.raises(Mem0ResumableLaunchError, match="without an authoritative"):
        replay_launch(
            context=launch_context,
            preflight_path=preflight_path,
            expected_preflight_sha256=preflight_sha,
            launch_manifest_path=root / MANIFEST_NAME,
            expected_launch_manifest_sha256=launch_sha,
            run_root=root,
            dry_run=True,
        )


def test_replay_rejects_receipt_destination_inside_shard(
    tmp_path: Path, launch_context: LockedLaunchContext
) -> None:
    preflight_path, preflight_sha, root, launch_sha = _materialized(
        tmp_path, launch_context
    )
    unsafe = root / "shard-000" / "retrieval.json"
    with pytest.raises(Mem0ResumableLaunchError, match="fixed run-root"):
        replay_launch(
            context=launch_context,
            preflight_path=preflight_path,
            expected_preflight_sha256=preflight_sha,
            launch_manifest_path=root / MANIFEST_NAME,
            expected_launch_manifest_sha256=launch_sha,
            run_root=root,
            output_path=unsafe,
        )
    assert not unsafe.exists()


def test_replay_requires_the_materialized_root_preflight(
    tmp_path: Path, launch_context: LockedLaunchContext
) -> None:
    preflight_path, preflight_sha, root, launch_sha = _materialized(
        tmp_path, launch_context
    )
    preflight_path.unlink()
    preflight_path.with_name(preflight_path.name + ".sha256").unlink()
    with pytest.raises(Mem0ResumableLaunchError, match="valid sealed artifact"):
        replay_launch(
            context=launch_context,
            preflight_path=preflight_path,
            expected_preflight_sha256=preflight_sha,
            launch_manifest_path=root / MANIFEST_NAME,
            expected_launch_manifest_sha256=launch_sha,
            run_root=root,
            dry_run=True,
        )


def test_replay_rejects_transplanted_journal(
    tmp_path: Path, launch_context: LockedLaunchContext
) -> None:
    _preflight, preflight_sha, root, launch_sha = _materialized(
        tmp_path, launch_context
    )
    first = launch_context.shards[0]
    journal_path = root / "shard-000" / "resume.jsonl"
    AppendOnlyResumeJournal(journal_path, first.plan).create(
        owned_state_path="owned-state", snapshot_root_path="snapshots"
    )
    copied = tmp_path / "transplanted"
    shutil.copytree(root, copied)
    with pytest.raises(Mem0ResumableLaunchError, match="moved"):
        replay_launch(
            context=launch_context,
            preflight_path=copied / PREFLIGHT_NAME,
            expected_preflight_sha256=preflight_sha,
            launch_manifest_path=copied / MANIFEST_NAME,
            expected_launch_manifest_sha256=launch_sha,
            run_root=copied,
            dry_run=True,
        )


def test_replay_rejects_alternate_journal_state_paths(
    tmp_path: Path, launch_context: LockedLaunchContext
) -> None:
    preflight_path, preflight_sha, root, launch_sha = _materialized(
        tmp_path, launch_context
    )
    first = launch_context.shards[0]
    journal_path = root / "shard-000" / "resume.jsonl"
    AppendOnlyResumeJournal(journal_path, first.plan).create(
        owned_state_path="other-state", snapshot_root_path="other-snapshots"
    )
    with pytest.raises(Mem0ResumableLaunchError, match="state paths"):
        replay_launch(
            context=launch_context,
            preflight_path=preflight_path,
            expected_preflight_sha256=preflight_sha,
            launch_manifest_path=root / MANIFEST_NAME,
            expected_launch_manifest_sha256=launch_sha,
            run_root=root,
            dry_run=True,
        )


def test_replay_rejects_missing_snapshot_root(
    tmp_path: Path, launch_context: LockedLaunchContext
) -> None:
    preflight_path, preflight_sha, root, launch_sha = _materialized(
        tmp_path, launch_context
    )
    first = launch_context.shards[0]
    journal_path = root / "shard-000" / "resume.jsonl"
    AppendOnlyResumeJournal(journal_path, first.plan).create(
        owned_state_path="owned-state", snapshot_root_path="snapshots"
    )
    shutil.rmtree(root / "shard-000" / "snapshots")
    with pytest.raises(Mem0ResumableLaunchError, match="snapshot root"):
        replay_launch(
            context=launch_context,
            preflight_path=preflight_path,
            expected_preflight_sha256=preflight_sha,
            launch_manifest_path=root / MANIFEST_NAME,
            expected_launch_manifest_sha256=launch_sha,
            run_root=root,
            dry_run=True,
        )


def test_replay_reports_header_only_journal_as_next_segment_ready(
    tmp_path: Path, launch_context: LockedLaunchContext
) -> None:
    preflight_path, preflight_sha, root, launch_sha = _materialized(
        tmp_path, launch_context
    )
    first = launch_context.shards[0]
    AppendOnlyResumeJournal(
        root / "shard-000" / "resume.jsonl", first.plan
    ).create(owned_state_path="owned-state", snapshot_root_path="snapshots")
    replay = replay_launch(
        context=launch_context,
        preflight_path=preflight_path,
        expected_preflight_sha256=preflight_sha,
        launch_manifest_path=root / MANIFEST_NAME,
        expected_launch_manifest_sha256=launch_sha,
        run_root=root,
        dry_run=True,
    )
    assert replay["shards"][0]["status"] == "next_segment_ready"


def test_replay_reports_presend_intent_as_rollback_available(
    tmp_path: Path, launch_context: LockedLaunchContext
) -> None:
    preflight_path, preflight_sha, root, launch_sha = _materialized(
        tmp_path, launch_context
    )
    first = launch_context.shards[0]
    journal = AppendOnlyResumeJournal(
        root / "shard-000" / "resume.jsonl", first.plan
    )
    state = journal.create(
        owned_state_path="owned-state", snapshot_root_path="snapshots"
    )
    append_intent(
        journal, state, ordinal=0, session_sha256=_sha("pending-session")
    )
    replay = replay_launch(
        context=launch_context,
        preflight_path=preflight_path,
        expected_preflight_sha256=preflight_sha,
        launch_manifest_path=root / MANIFEST_NAME,
        expected_launch_manifest_sha256=launch_sha,
        run_root=root,
        dry_run=True,
    )
    assert replay["shards"][0]["status"] == "presend_rollback_available"


def test_replay_accepts_post_publication_pre_gc_ack_crash(
    tmp_path: Path,
    launch_context: LockedLaunchContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_path, preflight_sha, root, launch_sha = _materialized(
        tmp_path, launch_context
    )
    first = launch_context.shards[0]
    journal_path = root / "shard-000" / "resume.jsonl"
    journal = AppendOnlyResumeJournal(journal_path, first.plan)
    initial = journal.create(
        owned_state_path="owned-state", snapshot_root_path="snapshots"
    )
    shutil.rmtree(root / "shard-000" / "snapshots")
    artifact_path = root / "shard-000" / "retrieval.json"
    trace_path = root / "shard-000" / "retrieval.trace.json"
    artifact_path.write_bytes(b"sealed official artifact\n")
    trace_path.write_bytes(b"sealed official trace\n")
    published = {
        "official_artifact_path": "retrieval.json",
        "official_artifact_sha256": hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest(),
        "official_trace_path": "retrieval.trace.json",
        "official_trace_sha256": hashlib.sha256(
            trace_path.read_bytes()
        ).hexdigest(),
    }
    crash_state = SimpleNamespace(
        entries=initial.entries,
        plan=first.plan,
        terminal_search={"terminal_stage_sha256": _sha("terminal-stage")},
        terminal_published=published,
        checkpoint_gc=None,
        latest_prefix_seal=None,
        active_state_removed={"acknowledged": True},
        cleanup_closed=None,
        externally_ambiguous=False,
        requires_rollback=False,
        committed_prefix=first.plan.authorized_add_operations,
        sealed_prefix=first.plan.authorized_add_operations,
    )
    monkeypatch.setattr(
        resumable_launch, "replay_journal", lambda *_args, **_kwargs: crash_state
    )
    replay = replay_launch(
        context=launch_context,
        preflight_path=preflight_path,
        expected_preflight_sha256=preflight_sha,
        launch_manifest_path=root / MANIFEST_NAME,
        expected_launch_manifest_sha256=launch_sha,
        run_root=root,
        dry_run=True,
    )
    assert replay["shards"][0]["status"] == "terminal_gc_recovery_required"
