from __future__ import annotations

import json
import shutil
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import confirmation_typed_final as subject
from tools import confirmation_adaptive_source_map as base_stage
from tools import confirmation_adaptive_tail as adaptive_stage
from tools.matched_eval.artifacts import publish_sealed_json
from tools.matched_eval.contracts import identity_sha256
from tests import test_confirmation_source_streams as source_fixture
from tests.test_confirmation_adaptive_source_map import _MapperClient
from tests.test_confirmation_adaptive_tail import _Factory
from tests.test_run_locked_typed_memory_final_arm import _composition


class _Completions:
    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def create(self, **request):
        with self._lock:
            self.calls += 1
            self.requests.append(request)
            number = self.calls
        return SimpleNamespace(
            id=f"typed-final-{number}",
            model=request["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="{}"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=1,
                total_tokens=101,
            ),
        )


class _Client:
    def __init__(self) -> None:
        self.max_retries = 0
        self.chat = SimpleNamespace(completions=_Completions())
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _sealed_composition(tmp_path: Path, count: int):
    legacy = _composition().payload
    closure, _ = publish_sealed_json(
        tmp_path / subject.CLOSURE_INPUT_NAME,
        {
            "format": subject.CLOSURE_INPUT_FORMAT,
            "gold_loaded": False,
            "question_count": count,
        },
    )
    payload = {
        "closure_input_artifact_sha256": closure.sha256,
        "context_binding_identity_sha256": identity_sha256({"context": count}),
        "format": subject.COMPOSITION_ARTIFACT_FORMAT,
        "gold_loaded": False,
        "parent_adaptive_run_sha256": legacy["parent_adaptive_run_sha256"],
        "parent_map_run_sha256": legacy["parent_map_run_sha256"],
        "parent_source_materialization_sha256": legacy[
            "parent_source_materialization_sha256"
        ],
        "question_count": count,
        "questions": legacy["questions"][:count],
        "tail_materialization_sha256": legacy["tail_materialization_sha256"],
    }
    artifact, _ = publish_sealed_json(tmp_path / subject.COMPOSITION_NAME, payload)
    # Provider phases consume only sealed composition bytes.  The exact input
    # carrier is deliberately untouched here; full store replay is tested by
    # the integration case below.
    inputs = object.__new__(subject.ConfirmationTypedFinalInputs)
    return subject.ConfirmationTypedComposition(inputs, closure, artifact)


def _preflight(tmp_path: Path, count: int = 3):
    composition = _sealed_composition(tmp_path, count)
    preflight = subject.publish_confirmation_typed_final_preflight(
        composition,
        output_root=tmp_path,
        model="codex_sdk/test-terra",
        gateway_url="https://controlled.invalid/v1",
        max_concurrency=1,
    )
    return composition, preflight


def _full_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    count: int,
) -> subject.ConfirmationTypedFinalInputs:
    context, _query, map_plan, map_plane, _stream_root, streams = (
        source_fixture._materialized(
            tmp_path / "upstream", monkeypatch, count=count
        )
    )
    # The source-stream fixture intentionally uses the closure-test direct
    # parent, whose synthetic map snapshot differs from its real-store source
    # snapshot.  Production has one shared snapshot.  Align that test-only
    # adapter field before exercising the strict downstream join.
    adapter = replace(
        streams.query_map_adapter,
        snapshot_id=streams.base_population.questions[0].plan.parent.snapshot_id,
    )
    streams = replace(streams, query_map_adapter=adapter)
    base_root = tmp_path / "base"
    base_preflight = base_stage.publish_confirmation_adaptive_source_map_preflight(
        streams.base_population,
        streams.query_map_adapter,
        output_root=base_root,
        model="codex_sdk/test-terra",
        gateway_url="https://controlled.invalid/v1",
        max_concurrency=1,
    )
    base_release = base_stage.approve_confirmation_adaptive_source_map_release(
        base_preflight,
        output_root=base_root,
        expected_preflight_sha256=base_preflight.preflight_artifact.sha256,
        expected_work_manifest_sha256=base_preflight.work_manifest_artifact.sha256,
        approve_provider_release=True,
        authorized_provider_calls=base_preflight.required_provider_calls,
    )
    base_stage.run_confirmation_adaptive_source_map_provider(
        base_preflight,
        output_root=base_root,
        expected_preflight_sha256=base_preflight.preflight_artifact.sha256,
        expected_work_manifest_sha256=base_preflight.work_manifest_artifact.sha256,
        expected_release_sha256=base_release.sha256,
        enable_provider=True,
        authorized_provider_calls=base_preflight.required_provider_calls,
        client_factory=lambda _gateway, _env: _MapperClient(),
    )
    base_materialized = base_stage.materialize_confirmation_adaptive_source_map(
        base_preflight,
        output_root=base_root,
        expected_preflight_sha256=base_preflight.preflight_artifact.sha256,
        expected_work_manifest_sha256=base_preflight.work_manifest_artifact.sha256,
        expected_release_sha256=base_release.sha256,
    )
    base = base_stage.replay_confirmation_adaptive_source_map(
        streams.base_population,
        streams.query_map_adapter,
        output_root=base_root,
        expected_preflight_sha256=base_preflight.preflight_artifact.sha256,
        expected_work_manifest_sha256=base_preflight.work_manifest_artifact.sha256,
        expected_release_sha256=base_release.sha256,
        expected_materialization_sha256=base_materialized.materialization_artifact.sha256,
    )
    upstream = adaptive_stage.confirmation_adaptive_upstream(
        streams, base, map_plan, map_plane
    )

    solver_root = tmp_path / "solver"
    solver_plan = adaptive_stage.build_confirmation_adaptive_evidence_plan(upstream)
    solver_preflight = adaptive_stage.publish_confirmation_adaptive_evidence_preflight(
        solver_plan,
        output_root=solver_root,
        model="codex_sdk/test-terra",
        gateway_url="https://controlled.invalid/v1",
        max_concurrency=1,
    )
    solver_release = adaptive_stage.approve_confirmation_adaptive_release(
        solver_preflight,
        output_root=solver_root,
        expected_preflight_sha256=solver_preflight.artifact.sha256,
        approve_provider_release=True,
        authorized_provider_calls=solver_preflight.plan.required_calls,
    )
    adaptive_stage.run_confirmation_adaptive_provider(
        solver_preflight,
        output_root=solver_root,
        expected_preflight_sha256=solver_preflight.artifact.sha256,
        expected_release_sha256=solver_release.sha256,
        enable_provider=bool(solver_preflight.plan.required_calls),
        authorized_provider_calls=solver_preflight.plan.required_calls,
        client_factory=_Factory(lambda _request: "{}"),
    )
    solver_materialized = adaptive_stage.materialize_confirmation_adaptive_evidence(
        solver_preflight,
        output_root=solver_root,
        expected_preflight_sha256=solver_preflight.artifact.sha256,
        expected_release_sha256=solver_release.sha256,
    )
    adaptive = adaptive_stage.replay_confirmation_adaptive_evidence(
        solver_preflight,
        output_root=solver_root,
        expected_preflight_sha256=solver_preflight.artifact.sha256,
        expected_release_sha256=solver_release.sha256,
        expected_run_sha256=solver_materialized.run_artifact.sha256,
        expected_replay_sha256=solver_materialized.replay_artifact.sha256,
    )

    tail_root = tmp_path / "tail"
    tail_plan = adaptive_stage.build_confirmation_adaptive_tail_plan(upstream)
    tail_preflight = adaptive_stage.publish_confirmation_adaptive_tail_preflight(
        tail_plan,
        output_root=tail_root,
        model="codex_sdk/test-terra",
        gateway_url="https://controlled.invalid/v1",
        max_concurrency=1,
    )
    tail_release = adaptive_stage.approve_confirmation_adaptive_release(
        tail_preflight,
        output_root=tail_root,
        expected_preflight_sha256=tail_preflight.artifact.sha256,
        approve_provider_release=True,
        authorized_provider_calls=tail_preflight.plan.required_calls,
    )
    adaptive_stage.run_confirmation_adaptive_provider(
        tail_preflight,
        output_root=tail_root,
        expected_preflight_sha256=tail_preflight.artifact.sha256,
        expected_release_sha256=tail_release.sha256,
        enable_provider=bool(tail_preflight.plan.required_calls),
        authorized_provider_calls=tail_preflight.plan.required_calls,
        client_factory=_Factory(lambda _request: '{"facts":[]}'),
    )
    tail_materialized = adaptive_stage.materialize_confirmation_adaptive_tail(
        tail_preflight,
        output_root=tail_root,
        expected_preflight_sha256=tail_preflight.artifact.sha256,
        expected_release_sha256=tail_release.sha256,
    )
    tail = adaptive_stage.replay_confirmation_adaptive_tail(
        tail_preflight,
        output_root=tail_root,
        expected_preflight_sha256=tail_preflight.artifact.sha256,
        expected_release_sha256=tail_release.sha256,
        expected_run_sha256=tail_materialized.run_artifact.sha256,
        expected_replay_sha256=tail_materialized.replay_artifact.sha256,
    )
    return subject.ConfirmationTypedFinalInputs(context, adaptive, base, tail)


def test_arbitrary_n_compact_preflight_and_exact_parent_fallback(
    tmp_path: Path,
) -> None:
    composition, preflight = _preflight(tmp_path, count=3)
    assert preflight.required_provider_calls == 3
    assert preflight.artifact.payload["question_count"] == 3
    assert all(
        row["prompt_token_proxy"] + subject.OUTPUT_TOKEN_RESERVE <= 8_000
        for row in preflight.artifact.payload["physical_prompt_rows"]
    )
    serialized = json.dumps(preflight.artifact.payload["physical_prompt_rows"])
    assert '"namespace_id"' not in serialized
    assert '"source_id"' not in serialized

    with pytest.raises(subject.ConfirmationTypedFinalError, match="exact remaining"):
        subject.approve_confirmation_typed_final_release(
            preflight,
            output_root=tmp_path,
            expected_preflight_sha256=preflight.artifact.sha256,
            approve_provider_release=True,
            authorized_provider_calls=2,
        )
    release = subject.approve_confirmation_typed_final_release(
        preflight,
        output_root=tmp_path,
        expected_preflight_sha256=preflight.artifact.sha256,
        approve_provider_release=True,
        authorized_provider_calls=3,
    )
    client = _Client()
    execution = subject.run_confirmation_typed_final_provider(
        preflight,
        output_root=tmp_path,
        expected_preflight_sha256=preflight.artifact.sha256,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=3,
        client_factory=lambda _gateway, _env: client,
    )
    assert execution.physical_provider_calls == 3
    assert execution.checkpoint_hits == 0
    assert client.chat.completions.calls == 3
    materialized = subject.materialize_confirmation_typed_final(
        preflight,
        output_root=tmp_path,
        expected_preflight_sha256=preflight.artifact.sha256,
        expected_release_sha256=release.sha256,
    )
    expected = tuple(
        row["parent_prediction"]
        for row in composition.composition_artifact.payload["questions"]
    )
    assert materialized.predictions == expected
    assert materialized.run_artifact.payload[
        "invalid_completion_parent_fallback_count"
    ] == 3
    again = subject.materialize_confirmation_typed_final(
        preflight,
        output_root=tmp_path,
        expected_preflight_sha256=preflight.artifact.sha256,
        expected_release_sha256=release.sha256,
    )
    assert again.run_artifact.sha256 == materialized.run_artifact.sha256


def test_partial_resume_releases_only_remaining_call(tmp_path: Path) -> None:
    _composition_value, preflight = _preflight(tmp_path, count=3)
    prompts = tuple(
        tuple(dict(message) for message in row["messages"])
        for row in preflight.artifact.payload["physical_prompt_rows"]
    )
    seed_client = _Client()
    runtime = subject._runtime(  # noqa: SLF001 - seed native journals
        preflight.artifact,
        prompts,
        output_root=tmp_path,
        client=seed_client,
    )
    try:
        seeded = runtime.run()
    finally:
        runtime.close()
    missing = seeded.unique_records[-1].call_key_sha256
    (tmp_path / subject.CHECKPOINT_DIR_NAME / f"{missing}.request.json").unlink()
    (tmp_path / subject.CHECKPOINT_DIR_NAME / f"{missing}.response.json").unlink()

    release = subject.approve_confirmation_typed_final_release(
        preflight,
        output_root=tmp_path,
        expected_preflight_sha256=preflight.artifact.sha256,
        approve_provider_release=True,
        authorized_provider_calls=1,
    )
    client = _Client()
    execution = subject.run_confirmation_typed_final_provider(
        preflight,
        output_root=tmp_path,
        expected_preflight_sha256=preflight.artifact.sha256,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=1,
        client_factory=lambda _gateway, _env: client,
    )
    assert execution.physical_provider_calls == 1
    assert execution.checkpoint_hits == 2
    assert client.chat.completions.calls == 1


def test_release_survives_completed_journal_subset_before_resume(
    tmp_path: Path,
) -> None:
    """A no-clobber release remains valid after an interrupted invocation."""

    _composition_value, preflight = _preflight(tmp_path, count=3)
    prompts = tuple(
        tuple(dict(message) for message in row["messages"])
        for row in preflight.artifact.payload["physical_prompt_rows"]
    )
    release = subject.approve_confirmation_typed_final_release(
        preflight,
        output_root=tmp_path,
        expected_preflight_sha256=preflight.artifact.sha256,
        approve_provider_release=True,
        authorized_provider_calls=3,
    )

    # Model a process that completed and sealed its first physical call after
    # release, then stopped before it submitted the remainder. Call one member
    # of the exact full preflighted runtime, then model the
    # process stopping before the remaining members are submitted.
    first_client = _Client()
    runtime = subject._runtime(  # noqa: SLF001 - exact interruption fixture
        preflight.artifact,
        prompts,
        output_root=tmp_path,
        client=first_client,
    )
    try:
        first = runtime._provider_call(runtime._unique_order[0])  # noqa: SLF001
    finally:
        runtime.close()
        first_client.close()
    assert first.physical_call is True

    resumed_client = _Client()
    resumed = subject.run_confirmation_typed_final_provider(
        preflight,
        output_root=tmp_path,
        expected_preflight_sha256=preflight.artifact.sha256,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=2,
        client_factory=lambda _gateway, _env: resumed_client,
    )
    assert resumed.physical_provider_calls == 2
    assert resumed.checkpoint_hits == 1
    assert resumed_client.chat.completions.calls == 2
    assert resumed_client.close_calls == 1


def test_real_sqlite_composition_streams_one_namespace_index_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _full_inputs(tmp_path, monkeypatch, count=3)
    output = tmp_path / "typed"
    composition = subject.materialize_confirmation_typed_composition(
        inputs, output_root=output
    )
    payload = composition.composition_artifact.payload
    assert payload["question_count"] == 3
    assert payload["unique_namespace_count"] == 2
    assert payload["maximum_simultaneous_namespace_indexes"] == 1
    assert composition.closure_input_artifact.payload[
        "maximum_simultaneous_namespace_indexes"
    ] == 1
    assert [row["ordinal"] for row in payload["questions"]] == [0, 1, 2]
    assert all(
        row["local_audit"]["selection_completed_before_cross_method_dedup"]
        for row in payload["questions"]
    )
    assert all(
        row["provider_projection"]["provider_input"]["typed_evidence"][
            "format"
        ]
        == "memory-condense-typed-memory-final-arm-v1-compact-provider-evidence-v1"
        for row in payload["questions"]
    )
    replayed = subject.replay_confirmation_typed_composition(
        inputs,
        output_root=output,
        expected_closure_input_sha256=composition.closure_input_artifact.sha256,
        expected_composition_sha256=composition.composition_artifact.sha256,
    )
    assert replayed.composition_artifact.sha256 == composition.composition_artifact.sha256

    preflight = subject.publish_confirmation_typed_final_preflight(
        composition,
        output_root=output,
        model="codex_sdk/test-terra",
        gateway_url="https://controlled.invalid/v1",
        max_concurrency=1,
    )
    release = subject.approve_confirmation_typed_final_release(
        preflight,
        output_root=output,
        expected_preflight_sha256=preflight.artifact.sha256,
        approve_provider_release=True,
        authorized_provider_calls=3,
    )
    subject.run_confirmation_typed_final_provider(
        preflight,
        output_root=output,
        expected_preflight_sha256=preflight.artifact.sha256,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=3,
        client_factory=lambda _gateway, _env: _Client(),
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(
            type(inputs.context),
            "revalidate_store_bytes",
            lambda _self: (_ for _ in ()).throw(
                AssertionError("answer materialization opened a store")
            ),
        )
        materialized = subject.materialize_confirmation_typed_final(
            preflight,
            output_root=output,
            expected_preflight_sha256=preflight.artifact.sha256,
            expected_release_sha256=release.sha256,
        )
    verified = subject.replay_confirmation_typed_final(
        preflight,
        output_root=output,
        expected_closure_input_sha256=composition.closure_input_artifact.sha256,
        expected_composition_sha256=composition.composition_artifact.sha256,
        expected_preflight_sha256=preflight.artifact.sha256,
        expected_release_sha256=release.sha256,
        expected_run_sha256=materialized.run_artifact.sha256,
    )
    assert type(verified) is subject.VerifiedConfirmationTypedFinalPlane
    assert verified.predictions == materialized.predictions
    assert verified.replay_artifact.payload["stores_revalidated_during_replay"] is True

    first_store = next(iter(inputs.context.store_dirs_by_namespace.values()))
    (first_store / "hnsw_index.bin").write_bytes(b"tampered-after-composition")
    with pytest.raises(Exception, match="index changed"):
        subject.replay_confirmation_typed_composition(
            inputs,
            output_root=output,
            expected_closure_input_sha256=composition.closure_input_artifact.sha256,
            expected_composition_sha256=composition.composition_artifact.sha256,
        )


def test_request_only_foreign_and_release_schema_fail_closed(tmp_path: Path) -> None:
    _composition_value, preflight = _preflight(tmp_path, count=1)
    checkpoint = tmp_path / subject.CHECKPOINT_DIR_NAME
    checkpoint.mkdir()
    (checkpoint / "foreign.json").write_text("{}", encoding="utf-8")
    with pytest.raises(subject.ConfirmationTypedFinalError, match="foreign state"):
        subject.approve_confirmation_typed_final_release(
            preflight,
            output_root=tmp_path,
            expected_preflight_sha256=preflight.artifact.sha256,
            approve_provider_release=True,
            authorized_provider_calls=1,
        )
    (checkpoint / "foreign.json").unlink()
    key = "a" * 64
    (checkpoint / f"{key}.request.json").write_text("{}", encoding="utf-8")
    with pytest.raises(subject.ConfirmationTypedFinalError, match="incomplete"):
        subject.approve_confirmation_typed_final_release(
            preflight,
            output_root=tmp_path,
            expected_preflight_sha256=preflight.artifact.sha256,
            approve_provider_release=True,
            authorized_provider_calls=1,
        )
    (checkpoint / f"{key}.request.json").unlink()
    release = subject.approve_confirmation_typed_final_release(
        preflight,
        output_root=tmp_path,
        expected_preflight_sha256=preflight.artifact.sha256,
        approve_provider_release=True,
        authorized_provider_calls=1,
    )
    changed = dict(release.payload)
    changed["unexpected"] = True
    changed.pop("release_identity_sha256")
    changed["release_identity_sha256"] = identity_sha256(changed)
    replacement, _ = publish_sealed_json(
        tmp_path / "replacement-release.json", changed
    )
    # Direct replacement tests the exact-schema firewall even when resealed.
    shutil.copyfile(replacement.path, release.path)
    release.path.with_suffix(release.path.suffix + ".sha256").write_bytes(
        f"{replacement.sha256}  {release.path.name}\n".encode("ascii")
    )
    with pytest.raises(subject.ConfirmationTypedFinalError, match="hash or schema"):
        subject.run_confirmation_typed_final_provider(
            preflight,
            output_root=tmp_path,
            expected_preflight_sha256=preflight.artifact.sha256,
            expected_release_sha256=replacement.sha256,
            enable_provider=True,
            authorized_provider_calls=1,
            client_factory=lambda _gateway, _env: (_ for _ in ()).throw(
                AssertionError("client constructed before firewall")
            ),
        )
