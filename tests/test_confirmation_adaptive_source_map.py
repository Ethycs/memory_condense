from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain.discourse import quote_sha256
from tests.test_matched_eval_locked_source_gate_adapter import _fixture
from tests.test_matched_eval_query_map_source_gate_adapter import (
    _map_plan,
    _map_plane,
    _plan,
)
from tools import confirmation_adaptive_source_map as subject
from tools.matched_eval.locked_source_gate_adapter import (
    LockedSourceHydrationInput,
    build_locked_source_gate_adapter,
    locked_activation_input_from_query_map_adapter,
    project_locked_lane_source_stream,
)
from tools.matched_eval.query_map_source_gate_adapter import (
    CONSOLIDATED_OBLIGATION_MODE,
    STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
    adapt_query_map_solver_v2,
)
from tools.matched_eval.source_history_mapper_live import (
    SourceMapperMaterialization,
)
from tools.matched_eval.source_history_fact_union import FactLane
from tools.run_locked_adaptive_source_map import source_gate_policy


class _MapperCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **request):
        self.calls += 1
        assert request["model"] == "codex_sdk/test-terra"
        assert request["max_tokens"] == 1024
        # Empty facts are a valid strict mapper result and keep this lifecycle
        # test independent of the exact synthetic source text.
        completion = json.dumps({"facts": []}, separators=(",", ":"))
        return SimpleNamespace(
            id=f"mapper-{self.calls}",
            model=request["model"],
            choices=(
                SimpleNamespace(
                    message=SimpleNamespace(content=completion),
                    finish_reason="stop",
                ),
            ),
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=3,
                total_tokens=103,
            ),
        )


class _MapperClient:
    def __init__(self) -> None:
        self.max_retries = 0
        self.completions = _MapperCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def _parents(tmp_path: Path):
    query_run, map_plan = _map_plan(tmp_path / "query-parent", _plan())
    query_adapter = adapt_query_map_solver_v2(
        query_run,
        map_plan,
        _map_plane(map_plan, ()),
        obligation_mode=CONSOLIDATED_OBLIGATION_MODE,
        state_chain_profile=STATE_CHAIN_DIRECT_AUTHORITY_PROFILE,
    )
    adapted = query_adapter.rows[0]
    assert adapted.activation is not None

    store_root = tmp_path / "store"
    store_root.mkdir()
    locked, _discarded_activation, artifacts = _fixture(store_root)
    guided = project_locked_lane_source_stream(
        FactLane.GUIDED,
        ("guided-only", "shared"),
        row_receipt=quote_sha256("confirmation-guided-row"),
        selected_ids=("guided-selection-0", "guided-selection-1"),
        artifact_sha256=quote_sha256("guided-artifact"),
    )
    locked = replace(
        locked,
        lane_streams=locked.lane_streams[:2] + (guided,),
    )
    # These two independent historical test helpers use different synthetic
    # snapshot seals.  A real ConfirmationSourceStreamsResult shares one; bind
    # the synthetic query plane to the store namespace before composing them.
    query_adapter = replace(
        query_adapter,
        snapshot_id=locked.namespace.snapshot_id,
    )
    adapted = query_adapter.rows[0]
    assert adapted.activation is not None
    packet = map_plan.rows[0].direct_plan_row.adapter.source.packet
    locked = replace(
        locked,
        question_id=packet.question_id,
        question_sha256=packet.question_sha256,
        dated_question=packet.dated_question,
        dated_question_sha256=packet.dated_question_sha256,
        source_packet_id=packet.packet_id,
    )
    activation = locked_activation_input_from_query_map_adapter(
        adapted,
        as_of_turn=0,
    )
    population = build_locked_source_gate_adapter(
        (locked,),
        (activation,),
        source_artifacts=artifacts,
        policy=source_gate_policy(1, 0, 1),
    )
    return population, query_adapter, store_root


def _preflight(tmp_path: Path):
    population, adapter, store_root = _parents(tmp_path)
    output = tmp_path / "output"
    preflight = subject.publish_confirmation_adaptive_source_map_preflight(
        population,
        adapter,
        output_root=output,
        model="codex_sdk/test-terra",
        gateway_url="https://controlled.invalid/v1",
        max_concurrency=1,
    )
    return population, adapter, store_root, output, preflight


def _approve(output: Path, preflight, calls: int):
    return subject.approve_confirmation_adaptive_source_map_release(
        preflight,
        output_root=output,
        expected_preflight_sha256=preflight.preflight_artifact.sha256,
        expected_work_manifest_sha256=preflight.work_manifest_artifact.sha256,
        approve_provider_release=True,
        authorized_provider_calls=calls,
    )


def test_full_lifecycle_is_store_free_after_preflight_and_revalidates_on_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population, adapter, store_root, output, preflight = _preflight(tmp_path)

    assert preflight.required_provider_calls == 2
    assert len(preflight.plan.hydration_batches) == 1
    assert preflight.plan.hydration_batches[0].projection()[
        "namespace_read_count"
    ] == 1
    assert preflight.preflight_artifact.payload["question_count"] == 1
    assert preflight.preflight_artifact.payload["provider_calls"] == 0
    assert not (output / subject.CHECKPOINT_DIR_NAME).exists()

    with pytest.raises(
        subject.ConfirmationAdaptiveSourceMapError,
        match="exact remaining calls",
    ):
        _approve(output, preflight, preflight.required_provider_calls - 1)

    release = _approve(output, preflight, preflight.required_provider_calls)
    client = _MapperClient()
    batch = subject.run_confirmation_adaptive_source_map_provider(
        preflight,
        output_root=output,
        expected_preflight_sha256=preflight.preflight_artifact.sha256,
        expected_work_manifest_sha256=preflight.work_manifest_artifact.sha256,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=preflight.required_provider_calls,
        client_factory=lambda _gateway, _env: client,
    )
    assert batch.usage.physical_calls == preflight.required_provider_calls
    assert batch.usage.checkpoint_hits == 0
    assert client.completions.calls == preflight.required_provider_calls

    def forbidden_store_read(_self):
        raise AssertionError("materialization opened a source store")

    with monkeypatch.context() as scoped:
        scoped.setattr(
            LockedSourceHydrationInput,
            "open_read_only_database",
            forbidden_store_read,
        )
        materialized = subject.materialize_confirmation_adaptive_source_map(
            preflight,
            output_root=output,
            expected_preflight_sha256=preflight.preflight_artifact.sha256,
            expected_work_manifest_sha256=preflight.work_manifest_artifact.sha256,
            expected_release_sha256=release.sha256,
        )
    assert materialized.materialization_artifact.payload[
        "store_reads_during_materialization"
    ] == 0
    assert all(
        type(row) is SourceMapperMaterialization
        for row in materialized.materializations
    )

    replayed = subject.replay_confirmation_adaptive_source_map(
        population,
        adapter,
        output_root=output,
        expected_preflight_sha256=preflight.preflight_artifact.sha256,
        expected_work_manifest_sha256=preflight.work_manifest_artifact.sha256,
        expected_release_sha256=release.sha256,
        expected_materialization_sha256=(
            materialized.materialization_artifact.sha256
        ),
    )
    assert type(replayed) is subject.VerifiedConfirmationAdaptiveSourceMapPlane
    assert replayed.replay_artifact.payload["stores_revalidated_during_replay"] is True
    assert replayed.materializations == materialized.materializations

    (store_root / "hnsw_index.bin").write_bytes(b"tampered-after-materialization")
    with pytest.raises(Exception, match="store index changed"):
        subject.replay_confirmation_adaptive_source_map(
            population,
            adapter,
            output_root=output,
            expected_preflight_sha256=preflight.preflight_artifact.sha256,
            expected_work_manifest_sha256=preflight.work_manifest_artifact.sha256,
            expected_release_sha256=release.sha256,
            expected_materialization_sha256=(
                materialized.materialization_artifact.sha256
            ),
        )


def test_partial_resume_and_checkpoint_corruption_fail_closed(tmp_path: Path) -> None:
    _population, _adapter, _store_root, output, preflight = _preflight(tmp_path)
    artifact = preflight.preflight_artifact
    checkpoint = output / subject.CHECKPOINT_DIR_NAME

    checkpoint.mkdir(parents=True)
    foreign = checkpoint / "foreign.json"
    foreign.write_text("{}", encoding="utf-8")
    with pytest.raises(
        subject.ConfirmationAdaptiveSourceMapError,
        match="foreign state",
    ):
        _approve(output, preflight, preflight.required_provider_calls)
    foreign.unlink()

    # Seed native full-population journals, then remove one complete pair to
    # emulate a safely resumable prior process (never a request-only retry).
    seed_client = _MapperClient()
    runtime = subject._runtime(
        preflight,
        artifact,
        checkpoint_dir=checkpoint,
        client=seed_client,
    )
    try:
        seeded = runtime.run()
    finally:
        runtime.close()
    assert seeded.usage.physical_calls == preflight.required_provider_calls

    responses = sorted(checkpoint.glob("*.response.json"))
    request = responses[0].with_name(
        responses[0].name.replace(".response.json", ".request.json")
    )
    response = responses[0]
    original_response = response.read_bytes()
    response.write_bytes(b"{}")
    with pytest.raises(Exception, match="canonical|receipt|fields|provenance"):
        _approve(output, preflight, 0)
    response.write_bytes(original_response)

    response.unlink()
    with pytest.raises(
        subject.ConfirmationAdaptiveSourceMapError,
        match="incomplete",
    ):
        _approve(output, preflight, 0)
    response.write_bytes(original_response)

    request.unlink()
    response.unlink()
    release = _approve(output, preflight, 1)
    assert release.payload["checkpoint_snapshot"][
        "authenticated_complete_count"
    ] == preflight.required_provider_calls - 1
    resume_client = _MapperClient()
    resumed = subject.run_confirmation_adaptive_source_map_provider(
        preflight,
        output_root=output,
        expected_preflight_sha256=preflight.preflight_artifact.sha256,
        expected_work_manifest_sha256=preflight.work_manifest_artifact.sha256,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=1,
        client_factory=lambda _gateway, _env: resume_client,
    )
    assert resumed.usage.physical_calls == 1
    assert resumed.usage.checkpoint_hits == preflight.required_provider_calls - 1
    assert resume_client.completions.calls == 1


def test_rejects_nonfrozen_policy_before_hydration(tmp_path: Path) -> None:
    population, adapter, _store_root = _parents(tmp_path)
    changed = replace(
        population,
        questions=tuple(
            replace(row, plan=replace(row.plan, policy=source_gate_policy(2, 0, 1)))
            for row in population.questions
        ),
    )

    with pytest.raises(
        subject.ConfirmationAdaptiveSourceMapError,
        match="source/query binding changed",
    ):
        subject.publish_confirmation_adaptive_source_map_preflight(
            changed,
            adapter,
            output_root=tmp_path / "changed-output",
            model="codex_sdk/test-terra",
            gateway_url="https://controlled.invalid/v1",
            max_concurrency=1,
        )
    assert not (tmp_path / "changed-output").exists()
