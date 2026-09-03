from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_confirmation_protected_s0_plane import (
    _completion_fixture,
    _kwargs,
    _query_namespaces,
)
from tests.test_confirmation_s0_prompt_preflight import _build_fixture
from tests.test_matched_eval_query_expansion import (
    _FakePartitionSearch,
    _StructuredClient,
    _candidate,
    _valid_completion,
)
from tools import confirmation_protected_s0_plane as protected_s0
from tools import confirmation_query_payload_parent as confirmation_parent
from tools.confirmation_contracts import publish_sealed_json
from tools.matched_eval import live, query_payload_live
from tools.matched_eval.contracts import (
    MatchedEvalContractError,
    canonical_json_bytes,
    identity_sha256,
)
from tools.matched_eval.ledger import _validated_runtime_ledger
from tools.matched_eval.query_expansion import (
    RUNTIME_LEDGER_NAME,
    RUNTIME_LEDGER_REPLAY_NAME,
    RUN_NAME,
    RUN_REPLAY_NAME,
    FrozenSourceMembership,
    FrozenSourceNamespace,
    build_query_expansion_population,
    preflight_query_expansion,
    replay_query_expansion,
    run_query_expansion,
)
from tools.matched_eval.query_fact_adapter import build_query_fact_population
from tools.matched_eval.query_payload_live import build_query_payload_answer_plan
from tools.matched_eval.query_evidence_map_solver_v2_live import (
    build_evidence_map_plan,
)
from tools.matched_eval.renderer import V4_RENDERER_ID


def _protected(tmp_path: Path, *, count: int = 3):
    fixture = _build_fixture(
        tmp_path / "fixture",
        semantics=tuple(range(count)),
        id_prefix="query-parent",
        namespace_sizes=((1,) if count == 1 else (1, count - 1)),
    )
    prompt, completion, lifecycle_sha, release_sha, predictions = (
        _completion_fixture(fixture)
    )
    kwargs = _kwargs(fixture, prompt, completion, lifecycle_sha, release_sha)
    artifact, _created, plane = protected_s0.publish_protected_s0_answer_plane(
        fixture.root / "protected-s0.json", **kwargs
    )
    parent = confirmation_parent.materialize_verified_protected_s0_parent(
        protected_s0_plane_path=artifact.path,
        expected_protected_s0_plane_sha256=artifact.sha256,
        output_root=fixture.root / confirmation_parent.PARENT_DIR_NAME,
        **kwargs,
    )
    return fixture, kwargs, artifact, plane, parent, predictions


def _query_adapter(tmp_path: Path, source_plane):
    namespaces = {}
    for index, row in enumerate(source_plane.source_population.rows):
        namespace = FrozenSourceNamespace(
            snapshot_id=source_plane.source_population.snapshot.snapshot_id,
            combined_store_receipt_sha256=hashlib.sha256(
                f"query-store:{index}".encode()
            ).hexdigest(),
            sources=(
                FrozenSourceMembership(
                    source_id=f"unrelated-history::episode-{index}",
                    content_chunk_ids=(f"global-content-{index}",),
                    metadata_chunk_ids=(),
                    stream_sha256=hashlib.sha256(
                        f"query-stream:{index}".encode()
                    ).hexdigest(),
                ),
            ),
        )
        namespaces[row.packet.question_id] = namespace
    query_population = build_query_expansion_population(
        source_plane.source_population,
        namespaces_by_question=namespaces,
        include_s0_evidence=True,
    )
    output = tmp_path / "query-expansion"
    preflight = preflight_query_expansion(query_population, output_root=output)
    retrievers = {}
    for namespace in query_population.namespaces:
        membership = namespace.sources[0]
        chunk_id = membership.content_chunk_ids[0]
        retrievers[namespace.namespace_id] = _FakePartitionSearch(
            namespace,
            (
                _candidate(
                    chunk_id=chunk_id,
                    source_id=membership.source_id,
                    text="A globally retrieved exact memory fact.",
                    score=0.91,
                ),
            ),
        )
    run = run_query_expansion(
        query_population,
        output_root=output,
        retrievers_by_namespace=retrievers,
        enable_provider=True,
        authorized_provider_calls=(
            query_population.prompt_population.unique_prompt_count
        ),
        client=_StructuredClient(_valid_completion()),
        max_concurrency=1,
    )
    replay = replay_query_expansion(
        query_population,
        output_root=output,
        retrievers_by_namespace=retrievers,
        expected_run_sha256=run.run_artifact.sha256,
        max_concurrency=1,
    )
    adapter = build_query_fact_population(
        source_plane.source_population,
        query_preflight=preflight,
        query_run=run.run_artifact,
        expected_retrieval_sha256=source_plane.source_population.retrieval_sha256,
        expected_source_population_id=source_plane.source_population.population_id,
        expected_query_preflight_sha256=preflight.sha256,
        expected_query_run_sha256=run.run_artifact.sha256,
        expected_query_population_id=query_population.population_id,
        expected_query_prompt_population_sha256=(
            query_population.prompt_population.prompt_population_sha256
        ),
    )
    return query_population, adapter, preflight, run, replay, output


def test_authenticates_v4_completion_and_exposes_exact_live_parent(
    tmp_path: Path,
) -> None:
    _fixture, kwargs, _source, plane, parent, predictions = _protected(
        tmp_path, count=4
    )

    assert type(parent.exact_parent) is live.VerifiedS0V2AnswerPlane
    assert parent.exact_parent.renderer_id == V4_RENDERER_ID
    assert parent.exact_parent.matched_population_id == plane.source_population.population_id
    assert tuple(row.prediction for row in parent.exact_parent.rows) == predictions
    assert parent.run_artifact.sha256 == parent.replay_artifact.sha256
    identity, row_ids = _validated_runtime_ledger(
        parent.runtime_ledger_artifact.payload
    )
    assert identity == parent.runtime_ledger_artifact.payload[
        "ledger_identity_sha256"
    ]
    assert row_ids == tuple(row.runtime_row_id for row in parent.exact_parent.rows)
    assert parent.runtime_ledger_artifact.payload["total_provider_calls"] == 4
    assert parent.bridge_artifact.payload["physical_provider_calls"] == 0

    replayed = confirmation_parent.materialize_verified_protected_s0_parent(
        protected_s0_plane_path=parent.source_artifact.path,
        expected_protected_s0_plane_sha256=parent.source_artifact.sha256,
        output_root=parent.run_artifact.path.parent,
        **kwargs,
    )
    assert replayed.bridge_artifact.sha256 == parent.bridge_artifact.sha256
    assert replayed.exact_parent == parent.exact_parent


def test_resealed_but_unreconstructable_protected_parent_fails_closed(
    tmp_path: Path,
) -> None:
    fixture, kwargs, source, _plane, _parent, _predictions = _protected(tmp_path)
    changed = copy.deepcopy(source.payload)
    changed["ordered_rows"][0]["prediction"] = "invented"
    changed_source, _created = publish_sealed_json(
        fixture.root / "changed-protected-s0.json", changed
    )

    with pytest.raises(
        confirmation_parent.ConfirmationQueryPayloadError,
        match="authenticated reconstruction",
    ):
        confirmation_parent.materialize_verified_protected_s0_parent(
            protected_s0_plane_path=changed_source.path,
            expected_protected_s0_plane_sha256=changed_source.sha256,
            output_root=fixture.root / "changed-parent",
            **kwargs,
        )


def test_query_payload_plan_accepts_exact_registered_v4_parent(
    tmp_path: Path,
) -> None:
    fixture, _kwargs_value, _source, plane, parent, _predictions = _protected(
        tmp_path
    )
    query_population, adapter, _preflight, _run, _replay, _output = (
        _query_adapter(fixture.root, plane)
    )

    plan = build_query_payload_answer_plan(adapter, parent.exact_parent)

    assert plan.parent_plane is parent.exact_parent
    assert plan.adapter_population is adapter
    assert 0 <= plan.required_calls <= query_population.source_population.question_count
    assert plan.parent_plane.renderer_id == V4_RENDERER_ID


def test_query_payload_plan_rejects_renderer_mismatch_and_unregistered_value(
    tmp_path: Path,
) -> None:
    fixture, _kwargs_value, _source, plane, parent, _predictions = _protected(
        tmp_path
    )
    _query_population, adapter, _preflight, _run, _replay, _output = (
        _query_adapter(fixture.root, plane)
    )
    mismatch = replace(parent.exact_parent, renderer_id=live.RENDERER_ID)
    with pytest.raises(
        confirmation_parent.MatchedEvalContractError,
        match="registered matched S0 binding",
    ):
        build_query_payload_answer_plan(adapter, mismatch)

    unregistered = "unregistered_test_renderer"
    object.__setattr__(adapter.source_population, "renderer_id", unregistered)
    bad_parent = replace(parent.exact_parent, renderer_id=unregistered)
    with pytest.raises(MatchedEvalContractError, match="unsupported matched renderer"):
        build_query_payload_answer_plan(adapter, bad_parent)


def _confirmation_plan(tmp_path: Path, *, count: int = 3):
    fixture, _kwargs_value, _source, plane, parent, _predictions = _protected(
        tmp_path, count=count
    )
    query_population, _adapter, preflight, run, _replay, query_root = (
        _query_adapter(fixture.root, plane)
    )
    plan = confirmation_parent.build_confirmation_query_payload_plan(
        parent,
        query_preflight_path=query_root / "query-expansion-preflight.json",
        expected_query_preflight_sha256=preflight.sha256,
        query_run_path=query_root / RUN_NAME,
        query_run_replay_path=query_root / RUN_REPLAY_NAME,
        expected_query_run_sha256=run.run_artifact.sha256,
        query_runtime_ledger_path=query_root / RUNTIME_LEDGER_NAME,
        query_runtime_ledger_replay_path=query_root / RUNTIME_LEDGER_REPLAY_NAME,
        expected_query_runtime_ledger_sha256=run.runtime_ledger_artifact.sha256,
        expected_query_population_id=query_population.population_id,
        expected_query_prompt_population_sha256=(
            query_population.prompt_population.prompt_population_sha256
        ),
    )
    return fixture, plan


def test_full_preflight_release_provider_materialize_replay_feeds_evidence_map(
    tmp_path: Path,
) -> None:
    fixture, plan = _confirmation_plan(tmp_path)
    output = fixture.root / "query-payload"
    preflight = confirmation_parent.publish_confirmation_query_payload_preflight(
        plan, output_root=output, max_concurrency=1
    )
    assert plan.required_calls == 3
    assert preflight.prompt_artifact.payload["population"][
        "required_provider_calls"
    ] == 3
    assert len(preflight.prompt_artifact.payload["ordered_rows"]) == 3
    assert preflight.prompt_artifact.payload["physical_provider_calls"] == 0
    assert preflight.prompt_artifact.payload["runtime"]["retries"] == 0

    with pytest.raises(
        confirmation_parent.ConfirmationQueryPayloadError,
        match="exact remaining",
    ):
        confirmation_parent.approve_confirmation_query_payload_release(
            preflight,
            output_root=output,
            expected_prompt_sha256=preflight.prompt_artifact.sha256,
            expected_answer_preflight_sha256=(
                preflight.answer_preflight_artifact.sha256
            ),
            approve_provider_release=True,
            authorized_provider_calls=2,
        )
    release = confirmation_parent.approve_confirmation_query_payload_release(
        preflight,
        output_root=output,
        expected_prompt_sha256=preflight.prompt_artifact.sha256,
        expected_answer_preflight_sha256=(
            preflight.answer_preflight_artifact.sha256
        ),
        approve_provider_release=True,
        authorized_provider_calls=3,
    )
    assert release.payload["required_authorized_provider_calls"] == 3

    factory_calls = []
    client = _StructuredClient("The exact globally retrieved answer.")

    def factory(gateway_url: str, api_key_env: str):
        factory_calls.append((gateway_url, api_key_env))
        return client

    with pytest.raises(
        confirmation_parent.ConfirmationQueryPayloadError,
        match="exact remaining",
    ):
        confirmation_parent.run_confirmation_query_payload_provider(
            preflight,
            output_root=output,
            expected_prompt_sha256=preflight.prompt_artifact.sha256,
            expected_answer_preflight_sha256=(
                preflight.answer_preflight_artifact.sha256
            ),
            expected_release_sha256=release.sha256,
            enable_provider=True,
            authorized_provider_calls=2,
            client_factory=factory,
        )
    assert factory_calls == []
    provider = confirmation_parent.run_confirmation_query_payload_provider(
        preflight,
        output_root=output,
        expected_prompt_sha256=preflight.prompt_artifact.sha256,
        expected_answer_preflight_sha256=(
            preflight.answer_preflight_artifact.sha256
        ),
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=3,
        client_factory=factory,
    )
    assert provider.physical_provider_calls == 3
    assert provider.checkpoint_hits == 0
    assert len(factory_calls) == 1

    materialized = (
        confirmation_parent.materialize_confirmation_query_payload_answers(
            preflight,
            output_root=output,
            expected_prompt_sha256=preflight.prompt_artifact.sha256,
            expected_answer_preflight_sha256=(
                preflight.answer_preflight_artifact.sha256
            ),
            expected_release_sha256=release.sha256,
        )
    )
    assert materialized.physical_provider_calls == 0
    verified = confirmation_parent.replay_confirmation_query_payload_answers(
        preflight,
        output_root=output,
        expected_prompt_sha256=preflight.prompt_artifact.sha256,
        expected_answer_preflight_sha256=(
            preflight.answer_preflight_artifact.sha256
        ),
        expected_release_sha256=release.sha256,
        expected_run_sha256=materialized.answer_artifact.sha256,
    )
    assert type(verified) is confirmation_parent.VerifiedQueryPayloadAnswerPlane
    assert verified.run_sha256 == verified.replay_sha256
    assert len(verified.rows) == 3
    evidence_map = build_evidence_map_plan(plan.answer_plan, verified)
    assert len(evidence_map.rows) == 3


def test_release_refuses_incomplete_journal_before_client_construction(
    tmp_path: Path,
) -> None:
    fixture, plan = _confirmation_plan(tmp_path, count=1)
    output = fixture.root / "query-payload"
    preflight = confirmation_parent.publish_confirmation_query_payload_preflight(
        plan, output_root=output, max_concurrency=1
    )
    checkpoint = output / "terra-query-payload-answer-calls"
    checkpoint.mkdir()
    (checkpoint / ("a" * 64 + ".request.json")).write_text(
        "{}", encoding="utf-8"
    )

    with pytest.raises(
        confirmation_parent.ConfirmationQueryPayloadError,
        match="unsafe retry forbidden",
    ):
        confirmation_parent.approve_confirmation_query_payload_release(
            preflight,
            output_root=output,
            expected_prompt_sha256=preflight.prompt_artifact.sha256,
            expected_answer_preflight_sha256=(
                preflight.answer_preflight_artifact.sha256
            ),
            approve_provider_release=True,
            authorized_provider_calls=1,
        )


def test_partial_checkpoint_release_authorizes_only_remaining_calls(
    tmp_path: Path,
) -> None:
    fixture, plan = _confirmation_plan(tmp_path, count=3)
    output = fixture.root / "query-payload"
    preflight = confirmation_parent.publish_confirmation_query_payload_preflight(
        plan, output_root=output, max_concurrency=1
    )
    seed_client = _StructuredClient("First sealed partial response.")
    runtime = query_payload_live._runtime(
        plan.answer_plan,
        checkpoint_dir=output / query_payload_live.CHECKPOINT_DIR_NAME,
        client=seed_client,
        max_concurrency=1,
        gateway_url=live.DEFAULT_GATEWAY_URL,
        preflight_sha256=preflight.answer_preflight_artifact.sha256,
    )
    assert plan.answer_plan.prompt_population is not None
    first_messages_sha = (
        plan.answer_plan.prompt_population.ordered_rows[0].messages_sha256
    )
    try:
        first = runtime._provider_call(first_messages_sha)
    finally:
        runtime.close()
    assert first.physical_call is True

    release = confirmation_parent.approve_confirmation_query_payload_release(
        preflight,
        output_root=output,
        expected_prompt_sha256=preflight.prompt_artifact.sha256,
        expected_answer_preflight_sha256=(
            preflight.answer_preflight_artifact.sha256
        ),
        approve_provider_release=True,
        authorized_provider_calls=2,
    )
    assert release.payload["checkpoint_snapshot"][
        "authenticated_complete_count"
    ] == 1
    assert release.payload["required_authorized_provider_calls"] == 2

    resumed_client = _StructuredClient("Remaining exact response.")
    provider = confirmation_parent.run_confirmation_query_payload_provider(
        preflight,
        output_root=output,
        expected_prompt_sha256=preflight.prompt_artifact.sha256,
        expected_answer_preflight_sha256=(
            preflight.answer_preflight_artifact.sha256
        ),
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=2,
        client_factory=lambda _gateway, _env: resumed_client,
    )
    assert provider.physical_provider_calls == 2
    assert provider.checkpoint_hits == 1


def test_resealed_release_with_extra_field_fails_exact_schema_before_client(
    tmp_path: Path,
) -> None:
    fixture, plan = _confirmation_plan(tmp_path, count=1)
    output = fixture.root / "query-payload"
    preflight = confirmation_parent.publish_confirmation_query_payload_preflight(
        plan, output_root=output, max_concurrency=1
    )
    release = confirmation_parent.approve_confirmation_query_payload_release(
        preflight,
        output_root=output,
        expected_prompt_sha256=preflight.prompt_artifact.sha256,
        expected_answer_preflight_sha256=(
            preflight.answer_preflight_artifact.sha256
        ),
        approve_provider_release=True,
        authorized_provider_calls=1,
    )
    changed = copy.deepcopy(release.payload)
    changed["benign_extension"] = "not allowed"
    changed.pop("release_identity_sha256")
    changed["release_identity_sha256"] = identity_sha256(changed)
    raw = canonical_json_bytes(changed)
    digest = hashlib.sha256(raw).hexdigest()
    release.path.write_bytes(raw)
    release.path.with_name(release.path.name + ".sha256").write_bytes(
        f"{digest}  {release.path.name}\n".encode("ascii")
    )
    factory_calls = []

    with pytest.raises(
        confirmation_parent.ConfirmationQueryPayloadError,
        match="release schema changed",
    ):
        confirmation_parent.run_confirmation_query_payload_provider(
            preflight,
            output_root=output,
            expected_prompt_sha256=preflight.prompt_artifact.sha256,
            expected_answer_preflight_sha256=(
                preflight.answer_preflight_artifact.sha256
            ),
            expected_release_sha256=digest,
            enable_provider=True,
            authorized_provider_calls=1,
            client_factory=lambda *_args: factory_calls.append(True),
        )
    assert factory_calls == []
