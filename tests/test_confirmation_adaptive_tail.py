from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_condense.domain._tokenizer import count_tokens
from memory_condense.domain.discourse import quote_sha256
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime
from tests.test_matched_eval_adaptive_evidence_solver_live import _source_union
from tests.test_matched_eval_query_evidence_map_solver_v2_live import _terminal_map
from tools import confirmation_adaptive_tail as subject
from tools import run_locked_adaptive_source_map as source_cli
from tools._routed_repair_routing import route_question
from tools.matched_eval.adaptive_evidence_solver_live import (
    build_adaptive_evidence_solver_plan,
    preflight_adaptive_evidence_solver,
)
from tools.matched_eval.artifacts import publish_sealed_json
from tools.matched_eval.contracts import ArtifactRef, canonical_json_bytes, identity_sha256
from tools.matched_eval.locked_source_gate_adapter import (
    DIRECT_STREAM_PROFILE_REPACK_V2,
    DIRECT_STREAM_PROFILE_V1,
    LockedSourceGateAdapterPopulation,
    LockedSourceGateQuestion,
)
from tools.matched_eval.query_expansion import FrozenSourceMembership, FrozenSourceNamespace
from tools.matched_eval.source_gate_controller import (
    EligibleFrontierScope,
    ObligationKind,
    QuestionObligation,
    SourceGateActivationReceipt,
    SourceGateCandidate,
    SourceGatePlan,
)
from tools.matched_eval.source_history_fact_union import (
    FactLane,
    FrozenHistoryChunk,
    HydratedSourceHistory,
    ParentIdentity,
    direct_evidence_projection_sha256,
)
from tools.matched_eval.source_history_mapper_live import (
    MAX_PROMPT_TOKENS,
    OUTPUT_TOKEN_RESERVE,
    WorkDisposition,
)


def _sha(value: str) -> str:
    return quote_sha256(value)


class _Completions:
    def __init__(self, responder):
        self.responder = responder
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def create(self, **request):
        with self._lock:
            self.requests.append(request)
            number = len(self.requests)
        completion = self.responder(request)
        return SimpleNamespace(
            id=f"fake-{number}",
            model="fake-terra",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=completion),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=4,
                total_tokens=14,
            ),
        )


class _Client:
    def __init__(self, responder):
        self.max_retries = 0
        self.chat = SimpleNamespace(completions=_Completions(responder))
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _Factory:
    def __init__(self, responder):
        self.calls: list[tuple[str, str]] = []
        self.client = _Client(responder)

    def __call__(self, gateway_url: str, api_key_env: str):
        self.calls.append((gateway_url, api_key_env))
        return self.client


def _history(namespace_id: str, membership: FrozenSourceMembership, text: str):
    chunk_id = membership.content_chunk_ids[0]
    chunk = FrozenHistoryChunk(
        membership.source_id,
        chunk_id,
        _sha(f"turn:{membership.source_id}"),
        0,
        "user",
        "2026-08-01T00:00:00+00:00",
        0,
        len(text),
        text,
        count_tokens(text),
        quote_sha256(text),
        False,
    )
    projection_sha = identity_sha256(membership.projection())
    receipt = identity_sha256(
        {
            "chunks": [chunk.chunk_receipt_sha256],
            "namespace_id": namespace_id,
            "source_id": membership.source_id,
        }
    )
    return HydratedSourceHistory(
        namespace_id,
        membership.source_id,
        membership.content_chunk_ids,
        (),
        membership.stream_sha256,
        projection_sha,
        (chunk,),
        True,
        receipt,
    )


def _question(
    tmp_path: Path,
    ordinal: int,
    *,
    retain_marker: bool,
):
    question_id = f"q{ordinal}"
    dated = f"What are all the projects I completed for case {ordinal}?"
    source_ids = tuple(
        [f"{question_id}-d{rank}" for rank in range(5)]
        + [f"{question_id}-p0", f"{question_id}-g0", f"{question_id}-g1"]
    )
    memberships = tuple(
        FrozenSourceMembership(
            source_id,
            (_sha(f"chunk:{source_id}"),),
            (),
            _sha(f"stream:{source_id}"),
        )
        for source_id in source_ids
    )
    namespace = FrozenSourceNamespace(
        _sha(f"snapshot:{question_id}"),
        _sha(f"store:{question_id}"),
        memberships,
    )
    by_source = {row.source_id: row for row in memberships}
    histories = {
        (namespace.namespace_id, source_id): _history(
            namespace.namespace_id,
            by_source[source_id],
            (
                f"{source_id} retain-marker needle evidence."
                if retain_marker and source_id.endswith(("d0", "g0"))
                else f"{source_id} carries needle evidence."
            ),
        )
        for source_id in source_ids
    }
    candidates = tuple(
        SourceGateCandidate(
            lane,
            namespace.namespace_id,
            source_id,
            rank,
            identity_sha256(by_source[source_id].projection()),
            by_source[source_id].stream_sha256,
            _sha(f"lane-stream:{question_id}:{lane.value}"),
        )
        for lane, ranked in (
            (FactLane.DIRECT, tuple((f"{question_id}-d{i}", i) for i in range(5))),
            (FactLane.PARTITION, ((f"{question_id}-p0", 0),)),
            (FactLane.GUIDED, ((f"{question_id}-g0", 0), (f"{question_id}-g1", 1))),
        )
        for source_id, rank in ranked
    )
    parent = ParentIdentity(
        _sha("population"),
        _sha("question-order"),
        namespace.snapshot_id,
        namespace.namespace_id,
        _sha(f"map-packet:{question_id}"),
        _sha(f"map-stage:{question_id}"),
        direct_evidence_projection_sha256(()),
    )
    route = route_question(dated)
    obligation = QuestionObligation(
        ObligationKind.FRONTIER,
        ("needle",),
        1,
        1,
        1,
        False,
        True,
    )
    activation = SourceGateActivationReceipt(
        question_id,
        _sha(dated),
        _sha(dated),
        parent.parent_packet_id,
        _sha(f"upstream-plan:{question_id}"),
        _sha(f"upstream-frontier:{question_id}"),
        (obligation.obligation_id,),
        (obligation.obligation_id,),
    )
    plan = SourceGatePlan(
        parent,
        question_id,
        _sha(dated),
        dated,
        _sha(dated),
        0,
        route,
        (ArtifactRef("source", _sha(f"source:{question_id}"), "source.json"),),
        candidates,
        (obligation,),
        activation,
        EligibleFrontierScope(
            tuple(row.candidate_id for row in candidates),
            False,
            _sha(f"frontier:{question_id}"),
        ),
        source_cli.source_gate_policy(1, 0, 1),
    )
    store = tmp_path / f"store-{ordinal}"
    store.mkdir(parents=True)
    locked = LockedSourceGateQuestion(
        ordinal,
        plan,
        _sha(f"source-packet:{question_id}"),
        _sha(f"activation-input:{question_id}"),
        namespace,
        (),
        store,
        _sha(f"database:{question_id}"),
        _sha(f"index:{question_id}"),
    )
    return locked, histories


def _mapper_responder(request: dict) -> str:
    joined = "\n".join(row["content"] for row in request["messages"])
    if "retain-marker" not in joined:
        return '{"facts":[]}'
    return json.dumps(
        {
            "facts": [
                {
                    "chunk_alias": "C001",
                    "event_tuple": None,
                    "fact": "The source contains retain-marker needle evidence.",
                    "quote": "retain-marker needle evidence",
                    "source_alias": "S1",
                }
            ]
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _artifact(tmp_path: Path, name: str, payload: dict):
    return publish_sealed_json(tmp_path / name, payload)[0]


def _upstream(tmp_path: Path, count: int, *, retained: set[int] = set()):
    questions = []
    histories = {}
    for ordinal in range(count):
        question, rows = _question(
            tmp_path, ordinal, retain_marker=ordinal in retained
        )
        questions.append(question)
        histories.update(rows)
    artifacts = (ArtifactRef("source_plane", _sha("source-plane"), "plane.json"),)
    source = LockedSourceGateAdapterPopulation(
        artifacts, tuple(questions), DIRECT_STREAM_PROFILE_V1
    )
    repack = LockedSourceGateAdapterPopulation(
        artifacts, tuple(questions), DIRECT_STREAM_PROFILE_REPACK_V2
    )
    base_plan = source_cli.build_locked_base_round(
        source,
        prehydrated=((), histories),
    )
    prompts = tuple(
        tuple({"role": message.role, "content": message.content} for message in row.messages)
        for row in base_plan.submitted_prompt_rows
    )
    runtime = FastCompletionRuntime(
        checkpoint_dir=tmp_path / "base-calls",
        prompt_population=prompts,
        model="fixture-terra",
        client=_Client(_mapper_responder),
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_new_tokens=OUTPUT_TOKEN_RESERVE,
        max_concurrency=1,
        retries=0,
        benchmark_provenance={"fixture": "confirmation-adaptive-base"},
    )
    try:
        batch = runtime.run()
    finally:
        runtime.close()
    by_id = {row.plan.question_id: row for row in source.questions}
    typed_questions = tuple(
        source_cli.FastMaterializationQuestionPlan(
            row.ordinal,
            row.question_id,
            by_id[row.question_id].direct_evidence,
            row.hydration_plan,
            row.mapping_plan,
            row.mapper_preflight,
        )
        for row in base_plan.questions
    )
    materializations = source_cli.materialize_fast_question_plans(
        typed_questions, batch
    )
    preflight = _artifact(
        tmp_path,
        "base-preflight.json",
        {
            "format": "fixture-base-preflight",
            "gold_loaded": False,
            "source_gate_population_receipt_sha256": source.receipt_sha256,
        },
    )
    work = _artifact(
        tmp_path,
        "base-work.json",
        {"format": "fixture-base-work", "gold_loaded": False},
    )
    materialization = _artifact(
        tmp_path,
        "base-materialization.json",
        {
            "format": "fixture-base-materialization",
            "gold_loaded": False,
            "preflight_artifact_sha256": preflight.sha256,
        },
    )
    replay = _artifact(
        tmp_path,
        "base-replay.json",
        {
            "byte_identical": True,
            "expected_materialization_sha256": materialization.sha256,
            "format": "fixture-base-replay",
            "gold_loaded": False,
        },
    )
    source_stream = _artifact(
        tmp_path,
        "source-stream.json",
        {"format": "fixture-source-stream", "gold_loaded": False},
    )
    map_plan, _output, _pre, _provider, _journals, _result, map_plane = _terminal_map(
        tmp_path / "map-parent"
    )
    upstream = subject.ConfirmationAdaptiveUpstream(
        preflight,
        work,
        materialization,
        replay,
        typed_questions,
        materializations,
        batch,
        map_plan,
        map_plane,
        source,
        repack,
        source_stream,
    )
    return upstream, histories


def _tail_preflight(tmp_path: Path, count: int = 2):
    upstream, histories = _upstream(tmp_path / "parents", count)

    def hydrate(rows):
        keys = {
            (question.plan.parent.namespace_id, selection.source_id)
            for question, round_ in rows
            for selection in round_.selections
        }
        return (), {key: histories[key] for key in keys}

    plan = subject.build_confirmation_adaptive_tail_plan(
        upstream, hydrator=hydrate
    )
    preflight = subject.publish_confirmation_adaptive_tail_preflight(
        plan, output_root=tmp_path / "tail", max_concurrency=1
    )
    return upstream, plan, preflight


@pytest.mark.parametrize("count", [1, 3])
def test_tail_is_arbitrary_size_and_preserves_selection_before_dedup(
    tmp_path: Path, count: int
) -> None:
    _upstream_value, plan, preflight = _tail_preflight(tmp_path, count)

    assert len(plan.decisions) == count
    assert len(plan.questions) == count
    assert all(
        row.disposition is subject.tail_core.TailDisposition.SELECTED
        for row in plan.decisions
    )
    assert all(row.selected_lane is FactLane.PARTITION for row in plan.decisions)
    assert preflight.artifact.payload["logical_source_selected_before_physical_dedup"] is True
    assert preflight.artifact.payload["post_map_dedup_performed"] is False
    assert "100" not in json.dumps(preflight.artifact.payload["question_count"])


def test_solver_no_applicable_row_preserves_exact_parent_without_provider(
    tmp_path: Path,
) -> None:
    upstream, _histories = _upstream(tmp_path / "parents", 1)
    empty = replace(upstream, base_questions=(), base_materializations=(), base_completion_batch=None)
    plan = subject.build_confirmation_adaptive_evidence_plan(empty)
    preflight = subject.publish_confirmation_adaptive_evidence_preflight(
        plan, output_root=tmp_path / "solver", max_concurrency=1
    )
    release = subject.approve_confirmation_adaptive_release(
        preflight,
        output_root=tmp_path / "solver",
        expected_preflight_sha256=preflight.artifact.sha256,
        approve_provider_release=True,
        authorized_provider_calls=0,
    )
    factory = _Factory(lambda _request: (_ for _ in ()).throw(AssertionError()))
    execution = subject.run_confirmation_adaptive_provider(
        preflight,
        output_root=tmp_path / "solver",
        expected_preflight_sha256=preflight.artifact.sha256,
        expected_release_sha256=release.sha256,
        enable_provider=False,
        authorized_provider_calls=0,
        client_factory=factory,
    )
    verified = subject.materialize_confirmation_adaptive_evidence(
        preflight,
        output_root=tmp_path / "solver",
        expected_preflight_sha256=preflight.artifact.sha256,
        expected_release_sha256=release.sha256,
    )

    assert execution.physical_provider_calls == 0
    assert factory.calls == []
    assert verified.plane.rows[0].solver_decision == "not_submitted"
    assert verified.plane.rows[0].prediction == upstream.map_plan.rows[0].direct_answer_row.prediction


def test_solver_provider_materialize_and_exact_replay(tmp_path: Path) -> None:
    upstream, _histories = _upstream(tmp_path / "parents", 1)
    union = _source_union(
        tmp_path / "union", upstream.map_plan, upstream.map_plane
    )
    question_id = upstream.map_plane.rows[0].question_id
    core_plan = build_adaptive_evidence_solver_plan(
        upstream.map_plan,
        upstream.map_plane,
        source_fact_unions={question_id: union},
    )
    plan = subject.ConfirmationAdaptiveEvidencePlan(
        upstream,
        ((question_id, union),),
        core_plan,
        preflight_adaptive_evidence_solver(core_plan),
    )
    root = tmp_path / "solver"
    preflight = subject.publish_confirmation_adaptive_evidence_preflight(
        plan, output_root=root, max_concurrency=1
    )
    release = subject.approve_confirmation_adaptive_release(
        preflight,
        output_root=root,
        expected_preflight_sha256=preflight.artifact.sha256,
        approve_provider_release=True,
        authorized_provider_calls=1,
    )
    factory = _Factory(
        lambda _request: json.dumps(
            {
                "decision": "replace",
                "prediction": "Alpha stored the blue token.",
                "used_evidence_ids": ["D001"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    execution = subject.run_confirmation_adaptive_provider(
        preflight,
        output_root=root,
        expected_preflight_sha256=preflight.artifact.sha256,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=1,
        client_factory=factory,
    )
    first = subject.materialize_confirmation_adaptive_evidence(
        preflight,
        output_root=root,
        expected_preflight_sha256=preflight.artifact.sha256,
        expected_release_sha256=release.sha256,
    )
    second = subject.replay_confirmation_adaptive_evidence(
        preflight,
        output_root=root,
        expected_preflight_sha256=preflight.artifact.sha256,
        expected_release_sha256=release.sha256,
        expected_run_sha256=first.run_artifact.sha256,
        expected_replay_sha256=first.replay_artifact.sha256,
    )

    assert execution.physical_provider_calls == 1
    assert first.run_artifact.sha256 == second.run_artifact.sha256
    assert first.run == second.run
    assert first.plane == second.plane
    assert first.plane.rows[0].prediction == "Alpha stored the blue token."


def test_tail_partial_resume_releases_only_remaining_and_replays(
    tmp_path: Path,
) -> None:
    _upstream_value, plan, preflight = _tail_preflight(tmp_path, 2)
    root = tmp_path / "tail"
    stage, checkpoint, _release_name, cap, reserve, rows = subject._stage_config(preflight)
    seed_client = _Client(lambda _request: '{"facts":[]}')
    runtime = subject._runtime(
        preflight.artifact,
        rows,
        output_root=root,
        stage=stage,
        checkpoint_name=checkpoint,
        max_prompt_tokens=cap,
        output_token_reserve=reserve,
        client=seed_client,
    )
    try:
        runtime._provider_call(rows[0]["messages_sha256"])
    finally:
        runtime.close()
    release = subject.approve_confirmation_adaptive_release(
        preflight,
        output_root=root,
        expected_preflight_sha256=preflight.artifact.sha256,
        approve_provider_release=True,
        authorized_provider_calls=plan.required_calls - 1,
    )
    factory = _Factory(lambda _request: '{"facts":[]}')
    execution = subject.run_confirmation_adaptive_provider(
        preflight,
        output_root=root,
        expected_preflight_sha256=preflight.artifact.sha256,
        expected_release_sha256=release.sha256,
        enable_provider=True,
        authorized_provider_calls=plan.required_calls - 1,
        client_factory=factory,
    )
    first = subject.materialize_confirmation_adaptive_tail(
        preflight,
        output_root=root,
        expected_preflight_sha256=preflight.artifact.sha256,
        expected_release_sha256=release.sha256,
    )
    second = subject.replay_confirmation_adaptive_tail(
        preflight,
        output_root=root,
        expected_preflight_sha256=preflight.artifact.sha256,
        expected_release_sha256=release.sha256,
        expected_run_sha256=first.run_artifact.sha256,
        expected_replay_sha256=first.replay_artifact.sha256,
    )

    assert execution.checkpoint_hits == 1
    assert execution.physical_provider_calls == plan.required_calls - 1
    assert len(factory.client.chat.completions.requests) == plan.required_calls - 1
    assert first.run_artifact.sha256 == second.run_artifact.sha256
    assert first.materializations == second.materializations
    assert first.fact_union_rows == second.fact_union_rows
    assert all(row.fact_union.direct_exclusions == () for row in first.fact_union_rows)


def test_request_only_and_foreign_checkpoint_state_fail_closed(
    tmp_path: Path,
) -> None:
    _upstream_value, _plan, preflight = _tail_preflight(tmp_path, 1)
    root = tmp_path / "tail"
    stage, checkpoint, _release_name, cap, reserve, rows = subject._stage_config(preflight)
    runtime = subject._runtime(
        preflight.artifact,
        rows,
        output_root=root,
        stage=stage,
        checkpoint_name=checkpoint,
        max_prompt_tokens=cap,
        output_token_reserve=reserve,
        client=_Client(lambda _request: '{"facts":[]}'),
    )
    try:
        runtime._reserve(rows[0]["messages_sha256"])
    finally:
        runtime.close()
    with pytest.raises(subject.ConfirmationAdaptiveTailError, match="incomplete"):
        subject.approve_confirmation_adaptive_release(
            preflight,
            output_root=root,
            expected_preflight_sha256=preflight.artifact.sha256,
            approve_provider_release=True,
            authorized_provider_calls=1,
        )

    other_root = tmp_path / "foreign"
    _upstream_value, plan, other = _tail_preflight(other_root, 1)
    checkpoint_root = other_root / "tail" / subject.TAIL_CHECKPOINT_DIR_NAME
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "foreign.txt").write_text("foreign", encoding="utf-8")
    with pytest.raises(subject.ConfirmationAdaptiveTailError, match="foreign"):
        subject.approve_confirmation_adaptive_release(
            other,
            output_root=other_root / "tail",
            expected_preflight_sha256=other.artifact.sha256,
            approve_provider_release=True,
            authorized_provider_calls=plan.required_calls,
        )


def test_release_extra_field_and_parent_tamper_fail_closed(tmp_path: Path) -> None:
    _upstream_value, plan, preflight = _tail_preflight(tmp_path, 1)
    root = tmp_path / "tail"
    release = subject.approve_confirmation_adaptive_release(
        preflight,
        output_root=root,
        expected_preflight_sha256=preflight.artifact.sha256,
        approve_provider_release=True,
        authorized_provider_calls=plan.required_calls,
    )
    payload = dict(release.payload)
    payload["extra"] = "forbidden"
    raw = canonical_json_bytes(payload)
    release.path.write_bytes(raw)
    digest = __import__("hashlib").sha256(raw).hexdigest()
    release.path.with_name(release.path.name + ".sha256").write_bytes(
        f"{digest}  {release.path.name}\n".encode("ascii")
    )
    with pytest.raises(subject.ConfirmationAdaptiveTailError, match="schema"):
        subject.run_confirmation_adaptive_provider(
            preflight,
            output_root=root,
            expected_preflight_sha256=preflight.artifact.sha256,
            expected_release_sha256=digest,
            enable_provider=True,
            authorized_provider_calls=plan.required_calls,
            client_factory=_Factory(lambda _request: '{"facts":[]}'),
        )

    changed = replace(
        preflight.plan,
        max_new_provider_calls=preflight.plan.max_new_provider_calls - 1,
    )
    forged = replace(preflight, plan=changed)
    with pytest.raises(subject.ConfirmationAdaptiveTailError, match="exact typed parents"):
        subject.approve_confirmation_adaptive_release(
            forged,
            output_root=root,
            expected_preflight_sha256=preflight.artifact.sha256,
            approve_provider_release=True,
            authorized_provider_calls=plan.required_calls,
        )
