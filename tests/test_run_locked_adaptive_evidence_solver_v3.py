from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_matched_eval_adaptive_evidence_solver_live import (
    _parents,
    _source_union,
)
from tests.test_matched_eval_source_history_mapper_live import (
    _completion,
    _journal,
    _plans,
)
from tests.test_fast_completion_runtime import _FakeClient
from tools.matched_eval.adaptive_evidence_solver_live import (
    build_adaptive_evidence_solver_plan,
    preflight_adaptive_evidence_solver,
)
from tools.matched_eval.artifacts import SealedArtifact, read_sealed_json
from tools.matched_eval.source_history_fact_union import FactLane, LANE_ORDER
from tools.matched_eval.source_history_mapper_live import (
    build_source_history_mapper_preflight,
    materialize_source_history_mapper,
)
from tools.run_locked_adaptive_source_map import FastMaterializationQuestionPlan
from tools.run_locked_adaptive_evidence_solver_v3 import (
    LoadedAdaptiveSolverPlan,
    CHECKPOINT_DIR_NAME,
    RUN_NAME,
    _materialize,
    _preflight,
    _preflight_projection,
    _provider,
    _replay,
    _source_fact_unions,
    _validate_provider_preflight,
    parse_lane_filter,
)
import tools.run_locked_adaptive_evidence_solver_v3 as solver_cli


def _artifact(tmp_path: Path, name: str, digest: str) -> SealedArtifact:
    return SealedArtifact(tmp_path / name, digest * 64, {})


def _loaded(
    tmp_path: Path,
    *,
    with_source_fact: bool,
) -> LoadedAdaptiveSolverPlan:
    map_plan, map_plane = _parents(tmp_path / "parents")
    if with_source_fact:
        union = _source_union(tmp_path, map_plan, map_plane)
        unions = {map_plane.rows[0].question_id: union}
        lanes = (FactLane.DIRECT,)
    else:
        unions = {}
        lanes = LANE_ORDER
    plan = build_adaptive_evidence_solver_plan(
        map_plan,
        map_plane,
        source_fact_unions=unions,
    )
    preflight = preflight_adaptive_evidence_solver(plan)
    return LoadedAdaptiveSolverPlan(
        _artifact(tmp_path, "source-preflight.json", "a"),
        _artifact(tmp_path, "work.json", "b"),
        _artifact(tmp_path, "source-materialization.json", "c"),
        map_plan,
        map_plane,
        unions,
        lanes,
        "d" * 64,
        plan,
        preflight,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("all", LANE_ORDER),
        ("D", (FactLane.DIRECT,)),
        ("g,d", (FactLane.DIRECT, FactLane.GUIDED)),
        ("partition,em", (FactLane.PARTITION, FactLane.EM)),
    ],
)
def test_lane_filter_is_canonical_and_supports_isolated_or_combined_lanes(
    value: str,
    expected: tuple[FactLane, ...],
) -> None:
    assert parse_lane_filter(value) == expected


@pytest.mark.parametrize("value", ["", "all,d", "unknown"])
def test_lane_filter_rejects_ambiguous_or_unknown_profiles(value: str) -> None:
    with pytest.raises(Exception):
        parse_lane_filter(value)


def test_combined_mapper_materialization_derives_d_g_and_all_without_remap() -> None:
    _history, hydration, mapping = _plans()
    mapper_preflight = build_source_history_mapper_preflight(hydration, mapping)
    materialization = materialize_source_history_mapper(
        mapper_preflight,
        hydration,
        mapping,
        provider_journals=(_journal(mapper_preflight, _completion()),),
    )
    question = FastMaterializationQuestionPlan(
        0,
        "question-alpha",
        (),
        hydration,
        mapping,
        mapper_preflight,
    )

    direct = _source_fact_unions(
        (question,), (materialization,), lanes=(FactLane.DIRECT,)
    )["question-alpha"]
    guided = _source_fact_unions(
        (question,), (materialization,), lanes=(FactLane.GUIDED,)
    )["question-alpha"]
    combined = _source_fact_unions(
        (question,),
        (materialization,),
        lanes=(FactLane.DIRECT, FactLane.GUIDED),
    )["question-alpha"]

    assert direct.accepted_before_dedup_count == 1
    assert guided.accepted_before_dedup_count == 1
    assert combined.accepted_before_dedup_count == 2
    assert len(direct.retained_facts) == len(guided.retained_facts) == 1
    assert len(combined.retained_facts) == 1
    assert direct.retained_facts[0].owner_lane is FactLane.DIRECT
    assert guided.retained_facts[0].owner_lane is FactLane.GUIDED
    assert tuple(origin.lane for origin in combined.retained_facts[0].origins) == (
        FactLane.DIRECT,
        FactLane.GUIDED,
    )
    assert materialization.provider_calls_during_materialization == 0


def test_sealed_provider_preflight_accepts_only_actionable_source_fact_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(solver_cli, "EXPECTED_QUESTION_COUNT", 1)
    loaded = _loaded(tmp_path, with_source_fact=True)
    payload = _preflight_projection(
        loaded,
        gateway_url="https://controlled.invalid/v1",
        model="controlled/terra",
        max_concurrency=2,
    )
    artifact = SealedArtifact(tmp_path / "preflight.json", "e" * 64, payload)

    prompts, question_ids = _validate_provider_preflight(artifact)

    assert len(prompts) == loaded.plan.required_calls == 1
    assert question_ids == (loaded.plan.submitted_rows[0].question_id,)
    assert payload["actionable_submission_rule"] == (
        "at_least_one_admitted_source_fact_alias"
    )
    assert payload["provider_calls"] == 0
    assert payload["retained_transformer_token_state_bytes"] == 0


def test_map_only_or_absent_lane_population_seals_zero_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(solver_cli, "EXPECTED_QUESTION_COUNT", 1)
    loaded = _loaded(tmp_path, with_source_fact=False)
    payload = _preflight_projection(
        loaded,
        gateway_url="https://controlled.invalid/v1",
        model="controlled/terra",
        max_concurrency=2,
    )
    artifact = SealedArtifact(tmp_path / "preflight.json", "f" * 64, payload)

    prompts, question_ids = _validate_provider_preflight(artifact)

    assert prompts == ()
    assert question_ids == ()
    assert payload["required_authorized_provider_calls"] == 0
    assert all(not row.submitted for row in loaded.plan.rows)


def test_provider_preflight_rejects_prompt_or_authorization_count_tampering(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(solver_cli, "EXPECTED_QUESTION_COUNT", 1)
    loaded = _loaded(tmp_path, with_source_fact=True)
    payload = _preflight_projection(
        loaded,
        gateway_url="https://controlled.invalid/v1",
        model="controlled/terra",
        max_concurrency=2,
    )
    payload["required_authorized_provider_calls"] = 2
    artifact = SealedArtifact(tmp_path / "preflight.json", "1" * 64, payload)

    with pytest.raises(Exception, match="prompt/call population"):
        _validate_provider_preflight(artifact)


def test_provider_authorization_fails_before_environment_or_client(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(solver_cli, "EXPECTED_QUESTION_COUNT", 1)
    loaded = _loaded(tmp_path, with_source_fact=True)
    payload = _preflight_projection(
        loaded,
        gateway_url="https://controlled.invalid/v1",
        model="controlled/terra",
        max_concurrency=2,
    )
    artifact = SealedArtifact(tmp_path / "preflight.json", "2" * 64, payload)
    prompts, question_ids = _validate_provider_preflight(artifact)
    monkeypatch.setattr(
        solver_cli,
        "_read_preflight",
        lambda *_args, **_kwargs: (artifact, prompts, question_ids),
    )
    monkeypatch.setattr(
        solver_cli,
        "load_dotenv",
        lambda: (_ for _ in ()).throw(AssertionError("environment loaded")),
    )
    monkeypatch.setattr(
        solver_cli.live,
        "_make_provider_client",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("client created")
        ),
    )
    args = SimpleNamespace(
        output_root=tmp_path,
        expected_preflight_sha256=artifact.sha256,
        enable_provider=True,
        authorized_provider_calls=0,
        api_key_env="MISSING",
        model="controlled/terra",
        gateway_url="https://controlled.invalid/v1",
        max_concurrency=2,
    )

    with pytest.raises(Exception, match="exact authorization for 1 calls"):
        _provider(args)


def test_preflight_provider_materialize_and_replay_are_one_sealed_lifecycle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(solver_cli, "EXPECTED_QUESTION_COUNT", 1)
    loaded = _loaded(tmp_path / "inputs", with_source_fact=True)
    monkeypatch.setattr(solver_cli, "_load_plan", lambda _args: loaded)
    output = tmp_path / "output"
    settings = {
        "output_root": output,
        "gateway_url": "https://controlled.invalid/v1",
        "model": "controlled/terra",
        "max_concurrency": 1,
    }
    preflight_result = _preflight(SimpleNamespace(**settings))
    preflight_sha = preflight_result["preflight_sha256"]
    fake = _FakeClient(output / CHECKPOINT_DIR_NAME)
    monkeypatch.setenv("TEST_LITELLM_KEY", "controlled-key")
    monkeypatch.setattr(solver_cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        solver_cli.live,
        "_make_provider_client",
        lambda *_args, **_kwargs: fake,
    )
    provider_result = _provider(
        SimpleNamespace(
            **settings,
            expected_preflight_sha256=preflight_sha,
            enable_provider=True,
            authorized_provider_calls=1,
            api_key_env="TEST_LITELLM_KEY",
        )
    )
    materialized = _materialize(
        SimpleNamespace(
            **settings,
            expected_preflight_sha256=preflight_sha,
        )
    )
    terminal = read_sealed_json(output / RUN_NAME)
    replayed = _replay(
        SimpleNamespace(
            **settings,
            expected_preflight_sha256=preflight_sha,
            expected_run_sha256=terminal.sha256,
        )
    )

    assert provider_result["physical_provider_calls"] == 1
    assert materialized["physical_provider_calls"] == 0
    assert materialized["checkpoint_hits"] == 1
    assert materialized["run_sha256"] == terminal.sha256
    assert terminal.payload["physical_provider_calls_during_materialization"] == 0
    assert terminal.payload["retained_transformer_token_state_bytes"] == 0
    assert replayed["byte_identical"] is True
    assert replayed["run_sha256"] == terminal.sha256
    assert fake.close_calls == 1
