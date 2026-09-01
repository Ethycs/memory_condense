from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from tools.matched_eval.artifacts import publish_sealed_json
from tools.matched_eval.contracts import assert_gold_blind, identity_sha256
from tools.matched_eval.legacy import (
    DEFAULT_LEGACY_ROOT,
    EXTERNAL_S1_LABEL,
    JUDGE_BINDING_FORMAT,
    JUDGE_FORMAT,
    LEGACY_ARM_REGISTRY,
    RUN_FORMAT,
    LegacyArmMigration,
    LegacyArmSpec,
    LegacyArtifactPaths,
    LegacyImportError,
    import_legacy_artifacts,
    load_legacy_campaign,
    runtime_projection,
    score_projection,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _publish(path: Path, payload: dict[str, Any]) -> str:
    artifact, created = publish_sealed_json(path, payload)
    assert created
    return artifact.sha256


def _republish(path: Path, payload: dict[str, Any]) -> str:
    path.unlink()
    path.with_name(path.name + ".sha256").unlink()
    return _publish(path, payload)


def _synthetic_artifacts(tmp_path: Path) -> tuple[LegacyArtifactPaths, LegacyArmSpec]:
    arm_root = tmp_path / "synthetic-s0"
    paths = LegacyArtifactPaths(
        run=arm_root / "run.json",
        run_replay=arm_root / "run-replay.json",
        judge=arm_root / "semantic-judge-sol.json",
        judge_replay=arm_root / "semantic-judge-sol-replay.json",
    )
    population_sha = _sha("population")
    retrieval_sha = _sha("retrieval")
    run_rows: list[dict[str, Any]] = []
    for ordinal in range(100):
        prediction_text = f"prediction-{ordinal}"
        run_rows.append(
            {
                "call_key_sha256": _sha(f"call-{ordinal}"),
                "dated_question_sha256": _sha(f"dated-{ordinal}"),
                "ordinal": ordinal,
                "prediction": {
                    "sha256": _sha(prediction_text),
                    "text": prediction_text,
                },
                "provider_messages_sha256": _sha(f"prompt-{ordinal}"),
                "question_id": f"q{ordinal:03d}",
                "question_sha256": _sha(f"question-{ordinal}"),
                "request_journal_sha256": _sha(f"request-{ordinal}"),
                "response_journal_sha256": _sha(f"response-{ordinal}"),
                "retrieval_question_part_sha256": _sha(f"part-{ordinal}"),
                "source_binding_sha256": _sha(f"binding-{ordinal}"),
            }
        )
    run = {
        "arm_identity": {"parent_arm": None},
        "arm_label": "SYNTHETIC_S0",
        "format": RUN_FORMAT,
        "gold_loaded": False,
        "population_identity_sha256": population_sha,
        "question_count": 100,
        "questions": run_rows,
        "retained_request_token_state_bytes": 0,
        "retrieval_sha256": retrieval_sha,
    }
    run_sha = _publish(paths.run, run)
    run_replay_sha = _publish(paths.run_replay, run)

    judge_rows: list[dict[str, Any]] = []
    for ordinal, answer in enumerate(run_rows):
        baseline_correct = ordinal < 50
        correct = ordinal < 60
        judge_rows.append(
            {
                "baseline_correct": baseline_correct,
                "baseline_judge_row_sha256": _sha(f"baseline-row-{ordinal}"),
                "baseline_prediction_sha256": _sha(f"baseline-{ordinal}"),
                "changed_from_baseline": True,
                "correct": correct,
                "dated_question_sha256": answer["dated_question_sha256"],
                "evidence_topology_class": "point",
                "gold_answer_sha256": _sha(f"gold-{ordinal}"),
                "judge_verdict_sha256": _sha(f"verdict-{ordinal}"),
                "ordinal": ordinal,
                "prediction_sha256": answer["prediction"]["sha256"],
                "question_id": answer["question_id"],
                "question_only_demand_class": "direct_extract",
                "question_sha256": answer["question_sha256"],
                "regressed": baseline_correct and not correct,
                "rescued": not baseline_correct and correct,
                "verdict_source": "live_sol_judge",
            }
        )
    judge = {
        "aggregate": {
            "accepted_for_positive_only_composition": True,
            "baseline_correct": 50,
            "candidate_correct": 60,
            "net_marginal": 10,
            "regressed": 0,
            "rescued": 10,
        },
        "arm_label": "SYNTHETIC_S0",
        "arm_or_topology_labels_exposed_to_judge": False,
        "campaign_binding": {
            "arm_label": "SYNTHETIC_S0",
            "arm_run_replay_sha256": run_replay_sha,
            "arm_run_sha256": run_sha,
            "baseline_arm_label": EXTERNAL_S1_LABEL,
            "format": JUDGE_BINDING_FORMAT,
            "population_identity_sha256": population_sha,
            "question_count": 100,
            "retries": 0,
            "retrieval_sha256": retrieval_sha,
            "unique_judge_call_count": 100,
        },
        "explicit_gold_answer_text_persisted": False,
        "format": JUDGE_FORMAT,
        "gold_loaded_only_after_answer_run_replay": True,
        "logical_judgment_count": 100,
        "question_count": 100,
        "questions": judge_rows,
        "retained_request_token_state_bytes": 0,
        "topology_loaded_only_after_judge_prompt_seal": True,
        "unchanged_verdicts_reused_from_sealed_baseline": False,
        "unique_sol_completion_count": 100,
    }
    judge_sha = _publish(paths.judge, judge)
    judge_replay_sha = _publish(paths.judge_replay, judge)
    spec = LegacyArmSpec(
        arm_label="SYNTHETIC_S0",
        directory_name=arm_root.name,
        parent_arm_label=None,
        judge_baseline_arm_label=EXTERNAL_S1_LABEL,
        renderer_identity="legacy_renderer/s0_qa_v1",
        delta_kind="membership",
        run_sha256=run_sha,
        run_replay_sha256=run_replay_sha,
        judge_sha256=judge_sha,
        judge_replay_sha256=judge_replay_sha,
        expected_parent_run_sha256=None,
        baseline_correct=50,
        candidate_correct=60,
        rescued=10,
        regressed=0,
        unique_judge_call_count=100,
        accepted_for_positive_only_composition=True,
    )
    return paths, spec


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    assert type(value) is dict
    return value


def test_synthetic_import_separates_runtime_from_posthoc_score(tmp_path: Path) -> None:
    paths, spec = _synthetic_artifacts(tmp_path)

    migration = import_legacy_artifacts(paths, spec)

    assert isinstance(migration, LegacyArmMigration)
    assert migration.imported_provider_call_count == 0
    assert len(migration.runtime_observations) == 100
    assert len(migration.score_observations) == 100
    assert migration.score_aggregate.baseline_correct == 50
    assert migration.score_aggregate.candidate_correct == 60
    runtime = runtime_projection(migration)
    score = score_projection(migration)
    assert_gold_blind(runtime)
    assert runtime["renderer_identity"] == "legacy_renderer/s0_qa_v1"
    assert runtime["delta_kind"] == "membership"
    assert runtime["imported_provider_call_count"] == 0
    assert "aggregate" not in runtime
    assert score["aggregate"]["candidate_correct"] == 60
    source_run_rows = _load_json(paths.run)["questions"]
    source_judge_rows = _load_json(paths.judge)["questions"]
    assert tuple(row.source_row_sha256 for row in migration.runtime_observations) == tuple(
        identity_sha256(raw) for raw in source_run_rows
    )
    assert tuple(row.judge_row_sha256 for row in migration.score_observations) == tuple(
        identity_sha256(raw) for raw in source_judge_rows
    )
    assert runtime["observations"][0]["source_row_sha256"] == identity_sha256(
        source_run_rows[0]
    )
    assert score["observations"][0]["judge_row_sha256"] == identity_sha256(
        source_judge_rows[0]
    )
    json.dumps(runtime, allow_nan=False)
    json.dumps(score, allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        migration.imported_provider_call_count = 1  # type: ignore[misc]


def test_synthetic_import_rejects_gold_in_runtime(tmp_path: Path) -> None:
    paths, spec = _synthetic_artifacts(tmp_path)
    run = _load_json(paths.run)
    run["gold_loaded"] = True
    changed_sha = _republish(paths.run, run)
    changed_spec = replace(spec, run_sha256=changed_sha)

    with pytest.raises(LegacyImportError, match="loaded gold"):
        import_legacy_artifacts(paths, changed_spec)


def test_synthetic_import_rejects_run_replay_behavior_drift(tmp_path: Path) -> None:
    paths, spec = _synthetic_artifacts(tmp_path)
    replay = _load_json(paths.run_replay)
    replay["questions"][0]["prediction"] = {
        "sha256": _sha("changed prediction"),
        "text": "changed prediction",
    }
    changed_sha = _republish(paths.run_replay, replay)
    changed_spec = replace(spec, run_replay_sha256=changed_sha)

    with pytest.raises(LegacyImportError, match="behavior projection changed"):
        import_legacy_artifacts(paths, changed_spec)


def test_synthetic_import_rejects_unreproduced_judge_aggregate(
    tmp_path: Path,
) -> None:
    paths, spec = _synthetic_artifacts(tmp_path)
    judge = _load_json(paths.judge)
    judge["aggregate"]["candidate_correct"] = 61
    judge["aggregate"]["net_marginal"] = 11
    judge_sha = _republish(paths.judge, judge)
    judge_replay_sha = _republish(paths.judge_replay, judge)
    changed_spec = replace(
        spec,
        judge_sha256=judge_sha,
        judge_replay_sha256=judge_replay_sha,
        candidate_correct=61,
    )

    with pytest.raises(LegacyImportError, match="does not reproduce rows"):
        import_legacy_artifacts(paths, changed_spec)


def test_registry_has_only_the_three_explicit_legacy_renderers() -> None:
    assert tuple(LEGACY_ARM_REGISTRY) == (
        "S0_CONTROL",
        "S0_PLUS_EM_FACTS",
        "S0_PLUS_CAV_LINKS",
    )
    assert tuple(
        (spec.renderer_identity, spec.delta_kind)
        for spec in LEGACY_ARM_REGISTRY.values()
    ) == (
        ("legacy_renderer/s0_qa_v1", "membership"),
        ("legacy_renderer/em_facts_v1", "representation"),
        ("legacy_renderer/cav_links_v1", "linking"),
    )


def test_real_sealed_campaign_imports_when_artifacts_are_available() -> None:
    required = [
        path
        for spec in LEGACY_ARM_REGISTRY.values()
        for path in (
            LegacyArtifactPaths.under_root(DEFAULT_LEGACY_ROOT, spec).run,
            LegacyArtifactPaths.under_root(DEFAULT_LEGACY_ROOT, spec).run_replay,
            LegacyArtifactPaths.under_root(DEFAULT_LEGACY_ROOT, spec).judge,
            LegacyArtifactPaths.under_root(DEFAULT_LEGACY_ROOT, spec).judge_replay,
        )
    ]
    if not all(
        path.is_file() and path.with_name(path.name + ".sha256").is_file()
        for path in required
    ):
        pytest.skip("real sealed S0/EM/CAV checkpoint is not present")

    campaign = load_legacy_campaign(DEFAULT_LEGACY_ROOT)

    assert campaign.imported_provider_call_count == 0
    assert tuple(
        (
            arm.spec.arm_label,
            arm.score_aggregate.baseline_correct,
            arm.score_aggregate.candidate_correct,
        )
        for arm in campaign.arms
    ) == (
        ("S0_CONTROL", 56, 57),
        ("S0_PLUS_EM_FACTS", 57, 60),
        ("S0_PLUS_CAV_LINKS", 57, 53),
    )
    runtime = campaign.runtime_projection()
    assert_gold_blind(runtime)
    assert runtime["imported_provider_call_count"] == 0
    assert tuple(row["renderer_identity"] for row in runtime["arms"]) == (
        "legacy_renderer/s0_qa_v1",
        "legacy_renderer/em_facts_v1",
        "legacy_renderer/cav_links_v1",
    )
    assert campaign.score_projection()["arms"][1]["aggregate"][
        "candidate_correct"
    ] == 60
    assert campaign.manifest_projection()["arms"][2]["delta_kind"] == "linking"
