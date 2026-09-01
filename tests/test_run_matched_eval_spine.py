from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from tools.matched_eval.artifacts import read_sealed_json
from tools.matched_eval.contracts import identity_sha256
from tools.matched_eval.legacy import load_legacy_campaign
from tools.run_matched_eval_spine import (
    DEFAULT_CAMPAIGN_ROOT,
    inspect_outputs,
    migrate_legacy,
)


@pytest.mark.skipif(
    not DEFAULT_CAMPAIGN_ROOT.is_dir(), reason="sealed legacy campaign is absent"
)
def test_legacy_cli_path_emits_common_ledgers_and_is_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "migration"
    first = migrate_legacy(campaign_root=DEFAULT_CAMPAIGN_ROOT, output_root=output)
    second = migrate_legacy(campaign_root=DEFAULT_CAMPAIGN_ROOT, output_root=output)

    assert first["scores"] == {
        "S0_CONTROL": 57,
        "S0_PLUS_EM_FACTS": 60,
        "S0_PLUS_CAV_LINKS": 53,
    }
    assert first["migration_created"] is True
    assert second["migration_created"] is False
    assert second["migration_sha256"] == first["migration_sha256"]

    runtime = read_sealed_json(output / "legacy-runtime-ledger.json").payload
    scores = read_sealed_json(output / "legacy-score-ledger.json").payload
    campaign = load_legacy_campaign(DEFAULT_CAMPAIGN_ROOT)
    assert runtime["format"] == "memory-condense-matched-runtime-ledger-v2"
    assert runtime["question_count"] == 100
    assert runtime["row_count"] == 300
    assert runtime["total_provider_calls"] == 0
    assert runtime["total_historical_provider_calls"] == 362
    assert runtime["total_historical_local_model_calls"] == 4
    assert scores["row_count"] == 300
    assert scores["total_historical_provider_calls"] == 174
    assert inspect_outputs(output_root=output)["new_provider_calls"] == 0

    expected_runtime_artifacts = tuple(
        artifact
        for arm in campaign.arms
        for artifact in (
            {"role": f"{arm.spec.arm_label}:run", "sha256": arm.run_artifact.sha256},
            {
                "role": f"{arm.spec.arm_label}:run_replay",
                "sha256": arm.run_replay_artifact.sha256,
            },
        )
    )
    expected_score_artifacts = tuple(
        artifact
        for arm in campaign.arms
        for artifact in (
            {
                "role": f"{arm.spec.arm_label}:judge",
                "sha256": arm.judge_artifact.sha256,
            },
            {
                "role": f"{arm.spec.arm_label}:judge_replay",
                "sha256": arm.judge_replay_artifact.sha256,
            },
        )
    )
    assert tuple(runtime["source_artifacts"]) == expected_runtime_artifacts
    assert tuple(scores["source_artifacts"]) == expected_score_artifacts
    assert all("judge" not in row["role"] for row in runtime["source_artifacts"])

    expected_runtime_rows = tuple(
        observation
        for arm in campaign.arms
        for observation in arm.runtime_observations
    )
    expected_score_rows = tuple(
        observation for arm in campaign.arms for observation in arm.score_observations
    )
    assert tuple(row["source_row_sha256"] for row in runtime["rows"]) == tuple(
        row.source_row_sha256 for row in expected_runtime_rows
    )
    assert tuple(row["judge_row_sha256"] for row in scores["rows"]) == tuple(
        row.judge_row_sha256 for row in expected_score_rows
    )
    assert tuple(row["judge_verdict_sha256"] for row in scores["rows"]) == tuple(
        row.judge_verdict_sha256 for row in expected_score_rows
    )
    assert tuple(
        row["baseline_judge_row_sha256"] for row in scores["rows"]
    ) == tuple(row.baseline_judge_row_sha256 for row in expected_score_rows)

    for row in runtime["rows"]:
        body = dict(row)
        row_id = body.pop("row_id")
        assert row_id == identity_sha256(body)
    runtime_body = dict(runtime)
    runtime_identity = runtime_body.pop("ledger_identity_sha256")
    assert runtime_identity == identity_sha256(runtime_body)
    score_body = dict(scores)
    score_identity = score_body.pop("ledger_identity_sha256")
    assert score_identity == identity_sha256(score_body)

    runtime_by_id = {row["row_id"]: row for row in runtime["rows"]}
    assert scores["runtime_ledger_identity_sha256"] == runtime["ledger_identity_sha256"]
    assert set(runtime_by_id) == {
        row["runtime_row_id"] for row in scores["rows"]
    }
    candidate_correct: Counter[str] = Counter()
    runtime_calls: Counter[str] = Counter()
    judge_calls: Counter[str] = Counter()
    for row in runtime["rows"]:
        runtime_calls[row["arm_label"]] += row["historical_provider_calls"]
    for row in scores["rows"]:
        arm_label = runtime_by_id[row["runtime_row_id"]]["arm_label"]
        candidate_correct[arm_label] += int(row["correct"])
        judge_calls[arm_label] += row["historical_provider_calls"]
    assert dict(candidate_correct) == {
        "S0_CONTROL": 57,
        "S0_PLUS_EM_FACTS": 60,
        "S0_PLUS_CAV_LINKS": 53,
    }
    assert dict(runtime_calls) == {
        "S0_CONTROL": 100,
        "S0_PLUS_EM_FACTS": 162,
        "S0_PLUS_CAV_LINKS": 100,
    }
    assert dict(judge_calls) == {
        "S0_CONTROL": 100,
        "S0_PLUS_EM_FACTS": 43,
        "S0_PLUS_CAV_LINKS": 31,
    }
