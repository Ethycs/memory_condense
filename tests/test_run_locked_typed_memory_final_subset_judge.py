from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memory_condense.domain.discourse import quote_sha256
from tools import run_locked_typed_memory_final_judge as judge_cli
from tools.matched_eval.artifacts import SealedArtifact, read_sealed_json
from tools.matched_eval.typed_memory_final_judging import (
    PREFLIGHT_NAME,
    TypedFinalJudgeGoldRow,
    validate_preflight_artifact,
)


def _sha(value: str) -> str:
    return quote_sha256(value)


def _subset_rows() -> tuple[
    tuple[dict[str, Any], ...], tuple[TypedFinalJudgeGoldRow, ...]
]:
    sources: list[dict[str, Any]] = []
    gold: list[TypedFinalJudgeGoldRow] = []
    for ordinal in judge_cli.subset_cli.MISS_ORDINALS:
        question_id = f"subset-question-{ordinal:03d}"
        question = f"What is the sealed value for row {ordinal}?"
        dated = f"[Question asked at 2023/05/30 16:15]\n{question}"
        prediction = f"predicted value {ordinal}"
        reference = f"reference value {ordinal}"
        sources.append(
            {
                "changed_from_parent": True,
                "dated_question_sha256": _sha(dated),
                "format": "synthetic-subset-judge-row",
                "ordinal": ordinal,
                "parent_prediction_sha256": _sha(f"parent {ordinal}"),
                "prediction": prediction,
                "prediction_sha256": _sha(prediction),
                "prediction_source": "synthetic_subset_replacement_v1",
                "question_id": question_id,
                "question_sha256": _sha(question),
                "route_id": "extract",
                "source_row_sha256": _sha(f"source row {ordinal}"),
            }
        )
        gold.append(
            TypedFinalJudgeGoldRow(
                ordinal,
                question_id,
                question,
                _sha(question),
                dated,
                _sha(dated),
                reference,
                _sha(reference),
                "synthetic",
            )
        )
    return tuple(sources), tuple(gold)


def _artifact(path: str, label: str) -> SealedArtifact:
    return SealedArtifact(Path(path), _sha(label), {"label": label})


def test_subset_preflight_verifies_replay_before_gold_and_seals_standard_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_rows, gold_rows = _subset_rows()
    run = _artifact("subset-run.json", "subset run")
    replay = _artifact("subset-replay.json", "subset replay")
    calls: list[object] = []

    def verified_reader(
        root: str | Path,
        *,
        expected_preflight_sha256: str,
        expected_run_sha256: str,
        expected_replay_sha256: str,
    ):
        calls.append(
            (
                "verified_subset",
                Path(root),
                expected_preflight_sha256,
                expected_run_sha256,
                expected_replay_sha256,
            )
        )
        return run, replay, source_rows

    def locked_gold(**kwargs: Any):
        calls.append(("locked_gold", kwargs))
        assert kwargs["allow_subset"] is True
        assert kwargs["source_rows"] == source_rows
        return gold_rows, _sha("subset gold population")

    monkeypatch.setattr(
        judge_cli.subset_cli,
        "read_verified_subset_run",
        verified_reader,
    )
    monkeypatch.setattr(judge_cli, "load_locked_typed_final_gold", locked_gold)
    args = SimpleNamespace(
        dataset=tmp_path / "locked.json",
        expected_subset_preflight_sha256=_sha("subset preflight"),
        expected_subset_replay_sha256=replay.sha256,
        expected_subset_run_sha256=run.sha256,
        gateway_url="http://sealed-gateway",
        judge_output_root=tmp_path / "sol-judge-v1",
        max_concurrency=3,
        model="sol-model",
        split=tmp_path / "locked-split.json",
        subset_root=tmp_path / "subset",
    )

    first = judge_cli._subset_preflight(args)
    artifact = read_sealed_json(Path(args.judge_output_root) / PREFLIGHT_NAME)
    prompts, rows = validate_preflight_artifact(artifact)
    second = judge_cli._subset_preflight(args)

    assert [call[0] for call in calls] == [
        "verified_subset",
        "locked_gold",
        "verified_subset",
        "locked_gold",
    ]
    assert first["created"] is True
    assert second["created"] is False
    assert first["preflight_sha256"] == second["preflight_sha256"] == artifact.sha256
    assert first["physical_provider_calls"] == 0
    assert first["judge_mode"] == "selected_subset"
    assert first["required_authorized_provider_calls"] == 27
    assert first["selected_ordinals"] == list(judge_cli.subset_cli.MISS_ORDINALS)
    assert artifact.payload["typed_final_run_sha256"] == run.sha256
    assert artifact.payload["typed_final_replay_sha256"] == replay.sha256
    assert artifact.payload["question_count"] == 100
    assert artifact.payload["selected_question_count"] == 27
    assert artifact.payload["judge_mode"] == "selected_subset"
    assert len(prompts) == len(rows) == 27
    assert tuple(row["ordinal"] for row in rows) == judge_cli.subset_cli.MISS_ORDINALS


def test_subset_preflight_rejects_population_drift_before_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_rows, _gold_rows = _subset_rows()
    run = _artifact("subset-run.json", "subset run")
    replay = _artifact("subset-replay.json", "subset replay")
    monkeypatch.setattr(
        judge_cli.subset_cli,
        "read_verified_subset_run",
        lambda *_args, **_kwargs: (run, replay, tuple(reversed(source_rows))),
    )
    monkeypatch.setattr(
        judge_cli,
        "load_locked_typed_final_gold",
        lambda **_kwargs: pytest.fail("gold opened for a drifted subset"),
    )

    with pytest.raises(
        judge_cli.TypedMemoryFinalJudgeError,
        match="subset judge source population changed",
    ):
        judge_cli._subset_preflight(
            SimpleNamespace(
                dataset=tmp_path / "locked.json",
                expected_subset_preflight_sha256=_sha("subset preflight"),
                expected_subset_replay_sha256=replay.sha256,
                expected_subset_run_sha256=run.sha256,
                gateway_url="http://sealed-gateway",
                judge_output_root=tmp_path / "sol-judge-v1",
                max_concurrency=3,
                model="sol-model",
                split=tmp_path / "locked-split.json",
                subset_root=tmp_path / "subset",
            )
        )


def test_parser_keeps_subset_authority_out_of_provider_runtime() -> None:
    parser = judge_cli._parser()
    subset = parser.parse_args(
        [
            "subset-preflight",
            "--expected-subset-preflight-sha256",
            "a" * 64,
            "--expected-subset-run-sha256",
            "b" * 64,
            "--expected-subset-replay-sha256",
            "c" * 64,
        ]
    )
    provider = parser.parse_args(
        [
            "provider-run",
            "--expected-judge-preflight-sha256",
            "d" * 64,
            "--authorized-provider-calls",
            "27",
        ]
    )

    assert subset.judge_output_root == judge_cli.DEFAULT_SUBSET_JUDGE_ROOT
    assert subset.subset_root == judge_cli.subset_cli.DEFAULT_OUTPUT
    assert not hasattr(provider, "subset_root")
    assert not hasattr(provider, "expected_subset_run_sha256")
    assert not hasattr(provider, "expected_subset_replay_sha256")
