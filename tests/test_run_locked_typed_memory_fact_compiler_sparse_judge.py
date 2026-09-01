from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.matched_eval.artifacts import read_sealed_json
from tools.matched_eval.typed_memory_final_judging import PREFLIGHT_NAME
import tools.run_locked_typed_memory_fact_compiler_sparse_judge as adapter


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=tmp_path / "gold.json",
        expected_answer_preflight_sha256=(
            adapter.EXPECTED_ANSWER_PREFLIGHT_SHA256
        ),
        expected_answer_replay_sha256=adapter.EXPECTED_ANSWER_REPLAY_SHA256,
        expected_answer_run_sha256=adapter.EXPECTED_ANSWER_RUN_SHA256,
        gateway_url="http://sealed-gateway",
        judge_output_root=tmp_path / "judge",
        max_concurrency=3,
        model="sol-model",
        expected_parent_judge_sha256=adapter.EXPECTED_PARENT_JUDGE_SHA256,
        parent_judge=tmp_path / "parent-judge.json",
        sparse_root=tmp_path / "sparse",
        split=tmp_path / "split.json",
    )


def test_sparse_judge_verifies_answers_then_parent_authority_and_seals_changed4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    source_rows = tuple(
        {
            "changed_from_parent": ordinal in adapter.CHANGED_ORDINALS,
            "ordinal": ordinal,
            "parent_prediction_sha256": f"{ordinal:064x}",
            "prediction": f"prediction {ordinal}",
        }
        for ordinal in adapter.sparse_cli.REMAINING_ORDINALS
    )
    run = SimpleNamespace(
        sha256=adapter.EXPECTED_ANSWER_RUN_SHA256,
        payload={},
    )
    replay = SimpleNamespace(
        sha256=adapter.EXPECTED_ANSWER_REPLAY_SHA256,
        payload={},
    )
    parent_judge = SimpleNamespace(
        sha256=adapter.EXPECTED_PARENT_JUDGE_SHA256,
        payload={
            "questions": [
                {
                    "correct": False,
                    "ordinal": row["ordinal"],
                    "prediction_sha256": row["parent_prediction_sha256"],
                }
                for row in source_rows
            ]
            + [
                {"correct": True, "ordinal": ordinal, "prediction_sha256": "f" * 64}
                for ordinal in (17, 74, 87)
            ]
        },
    )

    def verify_sparse(*_args, **kwargs):
        order.append("verify")
        assert kwargs["expected_preflight_sha256"] == (
            adapter.EXPECTED_ANSWER_PREFLIGHT_SHA256
        )
        assert kwargs["expected_run_sha256"] == adapter.EXPECTED_ANSWER_RUN_SHA256
        assert kwargs["expected_replay_sha256"] == (
            adapter.EXPECTED_ANSWER_REPLAY_SHA256
        )
        return run, replay, source_rows

    def open_gold(**kwargs):
        order.append("gold")
        assert order == ["verify", "authority", "gold"]
        assert tuple(row["ordinal"] for row in kwargs["source_rows"]) == (
            adapter.CHANGED_ORDINALS
        )
        assert kwargs["allow_subset"] is True
        return tuple(object() for _ in source_rows), "d" * 64

    def project(**kwargs):
        order.append("project")
        assert kwargs["run_artifact"] is run
        assert kwargs["replay_artifact_sha256"] == replay.sha256
        assert kwargs["mode"] == "selected_subset"
        rows = [
            {"ordinal": ordinal}
            for ordinal in adapter.CHANGED_ORDINALS
        ]
        return (
            {
                "format": "common-typed-final-judge-preflight",
                "judge_mode": "selected_subset",
                "prompt_rows": rows,
                "required_authorized_provider_calls": 4,
                "selected_question_count": 4,
            },
            (),
        )

    monkeypatch.setattr(
        adapter.sparse_cli,
        "read_verified_sparse_answer_run",
        verify_sparse,
    )
    monkeypatch.setattr(
        adapter,
        "read_sealed_json",
        lambda _path: order.append("authority") or parent_judge,
    )
    monkeypatch.setattr(adapter, "load_locked_typed_final_gold", open_gold)
    monkeypatch.setattr(adapter, "preflight_projection", project)

    result = adapter._preflight(_args(tmp_path))
    artifact = read_sealed_json(tmp_path / "judge" / PREFLIGHT_NAME)

    assert order == ["verify", "authority", "gold", "project"]
    assert result["physical_provider_calls"] == 0
    assert result["required_authorized_provider_calls"] == 4
    assert result["selected_question_count"] == 4
    assert result["inherited_incorrect_count"] == 20
    assert artifact.sha256 == result["judge_preflight_sha256"]
    assert [row["ordinal"] for row in artifact.payload["prompt_rows"]] == list(
        adapter.CHANGED_ORDINALS
    )


def test_sparse_verification_failure_never_opens_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter.sparse_cli,
        "read_verified_sparse_answer_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("changed")),
    )
    monkeypatch.setattr(
        adapter,
        "load_locked_typed_final_gold",
        lambda **_kwargs: pytest.fail("gold opened before sparse verification"),
    )

    with pytest.raises(ValueError, match="changed"):
        adapter._preflight(_args(tmp_path))
