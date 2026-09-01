from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from tools import run_r7_a1a_raw_retained_answer as a1a_cli
from tools import run_r7_after_union_a1 as a1_cli
from tools.matched_eval.artifacts import SealedArtifact, publish_sealed_json
from tools.matched_eval.r7_after_union_temporal_fail_open import (
    EFFECTIVE_DISPOSITIONS_FORMAT,
)


def _pair(
    root: Path,
    stem: str,
    payload: dict[str, object],
) -> tuple[SealedArtifact, SealedArtifact]:
    construction, _ = publish_sealed_json(root / f"{stem}.json", payload)
    replay, _ = publish_sealed_json(root / f"{stem}-replay.json", payload)
    return construction, replay


def _effective_inputs(tmp_path: Path) -> dict[str, SealedArtifact]:
    source, source_replay = _pair(tmp_path, "source", {"kind": "source"})
    temporal_a1, temporal_a1_replay = _pair(
        tmp_path,
        "temporal-a1",
        {"kind": "temporal-a1"},
    )
    effective, effective_replay = _pair(
        tmp_path,
        "effective",
        {
            "format": EFFECTIVE_DISPOSITIONS_FORMAT,
            "kind": "effective-dispositions",
        },
    )
    base, base_replay = _pair(
        tmp_path,
        "base-dispositions",
        {"kind": "base-dispositions"},
    )
    return {
        "base": base,
        "base_replay": base_replay,
        "effective": effective,
        "effective_replay": effective_replay,
        "source": source,
        "source_replay": source_replay,
        "temporal_a1": temporal_a1,
        "temporal_a1_replay": temporal_a1_replay,
    }


def _a1_args(
    tmp_path: Path,
    artifacts: dict[str, SealedArtifact],
) -> Namespace:
    return Namespace(
        base_dispositions=artifacts["base"].path,
        base_dispositions_replay=artifacts["base_replay"].path,
        compiler_outputs=None,
        dispositions=artifacts["effective"].path,
        dispositions_replay=artifacts["effective_replay"].path,
        expected_question_count=1,
        max_leaves_per_classifier_shard=48,
        max_leaves_per_shard=8,
        output_root=tmp_path / "a1-output",
        source_construction=artifacts["source"].path,
        source_replay=artifacts["source_replay"].path,
        temporal_a1_construction=artifacts["temporal_a1"].path,
        temporal_a1_replay=artifacts["temporal_a1_replay"].path,
    )


def _a1a_args(
    tmp_path: Path,
    artifacts: dict[str, SealedArtifact],
) -> Namespace:
    return Namespace(
        a1_construction=artifacts["temporal_a1"].path,
        a1_replay=artifacts["temporal_a1_replay"].path,
        base_dispositions=artifacts["base"].path,
        base_dispositions_replay=artifacts["base_replay"].path,
        dispositions=artifacts["effective"].path,
        dispositions_replay=artifacts["effective_replay"].path,
        expected_question_count=1,
        output_root=tmp_path / "a1a-output",
    )


@pytest.mark.parametrize(
    "missing",
    [
        "dispositions_replay",
        "temporal_a1_construction",
        "temporal_a1_replay",
        "base_dispositions",
        "base_dispositions_replay",
    ],
)
def test_a1_effective_runner_requires_every_parent_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    artifacts = _effective_inputs(tmp_path)
    args = _a1_args(tmp_path, artifacts)
    setattr(args, missing, None)
    monkeypatch.setattr(
        a1_cli,
        "build_r7_after_union_a1_payload",
        lambda *_args, **_kwargs: pytest.fail("invalid inputs reached A1 builder"),
    )

    with pytest.raises(ValueError, match="require replay, temporal A1"):
        a1_cli.run(args)


def test_a1_effective_runner_forwards_all_authenticated_parents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _effective_inputs(tmp_path)
    args = _a1_args(tmp_path, artifacts)
    captured: dict[str, dict[str, Any]] = {}
    payload = {
        "classifier_payload_class": "test-classifier",
        "classifier_request_count": 0,
        "compiler_payload_class": "test-compiler",
        "compiler_request_count": 0,
        "construction_status": "preflight_external_compilation_required",
        "missing_classifier_call_count": 0,
        "missing_compiler_call_count": 0,
        "missing_external_call_count": 0,
        "question_count": 1,
        "selected_leaf_count": 1,
        "selected_population_sha256": "a" * 64,
    }

    def fake_build(*_args: object, **kwargs: Any) -> dict[str, object]:
        captured["build"] = kwargs
        return payload

    def fake_replay(
        sealed: dict[str, object],
        *_args: object,
        **kwargs: Any,
    ) -> dict[str, object]:
        assert {
            "expected_question_count",
            "max_leaves_per_classifier_shard",
            "max_leaves_per_shard",
        }.isdisjoint(kwargs)
        captured["replay"] = kwargs
        return dict(sealed)

    monkeypatch.setattr(a1_cli, "build_r7_after_union_a1_payload", fake_build)
    monkeypatch.setattr(a1_cli, "replay_r7_after_union_a1_payload", fake_replay)

    result = a1_cli.run(args)

    assert result["replay_byte_identical"] is True
    assert captured["build"]["expected_question_count"] == 1
    assert captured["build"]["max_leaves_per_classifier_shard"] == 48
    assert captured["build"]["max_leaves_per_shard"] == 8
    for call in (captured["build"], captured["replay"]):
        assert call["disposition_replay_payload"] == artifacts[
            "effective_replay"
        ].payload
        assert call["disposition_replay_artifact_sha256"] == artifacts[
            "effective_replay"
        ].sha256
        assert call["temporal_a1_payload"] == artifacts["temporal_a1"].payload
        assert call["temporal_a1_artifact_sha256"] == artifacts[
            "temporal_a1"
        ].sha256
        assert call["temporal_a1_replay_payload"] == artifacts[
            "temporal_a1_replay"
        ].payload
        assert call["temporal_a1_replay_artifact_sha256"] == artifacts[
            "temporal_a1_replay"
        ].sha256
        assert call["base_disposition_payload"] == artifacts["base"].payload
        assert call["base_disposition_artifact_sha256"] == artifacts["base"].sha256
        assert call["base_disposition_replay_payload"] == artifacts[
            "base_replay"
        ].payload
        assert call["base_disposition_replay_artifact_sha256"] == artifacts[
            "base_replay"
        ].sha256


@pytest.mark.parametrize(
    "missing",
    ["dispositions_replay", "base_dispositions", "base_dispositions_replay"],
)
def test_a1a_effective_runner_requires_replay_and_base_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    artifacts = _effective_inputs(tmp_path)
    args = _a1a_args(tmp_path, artifacts)
    setattr(args, missing, None)
    monkeypatch.setattr(
        a1a_cli,
        "build_r7_a1a_raw_retained_payload",
        lambda *_args, **_kwargs: pytest.fail("invalid inputs reached A1a builder"),
    )

    with pytest.raises(ValueError, match="effective dispositions require"):
        a1a_cli.run(args)


def test_a1a_effective_runner_requires_byte_identical_a1_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _effective_inputs(tmp_path)
    foreign_replay, _ = publish_sealed_json(
        tmp_path / "foreign-a1-replay.json",
        {"kind": "foreign-temporal-a1"},
    )
    args = _a1a_args(tmp_path, artifacts)
    args.a1_replay = foreign_replay.path
    monkeypatch.setattr(
        a1a_cli,
        "build_r7_a1a_raw_retained_payload",
        lambda *_args, **_kwargs: pytest.fail("invalid inputs reached A1a builder"),
    )

    with pytest.raises(ValueError, match="A1 v2 construction and replay"):
        a1a_cli.run(args)


def test_a1a_effective_runner_forwards_all_authenticated_parents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _effective_inputs(tmp_path)
    args = _a1a_args(tmp_path, artifacts)
    captured: dict[str, dict[str, Any]] = {}
    payload = {
        "construction_status": "sealed_prompt_preflight_ready",
        "control_prompt_request_count": 1,
        "density_totals": {
            "fixed_union_leaf_count": 3,
            "pruned_leaf_count": 1,
            "retained_leaf_count": 2,
        },
        "max_fixed_union_control_prompt_token_proxy": 100,
        "max_terminal_prompt_token_proxy": 80,
        "prompt_request_count": 1,
        "question_count": 1,
    }

    def fake_build(*_args: object, **kwargs: Any) -> dict[str, object]:
        captured["build"] = kwargs
        return payload

    def fake_replay(
        sealed: dict[str, object],
        *_args: object,
        **kwargs: Any,
    ) -> dict[str, object]:
        assert "expected_question_count" not in kwargs
        captured["replay"] = kwargs
        return dict(sealed)

    monkeypatch.setattr(
        a1a_cli,
        "build_r7_a1a_raw_retained_payload",
        fake_build,
    )
    monkeypatch.setattr(
        a1a_cli,
        "replay_r7_a1a_raw_retained_payload",
        fake_replay,
    )

    result = a1a_cli.run(args)

    assert result["replay_byte_identical"] is True
    assert captured["build"]["expected_question_count"] == 1
    for call in (captured["build"], captured["replay"]):
        assert call["a1_preflight_replay_payload"] == artifacts[
            "temporal_a1_replay"
        ].payload
        assert call["disposition_replay_payload"] == artifacts[
            "effective_replay"
        ].payload
        assert call["disposition_replay_artifact_sha256"] == artifacts[
            "effective_replay"
        ].sha256
        assert call["base_disposition_payload"] == artifacts["base"].payload
        assert call["base_disposition_artifact_sha256"] == artifacts["base"].sha256
        assert call["base_disposition_replay_payload"] == artifacts[
            "base_replay"
        ].payload
        assert call["base_disposition_replay_artifact_sha256"] == artifacts[
            "base_replay"
        ].sha256


def test_effective_overlay_cli_flags_are_exposed() -> None:
    a1_dests = {action.dest for action in a1_cli.build_parser()._actions}
    assert {
        "base_dispositions",
        "base_dispositions_replay",
        "dispositions_replay",
        "expected_question_count",
        "temporal_a1_construction",
        "temporal_a1_replay",
    } <= a1_dests

    a1a_dests = {action.dest for action in a1a_cli.build_parser()._actions}
    assert {
        "base_dispositions",
        "base_dispositions_replay",
        "dispositions_replay",
        "expected_question_count",
    } <= a1a_dests
