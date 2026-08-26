#!/usr/bin/env python3
"""Strict provider-free loader for a sealed S0-plus-CAV-links answer run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_condense.eval._artifact_json import canonical_json_bytes
from tools import run_locked_s0_cav_links_arm as cav_arm


def _default_parent_path(run_path: Path) -> Path:
    return run_path.parent.parent / "s0-control-v1" / "run.json"


def _replay_args(
    run_path: Path,
    source: dict[str, Any],
    *,
    retrieval_path: Path,
    baseline_answers_path: Path,
    checkpoint_dir: Path | None,
    max_concurrency: int,
    expected_question_count: int,
    expected_retrieval_sha256: str,
    expected_baseline_answers_sha256: str,
) -> argparse.Namespace:
    if source.get("format") != cav_arm.RUN_FORMAT:
        raise ValueError("CAV answer run format changed")
    if source.get("arm_label") != cav_arm.ARM_LABEL:
        raise ValueError("loader accepts only S0_PLUS_CAV_LINKS")
    output_root = run_path.parent
    if run_path.resolve() != (output_root / "run.json").resolve():
        raise ValueError("CAV loader requires the canonical run.json location")
    expected_checkpoint = output_root / "terra-answer-calls"
    if (
        checkpoint_dir is not None
        and checkpoint_dir.resolve() != expected_checkpoint.resolve()
    ):
        raise ValueError(
            "CAV loader checkpoint must be the sealed answer-journal directory"
        )
    feature_sha = source.get("feature_artifact_sha256")
    parent_sha = source.get("s0_control_run_sha256")
    if not isinstance(feature_sha, str) or not isinstance(parent_sha, str):
        raise ValueError("CAV run omitted feature or S0 parent provenance")
    return argparse.Namespace(
        phase="replay",
        retrieval=retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        baseline_answers=baseline_answers_path,
        expected_baseline_answers_sha256=expected_baseline_answers_sha256,
        s0_run=_default_parent_path(run_path),
        s0_checkpoint_dir=None,
        expected_s0_run_sha256=parent_sha,
        output_root=output_root,
        features=output_root / "features.json",
        expected_features_sha256=feature_sha,
        model_dir=cav_arm.DEFAULT_MODEL_DIR,
        device="cuda",
        dtype="bfloat16",
        batch_size=8,
        event_cav=cav_arm.DEFAULT_EVENT_CAV,
        prefix_cav=cav_arm.DEFAULT_PREFIX_CAV,
        extraction_temperature=0.05,
        reinjection_temperature=0.05,
        alpha=1.0,
        expected_question_count=expected_question_count,
        gateway_url=cav_arm.DEFAULT_GATEWAY_URL,
        model=cav_arm.DEFAULT_MODEL,
        api_key_env="LITELLM_KEY",
        max_concurrency=max_concurrency,
        enable_feature_model=False,
        authorized_feature_encoder_calls=0,
        enable_provider=False,
        authorized_provider_calls=0,
    )


def load_verified_run(
    run_path: str | Path,
    *,
    expected_run_sha256: str,
    retrieval_path: str | Path = cav_arm.DEFAULT_RETRIEVAL,
    baseline_answers_path: str | Path = cav_arm.DEFAULT_BASELINE_ANSWERS,
    checkpoint_dir: str | Path | None = None,
    max_concurrency: int = 4,
    expected_question_count: int = cav_arm.EXPECTED_QUESTION_COUNT,
    expected_retrieval_sha256: str = cav_arm.EXPECTED_RETRIEVAL_SHA256,
    expected_baseline_answers_sha256: str = (
        cav_arm.EXPECTED_BASELINE_ANSWERS_SHA256
    ),
) -> tuple[dict[str, Any], str]:
    """Verify run/replay, features, S0, and answer journals with zero calls."""

    target = Path(run_path)
    source, source_sha = cav_arm._read(
        target,
        expected_sha256=expected_run_sha256,
    )
    replay, replay_sha = cav_arm._read(
        target.with_name("run-replay.json"),
        expected_sha256=expected_run_sha256,
    )
    if (
        replay_sha != source_sha
        or canonical_json_bytes(replay) != canonical_json_bytes(source)
    ):
        raise ValueError("CAV answer run/replay differ")
    args = _replay_args(
        target,
        source,
        retrieval_path=Path(retrieval_path),
        baseline_answers_path=Path(baseline_answers_path),
        checkpoint_dir=None if checkpoint_dir is None else Path(checkpoint_dir),
        max_concurrency=max_concurrency,
        expected_question_count=expected_question_count,
        expected_retrieval_sha256=expected_retrieval_sha256,
        expected_baseline_answers_sha256=expected_baseline_answers_sha256,
    )
    plan = cav_arm._build_answer_plan(args)
    batch = cav_arm._answer_batch(plan, args, client=None)
    if batch is not None and (
        batch.usage.physical_calls
        or batch.usage.checkpoint_hits != plan.unique_calls
    ):
        raise RuntimeError("CAV loader did not consume the exact sealed journals")
    expected = cav_arm._run_artifact(plan, batch)
    if canonical_json_bytes(expected) != canonical_json_bytes(source):
        raise ValueError("CAV run differs from zero-call reconstruction")
    return source, source_sha


__all__ = ["load_verified_run"]
