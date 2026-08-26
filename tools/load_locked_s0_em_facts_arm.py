#!/usr/bin/env python3
"""Strict, provider-free loader and replay publisher for the sealed EM arm.

The original EM runner predates the common retrieval-arm loader contract.  It
already has a complete zero-call reconstruction path; this adapter exposes that
path without changing the sealed answer artifact or its digest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_condense.eval._artifact_json import canonical_json_bytes
from tools._locked_em_repair_adapter import _read_canonical_artifact
from tools import run_locked_s0_em_facts_arm as em_arm


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
    parent_run_path: Path | None = None,
) -> argparse.Namespace:
    if source.get("format") != em_arm.RUN_FORMAT:
        raise ValueError("EM answer run format changed")
    if source.get("arm_label") != em_arm.ARM_LABEL:
        raise ValueError("loader accepts only S0_PLUS_EM_FACTS")
    parent_sha = source.get("s0_control_run_sha256")
    if not isinstance(parent_sha, str):
        raise ValueError("EM run omitted its sealed S0 parent digest")
    output_root = run_path.parent
    expected_checkpoint = output_root / "terra-answer-calls"
    if checkpoint_dir is not None and checkpoint_dir.resolve() != expected_checkpoint.resolve():
        raise ValueError("EM loader checkpoint must be the sealed answer-journal directory")
    if run_path.resolve() != (output_root / "run.json").resolve():
        raise ValueError("EM loader requires the canonical run.json location")
    return argparse.Namespace(
        phase="replay",
        retrieval=retrieval_path,
        expected_retrieval_sha256=expected_retrieval_sha256,
        baseline_answers=baseline_answers_path,
        expected_baseline_answers_sha256=expected_baseline_answers_sha256,
        s0_run=parent_run_path or _default_parent_path(run_path),
        expected_s0_run_sha256=parent_sha,
        output_root=output_root,
        expected_question_count=expected_question_count,
        gateway_url=em_arm.DEFAULT_GATEWAY_URL,
        model=em_arm.DEFAULT_MODEL,
        api_key_env="LITELLM_KEY",
        max_concurrency=max_concurrency,
        enable_provider=False,
        authorized_provider_calls=0,
    )


def _verify_replay(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """Mirror the EM zero-call replay without publishing another artifact."""

    plan = em_arm._build_answer_plan(args)
    source, digest = em_arm._read(em_arm._run_path(args))
    batch = em_arm._answer_batch(plan, args, client=None)
    if batch is not None and batch.usage.physical_calls:
        raise RuntimeError("EM answer replay unexpectedly made provider calls")
    expected = em_arm._run_artifact(plan, batch)
    if canonical_json_bytes(source) != canonical_json_bytes(expected):
        raise ValueError("EM run differs from its immutable journals")
    return source, digest


def load_verified_run(
    run_path: str | Path,
    *,
    expected_run_sha256: str,
    retrieval_path: str | Path = em_arm.DEFAULT_RETRIEVAL,
    baseline_answers_path: str | Path = em_arm.DEFAULT_BASELINE_ANSWERS,
    checkpoint_dir: str | Path | None = None,
    max_concurrency: int = 4,
    expected_question_count: int = em_arm.EXPECTED_QUESTION_COUNT,
    expected_retrieval_sha256: str = em_arm.EXPECTED_RETRIEVAL_SHA256,
    expected_baseline_answers_sha256: str = em_arm.EXPECTED_BASELINE_ANSWERS_SHA256,
) -> tuple[dict[str, Any], str]:
    """Reconstruct both EM journal populations and return the exact sealed run."""

    target = Path(run_path)
    source, source_sha = _read_canonical_artifact(
        target, expected_sha256=expected_run_sha256
    )
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
    verified, verified_sha = _verify_replay(args)
    if verified_sha != source_sha or canonical_json_bytes(verified) != canonical_json_bytes(source):
        raise RuntimeError("EM replay returned another answer artifact")
    return source, source_sha


def run_replay(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    target = Path(args.run)
    source, source_sha = _read_canonical_artifact(
        target, expected_sha256=args.expected_run_sha256
    )
    replay_args = _replay_args(
        target,
        source,
        retrieval_path=Path(args.retrieval),
        baseline_answers_path=Path(args.baseline_answers),
        checkpoint_dir=None if args.checkpoint_dir is None else Path(args.checkpoint_dir),
        max_concurrency=args.max_concurrency,
        expected_question_count=args.expected_question_count,
        expected_retrieval_sha256=args.expected_retrieval_sha256,
        expected_baseline_answers_sha256=args.expected_baseline_answers_sha256,
        parent_run_path=None if args.s0_run is None else Path(args.s0_run),
    )
    verified, verified_sha = _verify_replay(replay_args)
    if verified_sha != source_sha or canonical_json_bytes(verified) != canonical_json_bytes(source):
        raise RuntimeError("EM replay returned another answer artifact")
    replay_path = Path(args.run_replay or target.with_name("run-replay.json"))
    return verified, em_arm._publish(replay_path, verified)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("replay",))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expected-run-sha256", required=True)
    parser.add_argument("--run-replay", type=Path)
    parser.add_argument("--retrieval", type=Path, default=em_arm.DEFAULT_RETRIEVAL)
    parser.add_argument("--expected-retrieval-sha256", default=em_arm.EXPECTED_RETRIEVAL_SHA256)
    parser.add_argument("--baseline-answers", type=Path, default=em_arm.DEFAULT_BASELINE_ANSWERS)
    parser.add_argument("--expected-baseline-answers-sha256", default=em_arm.EXPECTED_BASELINE_ANSWERS_SHA256)
    parser.add_argument("--s0-run", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--expected-question-count", type=int, default=em_arm.EXPECTED_QUESTION_COUNT)
    parser.add_argument("--max-concurrency", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact, digest = run_replay(args)
    print(
        f"S0_PLUS_EM_FACTS zero-call replay {digest}; "
        f"questions={artifact['question_count']}; provider_calls=0",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "load_verified_run", "main", "run_replay"]
