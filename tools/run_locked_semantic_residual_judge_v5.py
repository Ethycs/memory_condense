#!/usr/bin/env python3
"""Judge replay-authenticated locked semantic-residual V5 answers.

This thin identity adapter authenticates the sealed V5 selector answer and
its byte-identical replay before the audited full-100 Sol judge lifecycle
opens locked gold. Provider execution is retry-free, zero-state, and starts
only with a fresh V5-specific checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.eval._binary_judge_protocol import JUDGE_MAX_TOKENS  # noqa: E402
from memory_condense.eval.fast_completion_runtime import FastCompletionRuntime  # noqa: E402
from tools import run_locked_semantic_residual_candidate_verifier_v5 as answer_v5  # noqa: E402
from tools import run_locked_specialist_final_judge as base  # noqa: E402
from tools.matched_eval.artifacts import SealedArtifact  # noqa: E402
from tools.matched_eval.contracts import MatchedEvalContractError  # noqa: E402
from tools.matched_eval.typed_memory_final_judging import TypedFinalJudgeGoldRow  # noqa: E402


QUESTION_COUNT = 100
EXACT_ORDINALS = tuple(range(QUESTION_COUNT))

ANSWER_RUN_FORMAT = answer_v5.FORMAT
PREFLIGHT_FORMAT = "memory-condense-locked-semantic-residual-sol-judge-preflight-v5"
JUDGE_FORMAT = "memory-condense-locked-semantic-residual-sol-judge-v5"
SCORE_FORMAT = "memory-condense-locked-semantic-residual-sol-score-v5"

ANSWER_RUN_NAME = answer_v5.RUN_NAME
ANSWER_REPLAY_NAME = answer_v5.REPLAY_NAME
PREFLIGHT_NAME = "locked-semantic-residual-sol-judge-preflight-v5.json"
JUDGE_NAME = "locked-semantic-residual-semantic-judge-sol-v5.json"
JUDGE_REPLAY_NAME = "locked-semantic-residual-semantic-judge-sol-replay-v5.json"
SCORE_NAME = "locked-semantic-residual-score-v5.json"
SCORE_REPLAY_NAME = "locked-semantic-residual-score-replay-v5.json"
CHECKPOINT_DIR_NAME = "sol-locked-semantic-residual-judge-calls-v5"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSWER_ROOT = answer_v5.DEFAULT_OUTPUT
DEFAULT_JUDGE_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-semantic-residual-sol-judge-v5-r1"
)
DEFAULT_SOL_MODEL = base.DEFAULT_SOL_MODEL
DEFAULT_CALLER_MODEL = base.DEFAULT_CALLER_MODEL
DEFAULT_MAX_PROMPT_TOKENS = base.DEFAULT_MAX_PROMPT_TOKENS
TARGET_ACCURACY = base.TARGET_ACCURACY


class LockedSemanticResidualJudgeV5Error(MatchedEvalContractError):
    """Raised when the full-100 semantic-residual V5 judge contract changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSemanticResidualJudgeV5Error(message)


def build_runtime(
    artifact: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    output_root: str | Path,
    model: str,
    gateway_url: str,
    max_concurrency: int,
    client: Any | None,
) -> FastCompletionRuntime:
    _require(
        artifact.payload.get("model") == model == DEFAULT_SOL_MODEL
        and artifact.payload.get("gateway_url") == gateway_url
        and artifact.payload.get("max_concurrency") == max_concurrency
        and artifact.payload.get("required_authorized_provider_calls") == QUESTION_COUNT
        and len(prompts) == QUESTION_COUNT,
        "full-100 semantic-residual V5 judge runtime differs from sealed preflight",
    )
    return FastCompletionRuntime(
        checkpoint_dir=Path(output_root) / CHECKPOINT_DIR_NAME,
        prompt_population=prompts,
        model=model,
        client=client,
        max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
        max_new_tokens=JUDGE_MAX_TOKENS,
        max_concurrency=max_concurrency,
        retries=0,
        benchmark_provenance={
            "answer_run_sha256": artifact.payload["answer_run_sha256"],
            "arm": "locked_semantic_residual_sol_judge_v5",
            "authorized_unique_calls": QUESTION_COUNT,
            "experiment_format": JUDGE_FORMAT,
            "preflight_artifact_sha256": artifact.sha256,
        },
    )


_BASE_LOCK = base._VERSION_CONTRACT_LOCK  # noqa: SLF001


def _version_globals() -> dict[str, Any]:
    return {
        "ANSWER_REPLAY_NAME": ANSWER_REPLAY_NAME,
        "ANSWER_RUN_FORMAT": ANSWER_RUN_FORMAT,
        "ANSWER_RUN_NAME": ANSWER_RUN_NAME,
        "CHECKPOINT_DIR_NAME": CHECKPOINT_DIR_NAME,
        "DEFAULT_ANSWER_ROOT": DEFAULT_ANSWER_ROOT,
        "DEFAULT_CALLER_MODEL": DEFAULT_CALLER_MODEL,
        "DEFAULT_JUDGE_ROOT": DEFAULT_JUDGE_ROOT,
        "DEFAULT_MAX_PROMPT_TOKENS": DEFAULT_MAX_PROMPT_TOKENS,
        "DEFAULT_SOL_MODEL": DEFAULT_SOL_MODEL,
        "EXACT_ORDINALS": EXACT_ORDINALS,
        "JUDGE_FORMAT": JUDGE_FORMAT,
        "JUDGE_NAME": JUDGE_NAME,
        "JUDGE_REPLAY_NAME": JUDGE_REPLAY_NAME,
        "PREFLIGHT_FORMAT": PREFLIGHT_FORMAT,
        "PREFLIGHT_NAME": PREFLIGHT_NAME,
        "QUESTION_COUNT": QUESTION_COUNT,
        "SCORE_FORMAT": SCORE_FORMAT,
        "SCORE_NAME": SCORE_NAME,
        "SCORE_REPLAY_NAME": SCORE_REPLAY_NAME,
        "TARGET_ACCURACY": TARGET_ACCURACY,
    }


@contextmanager
def _v5_base_contract() -> Iterator[None]:
    with _BASE_LOCK:
        values = {**_version_globals(), "build_runtime": build_runtime}
        previous = {name: getattr(base, name) for name in values}
        for name, value in values.items():
            setattr(base, name, value)
        try:
            yield
        finally:
            for name, value in previous.items():
                setattr(base, name, value)


def validate_answer_run_artifact(artifact: SealedArtifact):
    with _v5_base_contract():
        return base.validate_answer_run_artifact(artifact)


def load_verified_answer_judge_source(**kwargs: Any):
    with _v5_base_contract():
        return base.load_verified_answer_judge_source(**kwargs)


def build_preflight_payload(**kwargs: Any):
    with _v5_base_contract():
        return base.build_preflight_payload(**kwargs)


def validate_preflight_artifact(artifact: SealedArtifact):
    with _v5_base_contract():
        return base.validate_preflight_artifact(artifact)


def materialization_payloads(*args: Any, **kwargs: Any):
    with _v5_base_contract():
        return base.materialization_payloads(*args, **kwargs)


def run_preflight(
    args: argparse.Namespace,
    *,
    source_loader: Callable[..., Any] | None = None,
    gold_loader: Callable[..., tuple[tuple[TypedFinalJudgeGoldRow, ...], str]] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if source_loader is not None:
        kwargs["source_loader"] = source_loader
    if gold_loader is not None:
        kwargs["gold_loader"] = gold_loader
    with _v5_base_contract():
        return base.run_preflight(args, **kwargs)


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_root = Path(args.judge_output_root) / CHECKPOINT_DIR_NAME
    _require(
        not checkpoint_root.exists(),
        "V5 full-100 judge provider-run requires a fresh checkpoint directory",
    )
    with _v5_base_contract():
        return base.run_provider(args)


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    with _v5_base_contract():
        return base.run_materialize(args)


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    with _v5_base_contract():
        return base.run_replay(args)


def build_parser() -> argparse.ArgumentParser:
    with _v5_base_contract():
        parser = base.build_parser()
    parser.description = __doc__
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = run_preflight(args)
    elif args.command == "provider-run":
        result = run_provider(args)
    elif args.command == "materialize":
        result = run_materialize(args)
    else:
        result = run_replay(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANSWER_REPLAY_NAME", "ANSWER_RUN_FORMAT", "ANSWER_RUN_NAME",
    "CHECKPOINT_DIR_NAME", "DEFAULT_ANSWER_ROOT", "DEFAULT_JUDGE_ROOT",
    "DEFAULT_SOL_MODEL", "EXACT_ORDINALS", "JUDGE_FORMAT", "JUDGE_NAME",
    "JUDGE_REPLAY_NAME", "LockedSemanticResidualJudgeV5Error",
    "PREFLIGHT_FORMAT", "PREFLIGHT_NAME", "QUESTION_COUNT", "SCORE_FORMAT",
    "SCORE_NAME", "SCORE_REPLAY_NAME", "build_parser", "build_preflight_payload",
    "build_runtime", "load_verified_answer_judge_source", "main",
    "materialization_payloads", "run_materialize", "run_preflight",
    "run_provider", "run_replay", "validate_answer_run_artifact",
    "validate_preflight_artifact",
]
