#!/usr/bin/env python3
"""Checkpoint four Sol judgments for sealed missing-four v4 Terra answers.

The specialist-v2 judge remains the implementation of answer-seam replay
verification, gold loading, judge prompt construction, checkpoint journals,
materialization, scoring, and byte-identical replay.  This adapter supplies
only the four-question v4 contract, artifact names, and runtime provenance.

The ordering inherited from that engine is security-significant: the answer
run and replay are fully authenticated before the locked reference answers are
opened during Sol preflight.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionRuntime,
)
from tools import run_reduced_missing4_v4_answer as answer_v4  # noqa: E402
from tools import run_reduced_specialist_judge as base  # noqa: E402
from tools.matched_eval.artifacts import SealedArtifact  # noqa: E402
from tools.matched_eval.contracts import MatchedEvalContractError  # noqa: E402
from tools.matched_eval.query_answer_judging import JUDGE_MAX_TOKENS  # noqa: E402


EXACT_ORDINALS = tuple(answer_v4.EXPECTED_ORDINALS)
QUESTION_COUNT = len(EXACT_ORDINALS)

ANSWER_RUN_FORMAT = answer_v4.FORMAT
PREFLIGHT_FORMAT = "memory-condense-reduced-missing4-sol-judge-preflight-v4"
JUDGE_FORMAT = "memory-condense-reduced-missing4-sol-judge-v4"
SCORE_FORMAT = "memory-condense-reduced-missing4-sol-score-v4"

ANSWER_RUN_NAME = answer_v4.RUN_NAME
ANSWER_REPLAY_NAME = answer_v4.REPLAY_NAME
PREFLIGHT_NAME = "reduced-missing4-sol-judge-preflight-v4.json"
JUDGE_NAME = "reduced-missing4-semantic-judge-sol-v4.json"
JUDGE_REPLAY_NAME = "reduced-missing4-semantic-judge-sol-replay-v4.json"
SCORE_NAME = "reduced-missing4-score-v4.json"
SCORE_REPLAY_NAME = "reduced-missing4-score-replay-v4.json"
CHECKPOINT_DIR_NAME = "sol-reduced-missing4-judge-calls-v4"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSWER_ROOT = answer_v4.DEFAULT_OUTPUT
DEFAULT_JUDGE_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/reduced-missing4-sol-judge-v4"
)
DEFAULT_SOL_MODEL = base.DEFAULT_SOL_MODEL
DEFAULT_CALLER_MODEL = base.DEFAULT_CALLER_MODEL
DEFAULT_MAX_PROMPT_TOKENS = base.DEFAULT_MAX_PROMPT_TOKENS

AnswerSeamRow = base.AnswerSeamRow
load_locked_typed_final_gold = base.load_locked_typed_final_gold


class ReducedMissing4V4JudgeError(MatchedEvalContractError):
    """Raised when the v4 answer seam or judge protocol changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ReducedMissing4V4JudgeError(message)


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
    """Build the shared Sol checkpoint runtime with v4 provenance."""

    _require(
        artifact.payload.get("model") == model == DEFAULT_SOL_MODEL
        and artifact.payload.get("gateway_url") == gateway_url
        and artifact.payload.get("max_concurrency") == max_concurrency
        and len(prompts) == QUESTION_COUNT,
        "missing-four v4 judge runtime differs from sealed preflight",
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
            "arm": "reduced_missing4_sol_judge_v4",
            "authorized_unique_calls": QUESTION_COUNT,
            "experiment_format": JUDGE_FORMAT,
            "preflight_artifact_sha256": artifact.sha256,
        },
    )


_BASE_LOCK = RLock()


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
    }


@contextmanager
def _v4_base_contract() -> Iterator[None]:
    """Version the shared v2 judge stack and restore every global."""

    with _BASE_LOCK:
        values = _version_globals()
        module_values = {**values, "build_runtime": build_runtime}
        previous = {name: getattr(base, name) for name in module_values}
        for name, value in module_values.items():
            setattr(base, name, value)
        try:
            yield
        finally:
            for name, value in previous.items():
                setattr(base, name, value)


def load_verified_answer_seam(**kwargs: Any):
    with _v4_base_contract():
        return base.load_verified_answer_seam(**kwargs)


def build_preflight_payload(**kwargs: Any):
    with _v4_base_contract():
        return base.build_preflight_payload(**kwargs)


def validate_preflight_artifact(artifact: SealedArtifact):
    with _v4_base_contract():
        return base.validate_preflight_artifact(artifact)


def materialization_payloads(*args: Any, **kwargs: Any):
    with _v4_base_contract():
        return base.materialization_payloads(*args, **kwargs)


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    with _v4_base_contract():
        return base.run_preflight(args)


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    with _v4_base_contract():
        return base.run_provider(args)


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    with _v4_base_contract():
        return base.run_materialize(args)


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    with _v4_base_contract():
        return base.run_replay(args)


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--judge-output-root", type=Path, default=DEFAULT_JUDGE_ROOT)
    parser.add_argument("--model", default=DEFAULT_SOL_MODEL)
    parser.add_argument("--gateway-url", default=base.live.DEFAULT_GATEWAY_URL)
    parser.add_argument("--max-concurrency", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    _add_runtime_args(preflight)
    preflight.add_argument("--answer-root", type=Path, default=DEFAULT_ANSWER_ROOT)
    preflight.add_argument("--answer-run", type=Path)
    preflight.add_argument("--answer-replay", type=Path)
    preflight.add_argument("--expected-answer-run-sha256", required=True)
    preflight.add_argument("--expected-answer-replay-sha256", required=True)
    preflight.add_argument(
        "--dataset", type=Path, default=base.locked_judge_cli.DEFAULT_DATASET
    )
    preflight.add_argument(
        "--split", type=Path, default=base.spine_cli.DEFAULT_SPLIT
    )

    provider = commands.add_parser("provider-run")
    _add_runtime_args(provider)
    provider.add_argument("--expected-judge-preflight-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, required=True)
    provider.add_argument("--api-key-env", default=base.live.DEFAULT_API_KEY_ENV)

    materialize = commands.add_parser("materialize")
    _add_runtime_args(materialize)
    materialize.add_argument("--expected-judge-preflight-sha256", required=True)

    replay = commands.add_parser("replay")
    _add_runtime_args(replay)
    replay.add_argument("--expected-judge-preflight-sha256", required=True)
    replay.add_argument("--expected-judge-sha256", required=True)
    replay.add_argument("--expected-score-sha256", required=True)
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
    "ANSWER_REPLAY_NAME",
    "ANSWER_RUN_FORMAT",
    "ANSWER_RUN_NAME",
    "AnswerSeamRow",
    "CHECKPOINT_DIR_NAME",
    "DEFAULT_ANSWER_ROOT",
    "DEFAULT_JUDGE_ROOT",
    "DEFAULT_SOL_MODEL",
    "EXACT_ORDINALS",
    "JUDGE_FORMAT",
    "JUDGE_NAME",
    "JUDGE_REPLAY_NAME",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_NAME",
    "QUESTION_COUNT",
    "ReducedMissing4V4JudgeError",
    "SCORE_FORMAT",
    "SCORE_NAME",
    "SCORE_REPLAY_NAME",
    "build_parser",
    "build_preflight_payload",
    "build_runtime",
    "load_verified_answer_seam",
    "main",
    "materialization_payloads",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
    "validate_preflight_artifact",
]
