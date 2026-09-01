#!/usr/bin/env python3
"""Judge the replay-authenticated locked full-100 v2 answer population.

This is a version adapter over the audited full-100 v1 Sol lifecycle.  It
changes only artifact identities, source-answer format, and runtime provenance.
The shared engine authenticates the complete answer run and byte-identical
replay before opening locked gold, seals exactly 100 unique judge prompts, and
requires checkpoint-only materialization and replay with zero retained state.
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
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionRuntime,
)
from tools import run_locked_specialist_final_answer_v2 as answer_v2  # noqa: E402
from tools import run_locked_specialist_final_judge as base  # noqa: E402
from tools.matched_eval.artifacts import SealedArtifact  # noqa: E402
from tools.matched_eval.contracts import MatchedEvalContractError  # noqa: E402
from tools.matched_eval.typed_memory_final_judging import (  # noqa: E402
    TypedFinalJudgeGoldRow,
)


QUESTION_COUNT = 100
EXACT_ORDINALS = tuple(range(QUESTION_COUNT))

ANSWER_RUN_FORMAT = answer_v2.FORMAT
PREFLIGHT_FORMAT = "memory-condense-locked-specialist-final-sol-judge-preflight-v2"
JUDGE_FORMAT = "memory-condense-locked-specialist-final-sol-judge-v2"
SCORE_FORMAT = "memory-condense-locked-specialist-final-sol-score-v2"

ANSWER_RUN_NAME = answer_v2.RUN_NAME
ANSWER_REPLAY_NAME = answer_v2.REPLAY_NAME
PREFLIGHT_NAME = "locked-specialist-final-sol-judge-preflight-v2.json"
JUDGE_NAME = "locked-specialist-final-semantic-judge-sol-v2.json"
JUDGE_REPLAY_NAME = "locked-specialist-final-semantic-judge-sol-replay-v2.json"
SCORE_NAME = "locked-specialist-final-score-v2.json"
SCORE_REPLAY_NAME = "locked-specialist-final-score-replay-v2.json"
CHECKPOINT_DIR_NAME = "sol-locked-specialist-final-judge-calls-v2"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSWER_ROOT = answer_v2.DEFAULT_OUTPUT
DEFAULT_JUDGE_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/locked-specialist-final-sol-judge-v2"
)
DEFAULT_SOL_MODEL = base.DEFAULT_SOL_MODEL
DEFAULT_CALLER_MODEL = base.DEFAULT_CALLER_MODEL
DEFAULT_MAX_PROMPT_TOKENS = base.DEFAULT_MAX_PROMPT_TOKENS
TARGET_ACCURACY = base.TARGET_ACCURACY


class LockedSpecialistFinalJudgeV2Error(MatchedEvalContractError):
    """Raised when the full-100 v2 judge contract changes."""


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedSpecialistFinalJudgeV2Error(message)


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
        and artifact.payload.get("required_authorized_provider_calls")
        == QUESTION_COUNT
        and len(prompts) == QUESTION_COUNT,
        "full-100 v2 judge runtime differs from sealed preflight",
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
            "arm": "locked_specialist_final_sol_judge_v2",
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
def _v2_base_contract() -> Iterator[None]:
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
    with _v2_base_contract():
        return base.validate_answer_run_artifact(artifact)


def load_verified_answer_judge_source(**kwargs: Any):
    with _v2_base_contract():
        return base.load_verified_answer_judge_source(**kwargs)


def build_preflight_payload(**kwargs: Any):
    with _v2_base_contract():
        return base.build_preflight_payload(**kwargs)


def validate_preflight_artifact(artifact: SealedArtifact):
    with _v2_base_contract():
        return base.validate_preflight_artifact(artifact)


def materialization_payloads(*args: Any, **kwargs: Any):
    with _v2_base_contract():
        return base.materialization_payloads(*args, **kwargs)


def run_preflight(
    args: argparse.Namespace,
    *,
    source_loader: Callable[..., Any] | None = None,
    gold_loader: Callable[..., tuple[tuple[TypedFinalJudgeGoldRow, ...], str]]
    | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if source_loader is not None:
        kwargs["source_loader"] = source_loader
    if gold_loader is not None:
        kwargs["gold_loader"] = gold_loader
    with _v2_base_contract():
        return base.run_preflight(args, **kwargs)


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    with _v2_base_contract():
        return base.run_provider(args)


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    with _v2_base_contract():
        return base.run_materialize(args)


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    with _v2_base_contract():
        return base.run_replay(args)


def build_parser() -> argparse.ArgumentParser:
    with _v2_base_contract():
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
    "ANSWER_REPLAY_NAME",
    "ANSWER_RUN_FORMAT",
    "ANSWER_RUN_NAME",
    "CHECKPOINT_DIR_NAME",
    "DEFAULT_ANSWER_ROOT",
    "DEFAULT_JUDGE_ROOT",
    "DEFAULT_SOL_MODEL",
    "EXACT_ORDINALS",
    "JUDGE_FORMAT",
    "JUDGE_NAME",
    "JUDGE_REPLAY_NAME",
    "LockedSpecialistFinalJudgeV2Error",
    "PREFLIGHT_FORMAT",
    "PREFLIGHT_NAME",
    "QUESTION_COUNT",
    "SCORE_FORMAT",
    "SCORE_NAME",
    "SCORE_REPLAY_NAME",
    "build_parser",
    "build_preflight_payload",
    "build_runtime",
    "load_verified_answer_judge_source",
    "main",
    "materialization_payloads",
    "run_materialize",
    "run_preflight",
    "run_provider",
    "run_replay",
    "validate_answer_run_artifact",
    "validate_preflight_artifact",
]
