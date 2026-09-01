#!/usr/bin/env python3
"""Judge replay-verified specialist-v3 Terra answers with ten Sol calls.

The v3 answer run and its byte-identical replay are verified before locked gold
is opened.  Preflight then seals the ten standard benchmark judge prompts.
Provider execution writes checkpoints; materialization and replay are strictly
checkpoint-only.  The judge carries only the seven-field signed answer seam.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Mapping, Sequence

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from memory_condense.eval._binary_judge_protocol import JUDGE_MAX_TOKENS  # noqa: E402
from memory_condense.eval.fast_completion_runtime import (  # noqa: E402
    FastCompletionBatch,
    FastCompletionRuntime,
)
from tools import run_reduced_specialist_answer_v3 as answer_v3  # noqa: E402
from tools import run_reduced_specialist_judge as base  # noqa: E402
from tools.matched_eval import judging, live  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import (  # noqa: E402
    canonical_json_bytes,
    identity_sha256,
    require_sha256,
)
from tools.matched_eval.specialist_scoped_completion import (  # noqa: E402
    FORMAT as SCOPED_COMPLETION_FORMAT,
)


EXACT_ORDINALS = base.EXACT_ORDINALS
QUESTION_COUNT = len(EXACT_ORDINALS)

ANSWER_RUN_FORMAT = answer_v3.FORMAT
PREFLIGHT_FORMAT = "memory-condense-reduced-specialist-sol-judge-preflight-v3"
JUDGE_FORMAT = "memory-condense-reduced-specialist-sol-judge-v3"
SCORE_FORMAT = "memory-condense-reduced-specialist-sol-score-v3"

ANSWER_RUN_NAME = answer_v3.RUN_NAME
ANSWER_REPLAY_NAME = answer_v3.REPLAY_NAME
PREFLIGHT_NAME = "reduced-specialist-sol-judge-preflight-v3.json"
JUDGE_NAME = "reduced-specialist-semantic-judge-sol-v3.json"
JUDGE_REPLAY_NAME = "reduced-specialist-semantic-judge-sol-replay-v3.json"
SCORE_NAME = "reduced-specialist-score-v3.json"
SCORE_REPLAY_NAME = "reduced-specialist-score-replay-v3.json"
CHECKPOINT_DIR_NAME = "sol-reduced-specialist-judge-calls-v3"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSWER_ROOT = answer_v3.DEFAULT_OUTPUT
DEFAULT_JUDGE_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/reduced-specialist-sol-judge-v3"
)
DEFAULT_SOL_MODEL = base.DEFAULT_SOL_MODEL
DEFAULT_CALLER_MODEL = base.DEFAULT_CALLER_MODEL
DEFAULT_MAX_PROMPT_TOKENS = base.DEFAULT_MAX_PROMPT_TOKENS

AnswerSeamRow = base.AnswerSeamRow
ReducedSpecialistJudgeV3Error = base.ReducedSpecialistJudgeError
# Keep the familiar name for callers migrating directly from the v2 runner.
ReducedSpecialistJudgeError = ReducedSpecialistJudgeV3Error
load_locked_typed_final_gold = base.load_locked_typed_final_gold

_BASE_CONTRACT_LOCK = RLock()
_BASE_V3_GLOBALS = {
    "ANSWER_RUN_FORMAT": ANSWER_RUN_FORMAT,
    "ANSWER_RUN_NAME": ANSWER_RUN_NAME,
    "ANSWER_REPLAY_NAME": ANSWER_REPLAY_NAME,
    "CHECKPOINT_DIR_NAME": CHECKPOINT_DIR_NAME,
    "DEFAULT_ANSWER_ROOT": DEFAULT_ANSWER_ROOT,
    "DEFAULT_JUDGE_ROOT": DEFAULT_JUDGE_ROOT,
    "JUDGE_FORMAT": JUDGE_FORMAT,
    "JUDGE_NAME": JUDGE_NAME,
    "JUDGE_REPLAY_NAME": JUDGE_REPLAY_NAME,
    "PREFLIGHT_FORMAT": PREFLIGHT_FORMAT,
    "PREFLIGHT_NAME": PREFLIGHT_NAME,
    "SCORE_FORMAT": SCORE_FORMAT,
    "SCORE_NAME": SCORE_NAME,
    "SCORE_REPLAY_NAME": SCORE_REPLAY_NAME,
}


def _require(ok: object, message: str) -> None:
    if not ok:
        raise ReducedSpecialistJudgeV3Error(message)


@contextmanager
def _v3_base_contract() -> Iterator[None]:
    """Apply the immutable v3 names while reusing the audited v2 mechanics."""

    with _BASE_CONTRACT_LOCK:
        previous = {name: getattr(base, name) for name in _BASE_V3_GLOBALS}
        for name, value in _BASE_V3_GLOBALS.items():
            setattr(base, name, value)
        try:
            yield
        finally:
            for name, value in previous.items():
                setattr(base, name, value)


def _validate_v3_answer_envelope(payload: Mapping[str, Any]) -> None:
    """Require the scoped v3 answer protocol before exposing its tiny seam."""

    _require(
        payload.get("format") == ANSWER_RUN_FORMAT
        and payload.get("scoped_completion_format") == SCOPED_COMPLETION_FORMAT
        and payload.get("model") == answer_v3.DEFAULT_MODEL
        and payload.get("required_authorized_provider_calls") == QUESTION_COUNT
        and payload.get("gold_loaded") is False
        and payload.get("physical_provider_calls_during_materialization") == 0,
        "specialist v3 answer envelope changed",
    )
    require_sha256(
        payload.get("construction_artifact_sha256"),
        "specialist v3 construction artifact SHA-256",
    )
    require_sha256(
        payload.get("preflight_artifact_sha256"),
        "specialist v3 answer preflight SHA-256",
    )


def load_verified_answer_seam(
    *,
    answer_run_path: str | Path,
    answer_replay_path: str | Path,
    expected_answer_run_sha256: str,
    expected_answer_replay_sha256: str,
) -> tuple[SealedArtifact, SealedArtifact, tuple[AnswerSeamRow, ...]]:
    """Verify exact caller digests, replay identity, and the scoped-v3 envelope."""

    with _v3_base_contract():
        run, replay, rows = base.load_verified_answer_seam(
            answer_run_path=answer_run_path,
            answer_replay_path=answer_replay_path,
            expected_answer_run_sha256=expected_answer_run_sha256,
            expected_answer_replay_sha256=expected_answer_replay_sha256,
        )
    _validate_v3_answer_envelope(run.payload)
    return run, replay, rows


def build_preflight_payload(**kwargs: Any):
    """Reuse the standard judge preflight under v3 content-addressed names."""

    with _v3_base_contract():
        return base.build_preflight_payload(**kwargs)


def validate_preflight_artifact(artifact: SealedArtifact):
    with _v3_base_contract():
        return base.validate_preflight_artifact(artifact)


def materialization_payloads(
    preflight: SealedArtifact,
    prompt_rows: Sequence[Mapping[str, Any]],
    batch: FastCompletionBatch,
):
    with _v3_base_contract():
        return base.materialization_payloads(preflight, prompt_rows, batch)


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
    """Build the Sol runtime with v3-only checkpoint and provenance names."""

    _require(
        artifact.payload.get("model") == model == DEFAULT_SOL_MODEL
        and artifact.payload.get("gateway_url") == gateway_url
        and artifact.payload.get("max_concurrency") == max_concurrency
        and len(prompts) == QUESTION_COUNT,
        "specialist v3 judge runtime differs from sealed preflight",
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
            "arm": "reduced_specialist_sol_judge_v3",
            "authorized_unique_calls": QUESTION_COUNT,
            "experiment_format": JUDGE_FORMAT,
            "preflight_artifact_sha256": artifact.sha256,
        },
    )


def _answer_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(args.answer_root)
    return (
        Path(args.answer_run or root / ANSWER_RUN_NAME),
        Path(args.answer_replay or root / ANSWER_REPLAY_NAME),
    )


def _gold_source_rows(rows: Sequence[AnswerSeamRow]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "dated_question_sha256": row.dated_question_sha256,
            "ordinal": row.ordinal,
            "question_id": row.question_id,
            "question_sha256": row.question_sha256,
        }
        for row in rows
    )


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    run_path, replay_path = _answer_paths(args)
    # This verification finishes before either locked input is opened.
    run, replay, answer_rows = load_verified_answer_seam(
        answer_run_path=run_path,
        answer_replay_path=replay_path,
        expected_answer_run_sha256=args.expected_answer_run_sha256,
        expected_answer_replay_sha256=args.expected_answer_replay_sha256,
    )
    gold_rows, gold_sha = load_locked_typed_final_gold(
        dataset_path=args.dataset,
        split_path=args.split,
        source_rows=_gold_source_rows(answer_rows),
        allow_subset=True,
    )
    payload, _prompts = build_preflight_payload(
        run=run,
        replay=replay,
        answer_rows=answer_rows,
        gold_rows=gold_rows,
        gold_population_sha256=gold_sha,
        model=args.model,
        gateway_url=args.gateway_url,
        max_concurrency=args.max_concurrency,
    )
    artifact, created = publish_sealed_json(
        Path(args.judge_output_root) / PREFLIGHT_NAME,
        payload,
    )
    return {
        "answer_run_sha256": run.sha256,
        "created": created,
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": QUESTION_COUNT,
        "selected_ordinals": list(EXACT_ORDINALS),
    }


def _read_preflight(
    output_root: Path,
    expected_sha256: str,
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    artifact = read_sealed_json(output_root / PREFLIGHT_NAME)
    _require(
        artifact.sha256
        == require_sha256(expected_sha256, "expected v3 judge preflight SHA-256"),
        "specialist v3 judge preflight SHA-256 changed",
    )
    prompts, rows = validate_preflight_artifact(artifact)
    return artifact, prompts, rows


def _run_batch(
    artifact: SealedArtifact,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    args: argparse.Namespace,
    client: Any | None,
) -> FastCompletionBatch:
    runtime = build_runtime(
        artifact,
        prompts,
        output_root=args.judge_output_root,
        model=args.model,
        gateway_url=args.gateway_url,
        max_concurrency=args.max_concurrency,
        client=client,
    )
    try:
        return runtime.run()
    finally:
        runtime.close()


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    artifact, prompts, _rows = _read_preflight(
        Path(args.judge_output_root),
        args.expected_judge_preflight_sha256,
    )
    _require(
        args.enable_provider is True
        and args.authorized_provider_calls == QUESTION_COUNT
        and artifact.payload.get("required_authorized_provider_calls")
        == QUESTION_COUNT,
        "specialist v3 Sol judge requires exact authorization for 10 calls",
    )
    _require(
        artifact.payload.get("model") == args.model == DEFAULT_SOL_MODEL
        and artifact.payload.get("gateway_url") == args.gateway_url
        and artifact.payload.get("max_concurrency") == args.max_concurrency,
        "specialist v3 Sol judge runtime differs from sealed preflight",
    )
    load_dotenv()
    api_key = os.environ.get(args.api_key_env, "").strip()
    _require(bool(api_key), f"provider API key is empty: {args.api_key_env}")
    client = judging._make_provider_client(api_key, args.gateway_url)  # noqa: SLF001
    try:
        batch = _run_batch(artifact, prompts, args=args, client=client)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    _require(
        batch.usage.physical_calls + batch.usage.checkpoint_hits == QUESTION_COUNT,
        "specialist v3 Sol journal population changed after authorization",
    )
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": QUESTION_COUNT,
    }


def run_materialize(args: argparse.Namespace) -> dict[str, Any]:
    artifact, prompts, rows = _read_preflight(
        Path(args.judge_output_root),
        args.expected_judge_preflight_sha256,
    )
    batch = _run_batch(artifact, prompts, args=args, client=None)
    judge_payload, score_payload = materialization_payloads(artifact, rows, batch)
    root = Path(args.judge_output_root)
    judge_artifact, judge_created = publish_sealed_json(
        root / JUDGE_NAME, judge_payload
    )
    score_artifact, score_created = publish_sealed_json(
        root / SCORE_NAME, score_payload
    )
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "correct": score_payload["correct"],
        "judge_created": judge_created,
        "judge_sha256": judge_artifact.sha256,
        "physical_provider_calls": 0,
        "score_created": score_created,
        "score_sha256": score_artifact.sha256,
    }


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    artifact, prompts, rows = _read_preflight(
        Path(args.judge_output_root),
        args.expected_judge_preflight_sha256,
    )
    batch = _run_batch(artifact, prompts, args=args, client=None)
    expected_judge, expected_score = materialization_payloads(artifact, rows, batch)
    root = Path(args.judge_output_root)
    judge = read_sealed_json(root / JUDGE_NAME)
    score = read_sealed_json(root / SCORE_NAME)
    _require(
        judge.sha256
        == require_sha256(args.expected_judge_sha256, "expected v3 judge SHA-256")
        and score.sha256
        == require_sha256(args.expected_score_sha256, "expected v3 score SHA-256")
        and canonical_json_bytes(judge.payload) == canonical_json_bytes(expected_judge)
        and canonical_json_bytes(score.payload) == canonical_json_bytes(expected_score),
        "specialist v3 judge materialization differs from checkpoint replay",
    )
    judge_replay, _ = publish_sealed_json(
        root / JUDGE_REPLAY_NAME, expected_judge
    )
    score_replay, _ = publish_sealed_json(
        root / SCORE_REPLAY_NAME, expected_score
    )
    _require(
        judge_replay.sha256 == judge.sha256 and score_replay.sha256 == score.sha256,
        "specialist v3 judge replay is not byte-identical",
    )
    return {
        "byte_identical": True,
        "judge_replay_sha256": judge_replay.sha256,
        "physical_provider_calls": 0,
        "score_replay_sha256": score_replay.sha256,
    }


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--judge-output-root", type=Path, default=DEFAULT_JUDGE_ROOT)
    parser.add_argument("--model", default=DEFAULT_SOL_MODEL)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)
    parser.add_argument("--max-concurrency", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    _add_runtime_args(preflight)
    preflight.add_argument("--answer-root", type=Path, default=DEFAULT_ANSWER_ROOT)
    preflight.add_argument("--answer-run", type=Path)
    preflight.add_argument("--answer-replay", type=Path)
    preflight.add_argument("--expected-answer-run-sha256", required=True)
    preflight.add_argument("--expected-answer-replay-sha256", required=True)
    preflight.add_argument(
        "--dataset", type=Path, default=base.locked_judge_cli.DEFAULT_DATASET
    )
    preflight.add_argument("--split", type=Path, default=base.spine_cli.DEFAULT_SPLIT)

    provider = subparsers.add_parser("provider-run")
    _add_runtime_args(provider)
    provider.add_argument("--expected-judge-preflight-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, required=True)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)

    materialize = subparsers.add_parser("materialize")
    _add_runtime_args(materialize)
    materialize.add_argument("--expected-judge-preflight-sha256", required=True)

    replay = subparsers.add_parser("replay")
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
    "ReducedSpecialistJudgeError",
    "ReducedSpecialistJudgeV3Error",
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
