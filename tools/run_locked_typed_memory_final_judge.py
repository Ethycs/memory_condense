#!/usr/bin/env python3
"""Run full-100, changed-only, or sealed-subset Sol typed-memory judging."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from tools import run_locked_typed_memory_posthoc_miss_subset as subset_cli  # noqa: E402
from tools.matched_eval import judging, live  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    SealedArtifact,
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import require_sha256  # noqa: E402
from tools.matched_eval.typed_memory_final_judging import (  # noqa: E402
    JUDGE_NAME,
    PREFLIGHT_NAME,
    REPLAY_NAME,
    SCORE_NAME,
    SCORE_REPLAY_NAME,
    TypedMemoryFinalJudgeError,
    build_runtime,
    load_locked_typed_final_gold,
    load_verified_typed_final_judge_source,
    materialization_projection,
    preflight_projection,
    validate_preflight_artifact,
)
from tools.run_locked_query_answer_judge import DEFAULT_DATASET  # noqa: E402
from tools.run_matched_eval_spine import DEFAULT_SPLIT  # noqa: E402


DEFAULT_TYPED_ROOT = Path("eval_results/matched_eval_100/typed-memory-final-v1")
DEFAULT_JUDGE_ROOT = DEFAULT_TYPED_ROOT / "sol-judge-v1"
DEFAULT_SUBSET_JUDGE_ROOT = subset_cli.DEFAULT_OUTPUT / "sol-judge-v1"


def _require(ok: object, message: str) -> None:
    if not ok:
        raise TypedMemoryFinalJudgeError(message)


def _read_preflight(
    root: Path,
    expected_sha256: str,
) -> tuple[
    SealedArtifact,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[dict[str, Any], ...],
]:
    artifact = read_sealed_json(root / PREFLIGHT_NAME)
    _require(
        artifact.sha256
        == require_sha256(expected_sha256, "expected typed-final judge preflight"),
        "typed-final judge preflight changed",
    )
    prompts, rows = validate_preflight_artifact(artifact)
    return artifact, prompts, rows


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    # The complete responder run/replay is fixed before benchmark gold opens.
    run, replay, source_rows = load_verified_typed_final_judge_source(
        args.typed_root,
        expected_run_sha256=args.expected_typed_run_sha256,
        expected_replay_sha256=args.expected_typed_replay_sha256,
    )
    gold_rows, gold_sha = load_locked_typed_final_gold(
        dataset_path=args.dataset,
        split_path=args.split,
        source_rows=source_rows,
    )
    payload, _prompts = preflight_projection(
        run_artifact=run,
        replay_artifact_sha256=replay.sha256,
        source_rows=source_rows,
        gold_rows=gold_rows,
        gold_population_sha256=gold_sha,
        mode=args.mode,
        model=args.model,
        gateway_url=args.gateway_url,
        max_concurrency=args.max_concurrency,
    )
    artifact, created = publish_sealed_json(
        Path(args.judge_output_root) / PREFLIGHT_NAME,
        payload,
    )
    return {
        "created": created,
        "judge_mode": args.mode,
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": payload[
            "required_authorized_provider_calls"
        ],
        "selected_question_count": payload["selected_question_count"],
        "typed_final_run_sha256": run.sha256,
    }


def _subset_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Seal standard judge prompts for the replay-verified miss subset.

    The subset answer run and its byte-identical replay are verified before
    benchmark gold is opened.  Only the returned stable judge seam is paired
    with the matching locked-gold rows; selection authority is not reopened.
    """

    run, replay, source_rows = subset_cli.read_verified_subset_run(
        args.subset_root,
        expected_preflight_sha256=args.expected_subset_preflight_sha256,
        expected_run_sha256=args.expected_subset_run_sha256,
        expected_replay_sha256=args.expected_subset_replay_sha256,
    )
    _require(
        len(source_rows) == subset_cli.SUBSET_QUESTION_COUNT
        and tuple(row.get("ordinal") for row in source_rows)
        == subset_cli.MISS_ORDINALS,
        "typed-final subset judge source population changed",
    )
    gold_rows, gold_sha = load_locked_typed_final_gold(
        dataset_path=args.dataset,
        split_path=args.split,
        source_rows=source_rows,
        allow_subset=True,
    )
    payload, _prompts = preflight_projection(
        run_artifact=run,
        replay_artifact_sha256=replay.sha256,
        source_rows=source_rows,
        gold_rows=gold_rows,
        gold_population_sha256=gold_sha,
        mode="selected_subset",
        model=args.model,
        gateway_url=args.gateway_url,
        max_concurrency=args.max_concurrency,
    )
    _require(
        payload.get("selected_question_count") == subset_cli.SUBSET_QUESTION_COUNT
        and tuple(
            row.get("ordinal") for row in payload.get("prompt_rows", ())
        )
        == subset_cli.MISS_ORDINALS,
        "typed-final subset judge prompt projection changed",
    )
    artifact, created = publish_sealed_json(
        Path(args.judge_output_root) / PREFLIGHT_NAME,
        payload,
    )
    return {
        "created": created,
        "judge_mode": "selected_subset",
        "physical_provider_calls": 0,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": payload[
            "required_authorized_provider_calls"
        ],
        "selected_ordinals": list(subset_cli.MISS_ORDINALS),
        "selected_question_count": payload["selected_question_count"],
        "subset_replay_sha256": replay.sha256,
        "subset_run_sha256": run.sha256,
    }


def _run_batch(
    artifact: SealedArtifact,
    prompts: tuple[tuple[dict[str, str], ...], ...],
    *,
    args: argparse.Namespace,
    client: Any | None,
):
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


def _provider(args: argparse.Namespace) -> dict[str, Any]:
    artifact, prompts, _rows = _read_preflight(
        Path(args.judge_output_root), args.expected_judge_preflight_sha256
    )
    required = len(prompts)
    _require(
        args.enable_provider is True
        and args.authorized_provider_calls == required
        and artifact.payload.get("required_authorized_provider_calls") == required,
        f"typed-final judge provider requires exact authorization for {required} calls",
    )
    # Authorization and immutable population checks precede environment/client
    # access and every checkpoint mutation.
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
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "physical_provider_calls": batch.usage.physical_calls,
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": required,
    }


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    artifact, prompts, rows = _read_preflight(
        Path(args.judge_output_root), args.expected_judge_preflight_sha256
    )
    batch = _run_batch(artifact, prompts, args=args, client=None)
    judge, score = materialization_projection(artifact, rows, batch)
    judge_artifact, judge_created = publish_sealed_json(
        Path(args.judge_output_root) / JUDGE_NAME,
        judge,
    )
    score_artifact, score_created = publish_sealed_json(
        Path(args.judge_output_root) / SCORE_NAME,
        score,
    )
    return {
        "checkpoint_hits": batch.usage.checkpoint_hits,
        "correct": score["correct"],
        "judge_created": judge_created,
        "judge_sha256": judge_artifact.sha256,
        "physical_provider_calls": 0,
        "score_created": score_created,
        "score_sha256": score_artifact.sha256,
        "selected_question_count": score["selected_question_count"],
    }


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    artifact, prompts, rows = _read_preflight(
        Path(args.judge_output_root), args.expected_judge_preflight_sha256
    )
    batch = _run_batch(artifact, prompts, args=args, client=None)
    judge, score = materialization_projection(artifact, rows, batch)
    root = Path(args.judge_output_root)
    observed_judge = read_sealed_json(root / JUDGE_NAME)
    observed_score = read_sealed_json(root / SCORE_NAME)
    _require(
        observed_judge.sha256
        == require_sha256(args.expected_judge_sha256, "expected typed judge")
        and observed_score.sha256
        == require_sha256(args.expected_score_sha256, "expected typed judge score")
        and observed_judge.payload == judge
        and observed_score.payload == score,
        "typed-final judge materialization differs from checkpoint replay",
    )
    replay, _ = publish_sealed_json(root / REPLAY_NAME, judge)
    score_replay, _ = publish_sealed_json(root / SCORE_REPLAY_NAME, score)
    _require(
        replay.sha256 == observed_judge.sha256
        and score_replay.sha256 == observed_score.sha256,
        "typed-final judge replay is not byte-identical",
    )
    return {
        "byte_identical": True,
        "judge_replay_sha256": replay.sha256,
        "physical_provider_calls": 0,
        "score_replay_sha256": score_replay.sha256,
    }


def _add_runtime(
    parser: argparse.ArgumentParser,
    *,
    default_output_root: Path = DEFAULT_JUDGE_ROOT,
) -> None:
    parser.add_argument(
        "--judge-output-root", type=Path, default=default_output_root
    )
    parser.add_argument("--model", default=judging.DEFAULT_SOL_GATEWAY_MODEL)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)
    parser.add_argument("--max-concurrency", type=int, default=4)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    _add_runtime(preflight)
    preflight.add_argument("--typed-root", type=Path, default=DEFAULT_TYPED_ROOT)
    preflight.add_argument("--expected-typed-run-sha256", required=True)
    preflight.add_argument("--expected-typed-replay-sha256", required=True)
    preflight.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    preflight.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    preflight.add_argument(
        "--mode", choices=("full100", "changed_only"), default="full100"
    )
    subset_preflight = sub.add_parser(
        "subset-preflight",
        help="seal Sol judge prompts for the replay-verified posthoc miss subset",
    )
    _add_runtime(
        subset_preflight,
        default_output_root=DEFAULT_SUBSET_JUDGE_ROOT,
    )
    subset_preflight.add_argument(
        "--subset-root",
        type=Path,
        default=subset_cli.DEFAULT_OUTPUT,
    )
    subset_preflight.add_argument(
        "--expected-subset-preflight-sha256", required=True
    )
    subset_preflight.add_argument("--expected-subset-run-sha256", required=True)
    subset_preflight.add_argument(
        "--expected-subset-replay-sha256", required=True
    )
    subset_preflight.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    subset_preflight.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    provider = sub.add_parser("provider-run")
    _add_runtime(provider)
    provider.add_argument("--expected-judge-preflight-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, required=True)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)
    materialize = sub.add_parser("materialize")
    _add_runtime(materialize)
    materialize.add_argument("--expected-judge-preflight-sha256", required=True)
    replay = sub.add_parser("replay")
    _add_runtime(replay)
    replay.add_argument("--expected-judge-preflight-sha256", required=True)
    replay.add_argument("--expected-judge-sha256", required=True)
    replay.add_argument("--expected-score-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = _preflight(args)
    elif args.command == "subset-preflight":
        result = _subset_preflight(args)
    elif args.command == "provider-run":
        result = _provider(args)
    elif args.command == "materialize":
        result = _materialize(args)
    else:
        result = _replay(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
