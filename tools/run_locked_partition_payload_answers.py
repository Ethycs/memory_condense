#!/usr/bin/env python3
"""Run the isolated locked partition-scan-v2 direct-payload answer arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from tools import run_locked_query_payload_answers as payload_cli  # noqa: E402
from tools.matched_eval import live  # noqa: E402
from tools.matched_eval.contracts import MatchedEvalContractError  # noqa: E402
from tools.matched_eval.partition_payload_adapter import (  # noqa: E402
    DELTA_TIER,
    load_partition_payload_adapter,
)
from tools.matched_eval.population import EXPECTED_QUESTION_COUNT  # noqa: E402
from tools.matched_eval.query_fact_adapter import (  # noqa: E402
    QueryFactAdapterPopulation,
)
from tools.matched_eval.query_payload_live import (  # noqa: E402
    QueryPayloadAnswerPlan,
    build_query_payload_answer_plan,
)


DEFAULT_RETRIEVAL = payload_cli.DEFAULT_RETRIEVAL
DEFAULT_PARTITION_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/"
    "partition-scan-v2-r96"
)
DEFAULT_GENERATION = DEFAULT_PARTITION_ROOT / "retrieval-generation.json"
DEFAULT_PARENT_ROOT = payload_cli.DEFAULT_PARENT_ROOT
DEFAULT_OUTPUT = payload_cli.DEFAULT_CAMPAIGN_ROOT / (
    "s0-plus-partition-scan-v2-payload-v1"
)

EXPECTED_RETRIEVAL_SHA256 = payload_cli.EXPECTED_RETRIEVAL_SHA256
EXPECTED_SOURCE_POPULATION_ID = payload_cli.EXPECTED_SOURCE_POPULATION_ID
EXPECTED_GENERATION_SHA256 = (
    "671f0a3418364f544e61897c42569407805e827ae558980760289dae6b5cf388"
)
EXPECTED_ELIGIBILITY_SHA256 = (
    "748bd56a7efb8fd70d36bc96f099a53fc506469565577de9635908f6773bdee1"
)
EXPECTED_PARENT_ANSWER_RUN_SHA256 = payload_cli.EXPECTED_PARENT_ANSWER_RUN_SHA256


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--generation", type=Path, default=DEFAULT_GENERATION)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-retrieval-sha256",
        default=EXPECTED_RETRIEVAL_SHA256,
    )
    parser.add_argument(
        "--expected-source-population-id",
        default=EXPECTED_SOURCE_POPULATION_ID,
    )
    parser.add_argument(
        "--expected-generation-sha256",
        default=EXPECTED_GENERATION_SHA256,
    )
    parser.add_argument(
        "--expected-eligibility-sha256",
        default=EXPECTED_ELIGIBILITY_SHA256,
    )
    parser.add_argument(
        "--expected-parent-answer-run-sha256",
        default=EXPECTED_PARENT_ANSWER_RUN_SHA256,
    )
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    _add_inputs(preflight)
    provider = commands.add_parser("provider-run")
    _add_inputs(provider)
    provider.add_argument("--expected-answer-preflight-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, default=0)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)
    materialize = commands.add_parser("materialize")
    _add_inputs(materialize)
    materialize.add_argument("--expected-answer-preflight-sha256", required=True)
    replay = commands.add_parser("replay")
    _add_inputs(replay)
    replay.add_argument("--expected-answer-preflight-sha256", required=True)
    replay.add_argument("--expected-run-sha256", required=True)
    return parser


def _load_adapter(args: argparse.Namespace) -> QueryFactAdapterPopulation:
    return load_partition_payload_adapter(
        args.retrieval,
        generation_path=args.generation,
        expected_retrieval_sha256=args.expected_retrieval_sha256,
        expected_source_population_id=args.expected_source_population_id,
        expected_generation_sha256=args.expected_generation_sha256,
        expected_eligibility_manifest_sha256=args.expected_eligibility_sha256,
        expected_question_count=EXPECTED_QUESTION_COUNT,
    )


def _load_plan(args: argparse.Namespace) -> QueryPayloadAnswerPlan:
    adapter = _load_adapter(args)
    profile = live.execution_profile(live.RENDERER_ID)
    parent_root = Path(args.parent_root)
    parent = live._load_verified_s0_v2_answer_plane_for_population(
        population=adapter.source_population,
        run_path=parent_root / live.ANSWER_RUN_NAME,
        replay_path=parent_root / live.ANSWER_REPLAY_NAME,
        expected_run_sha256=args.expected_parent_answer_run_sha256,
        checkpoint_dir=parent_root / live.CHECKPOINT_DIR_NAME,
        max_concurrency=args.max_concurrency,
        profile=profile,
    )
    output = Path(args.output_root).resolve()
    if output in {parent_root.resolve(), Path(args.generation).parent.resolve()}:
        raise MatchedEvalContractError(
            "partition-payload output must be isolated from parent and generation roots"
        )
    plan = build_query_payload_answer_plan(
        adapter,
        parent,
        delta_tier=DELTA_TIER,
    )
    if len(plan.rows) != EXPECTED_QUESTION_COUNT:
        raise MatchedEvalContractError("locked partition-payload arm requires 100 rows")
    return plan


def _run(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_plan(args)
    if args.command == "preflight":
        return payload_cli._preflight_loaded_plan(plan, args)
    if args.command == "provider-run":
        return payload_cli._provider_loaded_plan(plan, args)
    if args.command == "materialize":
        return payload_cli._materialize_loaded_plan(plan, args)
    if args.command == "replay":
        return payload_cli._replay_loaded_plan(plan, args)
    raise AssertionError("unreachable command")  # pragma: no cover


def main(argv: list[str] | None = None) -> int:
    result = _run(_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
