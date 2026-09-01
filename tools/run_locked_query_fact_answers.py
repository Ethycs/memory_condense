#!/usr/bin/env python3
"""Run the locked query-expansion routed-fact answer arm in split phases."""

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

from tools.matched_eval import live  # noqa: E402
from tools.matched_eval.artifacts import read_sealed_json  # noqa: E402
from tools.matched_eval.contracts import MatchedEvalContractError, require_sha256  # noqa: E402
from tools.matched_eval.population import (  # noqa: E402
    EXPECTED_QUESTION_COUNT,
    EXPECTED_RETRIEVAL_SHA256,
)
from tools.matched_eval.query_fact_adapter import (  # noqa: E402
    QueryFactAdapterPopulation,
    load_query_fact_population,
)
from tools.matched_eval.query_fact_answer_live import (  # noqa: E402
    ANSWER_RUN_NAME,
    CHECKPOINT_DIR_NAME,
    QueryFactAnswerPlan,
    build_query_fact_answer_plan,
    load_query_fact_answer_provider_journals,
    load_query_fact_answer_provider_population,
    load_verified_query_fact_compression,
    materialize_query_fact_answers,
    preflight_query_fact_answers,
    replay_query_fact_answers,
    run_sealed_query_fact_answer_provider,
)


DEFAULT_STORE_ROOT = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"
)
DEFAULT_RETRIEVAL = DEFAULT_STORE_ROOT / "retrieval.json"
DEFAULT_CAMPAIGN_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/"
    "matched-eval-spine-v2"
)
DEFAULT_QUERY_ROOT = DEFAULT_CAMPAIGN_ROOT / "s0-plus-query-expansion-v1"
DEFAULT_QUERY_PREFLIGHT = DEFAULT_QUERY_ROOT / "query-expansion-preflight.json"
DEFAULT_QUERY_RUN = DEFAULT_QUERY_ROOT / "query-expansion-run.json"
DEFAULT_COMPRESSION_ROOT = (
    DEFAULT_CAMPAIGN_ROOT / "s0-plus-query-expansion-routed-facts-v1"
)
DEFAULT_PARENT_ROOT = DEFAULT_CAMPAIGN_ROOT / "s0-control-v2"
DEFAULT_OUTPUT = (
    DEFAULT_CAMPAIGN_ROOT / "s0-plus-query-expansion-routed-fact-answers-v1"
)

EXPECTED_SOURCE_POPULATION_ID = (
    "886e14025a0aedf5a9ba673be8ffc9183acc080b97645adc2b6dd003019438bf"
)
EXPECTED_QUERY_PREFLIGHT_SHA256 = (
    "dc357e4a4e946c541ca5cb278824c376692ba4e4a97a5947c5b18e8da86c5487"
)
EXPECTED_QUERY_RUN_SHA256 = (
    "68f7c0c073c405e33cf019c75e69db1ee5be9b9f3dd84f13cd5a427e6508ba07"
)
EXPECTED_QUERY_POPULATION_ID = (
    "5030a5ae9ce83be7ae39ad290b492db278c8f090303730766db21edecae33b5e"
)
EXPECTED_QUERY_PROMPT_POPULATION_SHA256 = (
    "c88a09f1817404d5f29e0cca77fdb260b1479bf004bb8339d543376a3741c02d"
)
EXPECTED_PARENT_ANSWER_RUN_SHA256 = (
    "1a2545655d4a5e2061dc1b80efae39c7f8c70f5dc394f36c97d1312f70f39d8a"
)
EXPECTED_COMPRESSION_SHA256 = (
    "6285330940844055f6d29af97b3febbd97848f5b7b2fd4fe042cbfbb2907b6b0"
)
EXPECTED_COMPRESSION_RUNTIME_LEDGER_SHA256 = (
    "cf7e4f7783876cb37e9b6eba9942a06e3141c1dbf34e4c42bce70c44e701aae0"
)


def _add_plan_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--query-preflight", type=Path, default=DEFAULT_QUERY_PREFLIGHT)
    parser.add_argument("--query-run", type=Path, default=DEFAULT_QUERY_RUN)
    parser.add_argument("--compression-root", type=Path, default=DEFAULT_COMPRESSION_ROOT)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-retrieval-sha256", default=EXPECTED_RETRIEVAL_SHA256)
    parser.add_argument("--expected-source-population-id", default=EXPECTED_SOURCE_POPULATION_ID)
    parser.add_argument("--expected-query-preflight-sha256", default=EXPECTED_QUERY_PREFLIGHT_SHA256)
    parser.add_argument("--expected-query-run-sha256", default=EXPECTED_QUERY_RUN_SHA256)
    parser.add_argument("--expected-query-population-id", default=EXPECTED_QUERY_POPULATION_ID)
    parser.add_argument(
        "--expected-query-prompt-population-sha256",
        default=EXPECTED_QUERY_PROMPT_POPULATION_SHA256,
    )
    parser.add_argument("--expected-compression-sha256", default=EXPECTED_COMPRESSION_SHA256)
    parser.add_argument(
        "--expected-compression-runtime-ledger-sha256",
        default=EXPECTED_COMPRESSION_RUNTIME_LEDGER_SHA256,
    )
    parser.add_argument("--expected-parent-answer-run-sha256", default=EXPECTED_PARENT_ANSWER_RUN_SHA256)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)


def _add_provider_inputs(parser: argparse.ArgumentParser) -> None:
    # Deliberately no retrieval, store, query, compression, parent, or gold
    # argument is accepted by the network-enabled phase.
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-answer-preflight-sha256", required=True)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--authorized-provider-calls", type=int, default=0)
    parser.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    _add_plan_inputs(preflight)
    provider = subparsers.add_parser("provider-run")
    _add_provider_inputs(provider)
    materialize = subparsers.add_parser("materialize")
    _add_plan_inputs(materialize)
    materialize.add_argument("--expected-answer-preflight-sha256", required=True)
    replay = subparsers.add_parser("replay")
    _add_plan_inputs(replay)
    replay.add_argument("--expected-answer-preflight-sha256", required=True)
    replay.add_argument("--expected-run-sha256", required=True)
    return parser


def _load_adapter(args: argparse.Namespace) -> QueryFactAdapterPopulation:
    return load_query_fact_population(
        args.retrieval,
        query_preflight_path=args.query_preflight,
        query_run_path=args.query_run,
        expected_retrieval_sha256=args.expected_retrieval_sha256,
        expected_source_population_id=args.expected_source_population_id,
        expected_query_preflight_sha256=args.expected_query_preflight_sha256,
        expected_query_run_sha256=args.expected_query_run_sha256,
        expected_query_population_id=args.expected_query_population_id,
        expected_query_prompt_population_sha256=args.expected_query_prompt_population_sha256,
        expected_question_count=EXPECTED_QUESTION_COUNT,
    )


def _load_plan(args: argparse.Namespace) -> QueryFactAnswerPlan:
    adapter = _load_adapter(args)
    compression = load_verified_query_fact_compression(
        adapter,
        compression_root=args.compression_root,
        expected_compression_sha256=args.expected_compression_sha256,
        expected_runtime_ledger_sha256=args.expected_compression_runtime_ledger_sha256,
    )
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
    forbidden = {
        parent_root.resolve(),
        Path(args.query_run).parent.resolve(),
        Path(args.compression_root).resolve(),
    }
    if output in forbidden:
        raise MatchedEvalContractError(
            "query-fact answer output must be isolated from all source roots"
        )
    plan = build_query_fact_answer_plan(adapter, compression, parent)
    if len(plan.rows) != EXPECTED_QUESTION_COUNT:
        raise MatchedEvalContractError("locked query-fact answer arm requires 100 rows")
    return plan


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_plan(args)
    artifact = preflight_query_fact_answers(plan, output_root=args.output_root)
    return {
        "artifact": artifact.path.as_posix(),
        "fallback_count": plan.fallback_count,
        "gold_loaded": False,
        "max_prompt_token_proxy": max(
            (row.prompt_token_proxy for row in plan.submitted_rows), default=0
        ),
        "preflight_sha256": artifact.sha256,
        "provider_calls": 0,
        "question_count": len(plan.rows),
        "required_authorized_provider_calls": plan.required_calls,
    }


def _provider(args: argparse.Namespace) -> dict[str, Any]:
    expected = require_sha256(
        args.expected_answer_preflight_sha256,
        "expected query-fact answer preflight",
    )
    population = load_query_fact_answer_provider_population(
        output_root=args.output_root,
        expected_preflight_sha256=expected,
    )
    if (
        args.enable_provider is not True
        or args.authorized_provider_calls != population.required_calls
    ):
        raise MatchedEvalContractError(
            "provider-run requires exact authorization for "
            f"{population.required_calls} calls"
        )
    # Authorization precedes environment loading, client construction, and any
    # new request/response journal.
    load_dotenv()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise MatchedEvalContractError(f"provider API key is empty: {args.api_key_env}")
    client = live._make_provider_client(api_key, str(args.gateway_url))
    result = run_sealed_query_fact_answer_provider(
        population,
        enable_provider=True,
        authorized_provider_calls=population.required_calls,
        client=client,
        max_concurrency=args.max_concurrency,
        gateway_url=str(args.gateway_url),
    )
    checkpoint = Path(args.output_root) / CHECKPOINT_DIR_NAME
    return {
        "checkpoint_hits": result.checkpoint_hits,
        "command": "provider-run",
        "gold_loaded": False,
        "physical_provider_calls": result.physical_provider_calls,
        "preflight_sha256": result.preflight_artifact.sha256,
        "request_journal_count": len(tuple(checkpoint.glob("*.request.json"))),
        "response_journal_count": len(tuple(checkpoint.glob("*.response.json"))),
    }


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_plan(args)
    expected = require_sha256(
        args.expected_answer_preflight_sha256,
        "expected query-fact answer preflight",
    )
    journals = load_query_fact_answer_provider_journals(
        plan,
        output_root=args.output_root,
        expected_preflight_sha256=expected,
        max_concurrency=args.max_concurrency,
        gateway_url=str(args.gateway_url),
    )
    output = Path(args.output_root)
    existing = output / ANSWER_RUN_NAME
    if existing.exists():
        source = read_sealed_json(existing)
        verified = replay_query_fact_answers(
            plan,
            output_root=output,
            expected_preflight_sha256=expected,
            expected_run_sha256=source.sha256,
            max_concurrency=args.max_concurrency,
            gateway_url=str(args.gateway_url),
        )
        return {
            "checkpoint_hits": journals.checkpoint_hits,
            "changed_prediction_count": len(verified.changed_rows),
            "command": "materialize",
            "gold_loaded": False,
            "physical_provider_calls": 0,
            "run_sha256": verified.run_sha256,
            "runtime_ledger_sha256": verified.runtime_ledger_sha256,
            "terminal_run_replayed": True,
        }
    result = materialize_query_fact_answers(
        plan,
        output_root=output,
        expected_preflight_sha256=expected,
        completion_batch=journals.batch,
        gateway_url=str(args.gateway_url),
    )
    return {
        "checkpoint_hits": journals.checkpoint_hits,
        "command": "materialize",
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "run_sha256": result.answer_artifact.sha256,
        "runtime_ledger_sha256": result.runtime_ledger_artifact.sha256,
        "terminal_run_replayed": False,
    }


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_plan(args)
    verified = replay_query_fact_answers(
        plan,
        output_root=args.output_root,
        expected_preflight_sha256=args.expected_answer_preflight_sha256,
        expected_run_sha256=args.expected_run_sha256,
        max_concurrency=args.max_concurrency,
        gateway_url=str(args.gateway_url),
    )
    return {
        "changed_prediction_count": len(verified.changed_rows),
        "command": "replay",
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "replay_sha256": verified.replay_sha256,
        "run_sha256": verified.run_sha256,
        "runtime_ledger_sha256": verified.runtime_ledger_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    actions = {
        "preflight": _preflight,
        "provider-run": _provider,
        "materialize": _materialize,
        "replay": _replay,
    }
    result = actions[args.command](args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
