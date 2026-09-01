#!/usr/bin/env python3
"""Run the locked structured query-operator refinement answer arm."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from tools import run_locked_query_payload_answers as payload_cli  # noqa: E402
from tools.matched_eval import live  # noqa: E402
from tools.matched_eval.artifacts import read_sealed_json  # noqa: E402
from tools.matched_eval.contracts import MatchedEvalContractError, require_sha256  # noqa: E402
from tools.matched_eval.population import EXPECTED_QUESTION_COUNT  # noqa: E402
from tools.matched_eval.query_operator_refinement_live import (  # noqa: E402
    ANSWER_RUN_NAME,
    CHECKPOINT_DIR_NAME,
    QueryOperatorRefinementPlan,
    build_query_operator_refinement_plan,
    load_query_operator_provider_journals,
    load_query_operator_provider_population,
    materialize_query_operator_refinement_answers,
    preflight_query_operator_refinement_answers,
    replay_query_operator_refinement_answers,
    run_sealed_query_operator_provider,
)
from tools.matched_eval.query_payload_live import replay_query_payload_answers  # noqa: E402


DEFAULT_DIRECT_ANSWER_ROOT = payload_cli.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    payload_cli.DEFAULT_CAMPAIGN_ROOT / "s0-plus-query-operator-refinement-v1"
)
EXPECTED_DIRECT_ANSWER_PREFLIGHT_SHA256 = (
    "c5c705470259743ce1fb7e07bd72374ada32352f5240e44d06a17cf450f7ac9d"
)
EXPECTED_DIRECT_ANSWER_RUN_SHA256 = (
    "ab271ccb1bb830346fea64c9b11f3c7d504f048cc1ba392da39b177869106c6d"
)


def _add_plan_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retrieval", type=Path, default=payload_cli.DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--query-preflight", type=Path, default=payload_cli.DEFAULT_QUERY_PREFLIGHT
    )
    parser.add_argument("--query-run", type=Path, default=payload_cli.DEFAULT_QUERY_RUN)
    parser.add_argument("--parent-root", type=Path, default=payload_cli.DEFAULT_PARENT_ROOT)
    parser.add_argument(
        "--direct-answer-root", type=Path, default=DEFAULT_DIRECT_ANSWER_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-retrieval-sha256",
        default=payload_cli.EXPECTED_RETRIEVAL_SHA256,
    )
    parser.add_argument(
        "--expected-source-population-id",
        default=payload_cli.EXPECTED_SOURCE_POPULATION_ID,
    )
    parser.add_argument(
        "--expected-query-preflight-sha256",
        default=payload_cli.EXPECTED_QUERY_PREFLIGHT_SHA256,
    )
    parser.add_argument(
        "--expected-query-run-sha256",
        default=payload_cli.EXPECTED_QUERY_RUN_SHA256,
    )
    parser.add_argument(
        "--expected-query-population-id",
        default=payload_cli.EXPECTED_QUERY_POPULATION_ID,
    )
    parser.add_argument(
        "--expected-query-prompt-population-sha256",
        default=payload_cli.EXPECTED_QUERY_PROMPT_POPULATION_SHA256,
    )
    parser.add_argument(
        "--expected-parent-answer-run-sha256",
        default=payload_cli.EXPECTED_PARENT_ANSWER_RUN_SHA256,
    )
    parser.add_argument(
        "--expected-direct-answer-preflight-sha256",
        default=EXPECTED_DIRECT_ANSWER_PREFLIGHT_SHA256,
    )
    parser.add_argument(
        "--expected-direct-answer-run-sha256",
        default=EXPECTED_DIRECT_ANSWER_RUN_SHA256,
    )
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)


def _add_provider_inputs(parser: argparse.ArgumentParser) -> None:
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


def _load_plan(args: argparse.Namespace) -> QueryOperatorRefinementPlan:
    direct_root = Path(args.direct_answer_root)
    output = Path(args.output_root).resolve()
    forbidden = {
        direct_root.resolve(),
        Path(args.parent_root).resolve(),
        Path(args.query_run).parent.resolve(),
    }
    if output in forbidden:
        raise MatchedEvalContractError(
            "operator-refinement output must be isolated from all source planes"
        )
    payload_args = argparse.Namespace(**vars(args))
    payload_args.output_root = direct_root
    direct_plan = payload_cli._load_plan(payload_args)
    direct_plane = replay_query_payload_answers(
        direct_plan,
        output_root=direct_root,
        expected_preflight_sha256=args.expected_direct_answer_preflight_sha256,
        expected_run_sha256=args.expected_direct_answer_run_sha256,
        max_concurrency=args.max_concurrency,
        gateway_url=str(args.gateway_url),
    )
    plan = build_query_operator_refinement_plan(direct_plan, direct_plane)
    if len(plan.rows) != EXPECTED_QUESTION_COUNT:
        raise MatchedEvalContractError("locked operator-refinement arm requires 100 rows")
    return plan


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_plan(args)
    artifact = preflight_query_operator_refinement_answers(
        plan, output_root=args.output_root
    )
    route_counts = Counter(row.route.style.value for row in plan.rows)
    return {
        "artifact": artifact.path.as_posix(),
        "dropped_evidence_count": plan.dropped_evidence_count,
        "dropped_row_count": plan.dropped_row_count,
        "fallback_count": plan.fallback_count,
        "gold_loaded": False,
        "max_prompt_token_proxy": max(
            (row.prompt_token_proxy for row in plan.submitted_rows), default=0
        ),
        "preflight_sha256": artifact.sha256,
        "provider_calls": 0,
        "question_count": len(plan.rows),
        "required_authorized_provider_calls": plan.required_calls,
        "route_counts": dict(sorted(route_counts.items())),
    }


def _provider(args: argparse.Namespace) -> dict[str, Any]:
    # The network phase accepts only the sealed prompt population. Exact call
    # authorization is checked before environment loading or client creation.
    population = load_query_operator_provider_population(
        output_root=args.output_root,
        expected_preflight_sha256=args.expected_answer_preflight_sha256,
    )
    if (
        args.enable_provider is not True
        or args.authorized_provider_calls != population.required_calls
    ):
        raise MatchedEvalContractError(
            f"provider-run requires exact authorization for {population.required_calls} calls"
        )
    load_dotenv()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise MatchedEvalContractError(f"provider API key is empty: {args.api_key_env}")
    client = live._make_provider_client(api_key, str(args.gateway_url))
    try:
        result = run_sealed_query_operator_provider(
            population,
            enable_provider=True,
            authorized_provider_calls=population.required_calls,
            client=client,
            max_concurrency=args.max_concurrency,
            gateway_url=str(args.gateway_url),
        )
    except BaseException:
        close = getattr(client, "close", None)
        if callable(close):
            close()
        raise
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
        "expected operator-refinement preflight",
    )
    journals = load_query_operator_provider_journals(
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
        verified = replay_query_operator_refinement_answers(
            plan,
            output_root=output,
            expected_preflight_sha256=expected,
            expected_run_sha256=source.sha256,
            max_concurrency=args.max_concurrency,
            gateway_url=str(args.gateway_url),
        )
        return {
            "checkpoint_hits": journals.checkpoint_hits,
            "command": "materialize",
            "gold_loaded": False,
            "physical_provider_calls": 0,
            "run_sha256": verified.run_sha256,
            "runtime_ledger_sha256": verified.runtime_ledger_sha256,
            "terminal_run_replayed": True,
        }
    result = materialize_query_operator_refinement_answers(
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
    verified = replay_query_operator_refinement_answers(
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
    if args.command == "preflight":
        result = _preflight(args)
    elif args.command == "provider-run":
        result = _provider(args)
    elif args.command == "materialize":
        result = _materialize(args)
    elif args.command == "replay":
        result = _replay(args)
    else:  # pragma: no cover
        raise AssertionError("unreachable command")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
