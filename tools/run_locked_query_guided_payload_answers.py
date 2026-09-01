#!/usr/bin/env python3
"""Run the locked query-guided direct-payload answer arm in split phases."""

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
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    require_sha256,
)
from tools.matched_eval.population import (  # noqa: E402
    EXPECTED_QUESTION_COUNT,
    EXPECTED_RETRIEVAL_SHA256,
)
from tools.matched_eval.query_guided_payload_adapter import (  # noqa: E402
    DELTA_TIER,
    load_query_guided_payload_adapter,
)
from tools.matched_eval.query_payload_live import (  # noqa: E402
    ANSWER_RUN_NAME,
    CHECKPOINT_DIR_NAME,
    QueryPayloadAnswerPlan,
    build_query_payload_answer_plan,
    load_query_payload_answer_provider_journals,
    materialize_query_payload_answers,
    preflight_query_payload_answers,
    replay_query_payload_answers,
    run_query_payload_answer_provider,
)
from tools.run_locked_query_payload_answers import (  # noqa: E402
    DEFAULT_CAMPAIGN_ROOT,
    DEFAULT_PARENT_ROOT,
    DEFAULT_RETRIEVAL,
    EXPECTED_PARENT_ANSWER_RUN_SHA256,
    EXPECTED_QUERY_POPULATION_ID,
    EXPECTED_QUERY_PREFLIGHT_SHA256,
    EXPECTED_QUERY_PROMPT_POPULATION_SHA256,
    EXPECTED_QUERY_RUN_SHA256,
    EXPECTED_SOURCE_POPULATION_ID,
)


DEFAULT_QUERY_PARENT_ROOT = DEFAULT_CAMPAIGN_ROOT / "s0-plus-query-expansion-v1"
DEFAULT_GUIDED_ROOT = DEFAULT_CAMPAIGN_ROOT / "s0-plus-query-guided-scan-v1"
DEFAULT_OUTPUT = DEFAULT_CAMPAIGN_ROOT / "s0-plus-query-guided-payload-v1"

EXPECTED_QUERY_RUNTIME_LEDGER_SHA256 = (
    "16d5ceedee9a86d7c719d3d66538a4d8fa23cf8fbee5763097df69f28afc7c94"
)
EXPECTED_GUIDED_RUN_SHA256 = (
    "a544ae9e6e554fcfc9cfc6167018f06b573fcf6546c9c3f3a6e3feda6ed821ff"
)
EXPECTED_GUIDED_RUNTIME_LEDGER_SHA256 = (
    "b0edd491ddca674c24728f31cda337226090624db04c63a507eb6188eb802af7"
)


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--query-parent-root",
        type=Path,
        default=DEFAULT_QUERY_PARENT_ROOT,
    )
    parser.add_argument("--guided-root", type=Path, default=DEFAULT_GUIDED_ROOT)
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
        "--expected-query-preflight-sha256",
        default=EXPECTED_QUERY_PREFLIGHT_SHA256,
    )
    parser.add_argument(
        "--expected-query-run-sha256",
        default=EXPECTED_QUERY_RUN_SHA256,
    )
    parser.add_argument(
        "--expected-query-runtime-ledger-sha256",
        default=EXPECTED_QUERY_RUNTIME_LEDGER_SHA256,
    )
    parser.add_argument(
        "--expected-query-population-id",
        default=EXPECTED_QUERY_POPULATION_ID,
    )
    parser.add_argument(
        "--expected-query-prompt-population-sha256",
        default=EXPECTED_QUERY_PROMPT_POPULATION_SHA256,
    )
    parser.add_argument(
        "--expected-guided-run-sha256",
        default=EXPECTED_GUIDED_RUN_SHA256,
    )
    parser.add_argument(
        "--expected-guided-runtime-ledger-sha256",
        default=EXPECTED_GUIDED_RUNTIME_LEDGER_SHA256,
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


def _load_plan(args: argparse.Namespace) -> QueryPayloadAnswerPlan:
    adapter = load_query_guided_payload_adapter(
        args.retrieval,
        query_parent_root=args.query_parent_root,
        guided_root=args.guided_root,
        expected_retrieval_sha256=args.expected_retrieval_sha256,
        expected_source_population_id=args.expected_source_population_id,
        expected_query_preflight_sha256=args.expected_query_preflight_sha256,
        expected_query_run_sha256=args.expected_query_run_sha256,
        expected_query_runtime_ledger_sha256=(
            args.expected_query_runtime_ledger_sha256
        ),
        expected_query_population_id=args.expected_query_population_id,
        expected_query_prompt_population_sha256=(
            args.expected_query_prompt_population_sha256
        ),
        expected_guided_run_sha256=args.expected_guided_run_sha256,
        expected_guided_runtime_ledger_sha256=(
            args.expected_guided_runtime_ledger_sha256
        ),
        expected_question_count=EXPECTED_QUESTION_COUNT,
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
        Path(args.query_parent_root).resolve(),
        Path(args.guided_root).resolve(),
    }
    if output in forbidden:
        raise MatchedEvalContractError(
            "guided-payload output must be isolated from all source roots"
        )
    plan = build_query_payload_answer_plan(
        adapter,
        parent,
        delta_tier=DELTA_TIER,
    )
    if len(plan.rows) != EXPECTED_QUESTION_COUNT:
        raise MatchedEvalContractError("locked guided-payload arm requires 100 rows")
    return plan


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_plan(args)
    artifact = preflight_query_payload_answers(plan, output_root=args.output_root)
    return {
        "adapter_population_id": plan.adapter_population.population_id,
        "artifact": artifact.path.as_posix(),
        "delta_tier": plan.delta_tier,
        "dropped_evidence_count": plan.dropped_evidence_count,
        "dropped_row_count": plan.dropped_row_count,
        "fallback_count": len(plan.rows) - plan.required_calls,
        "gold_loaded": False,
        "guided_run_sha256": args.expected_guided_run_sha256,
        "guided_runtime_ledger_sha256": (
            args.expected_guided_runtime_ledger_sha256
        ),
        "max_prompt_token_proxy": max(
            (row.prompt_token_proxy for row in plan.submitted_rows),
            default=0,
        ),
        "preflight_sha256": artifact.sha256,
        "provider_calls": 0,
        "question_count": len(plan.rows),
        "required_authorized_provider_calls": plan.required_calls,
    }


def _provider(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_plan(args)
    if (
        args.enable_provider is not True
        or args.authorized_provider_calls != plan.required_calls
    ):
        raise MatchedEvalContractError(
            f"provider-run requires exact authorization for {plan.required_calls} calls"
        )
    expected = require_sha256(
        args.expected_answer_preflight_sha256,
        "expected guided-payload answer preflight",
    )
    load_dotenv()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise MatchedEvalContractError(f"provider API key is empty: {args.api_key_env}")
    client = live._make_provider_client(api_key, str(args.gateway_url))
    try:
        result = run_query_payload_answer_provider(
            plan,
            output_root=args.output_root,
            expected_preflight_sha256=expected,
            enable_provider=True,
            authorized_provider_calls=plan.required_calls,
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
        "expected guided-payload answer preflight",
    )
    journals = load_query_payload_answer_provider_journals(
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
        verified = replay_query_payload_answers(
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
    result = materialize_query_payload_answers(
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
    verified = replay_query_payload_answers(
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
