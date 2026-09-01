#!/usr/bin/env python3
"""Run the locked direct query-payload answer arm in split phases."""

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
from tools.matched_eval.payload_arm_identity import (  # noqa: E402
    ensure_payload_semantic_arm_binding,
    profile_for_delta_tier,
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
DEFAULT_PARENT_ROOT = DEFAULT_CAMPAIGN_ROOT / "s0-control-v2"
DEFAULT_OUTPUT = DEFAULT_CAMPAIGN_ROOT / "s0-plus-query-payload-v1"

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


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--query-preflight", type=Path, default=DEFAULT_QUERY_PREFLIGHT)
    parser.add_argument("--query-run", type=Path, default=DEFAULT_QUERY_RUN)
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
    parser.add_argument(
        "--expected-parent-answer-run-sha256",
        default=EXPECTED_PARENT_ANSWER_RUN_SHA256,
    )
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    _add_inputs(preflight)
    provider = subparsers.add_parser("provider-run")
    _add_inputs(provider)
    provider.add_argument("--expected-answer-preflight-sha256", required=True)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, default=0)
    provider.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)
    materialize = subparsers.add_parser("materialize")
    _add_inputs(materialize)
    materialize.add_argument("--expected-answer-preflight-sha256", required=True)
    replay = subparsers.add_parser("replay")
    _add_inputs(replay)
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
        expected_query_prompt_population_sha256=(
            args.expected_query_prompt_population_sha256
        ),
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
    if output in {parent_root.resolve(), Path(args.query_run).parent.resolve()}:
        raise MatchedEvalContractError(
            "query-payload output must be isolated from parent and query roots"
        )
    plan = build_query_payload_answer_plan(adapter, parent)
    if len(plan.rows) != EXPECTED_QUESTION_COUNT:
        raise MatchedEvalContractError("locked query-payload arm requires 100 rows")
    return plan


def _preflight_loaded_plan(
    plan: QueryPayloadAnswerPlan,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Publish a zero-call preflight for any verified shared payload plan."""

    artifact = preflight_query_payload_answers(plan, output_root=args.output_root)
    return {
        "artifact": artifact.path.as_posix(),
        "dropped_evidence_count": plan.dropped_evidence_count,
        "dropped_row_count": plan.dropped_row_count,
        "fallback_count": len(plan.rows) - plan.required_calls,
        "gold_loaded": False,
        "max_prompt_token_proxy": max(
            (row.prompt_token_proxy for row in plan.submitted_rows), default=0
        ),
        "preflight_sha256": artifact.sha256,
        "provider_calls": 0,
        "question_count": len(plan.rows),
        "required_authorized_provider_calls": plan.required_calls,
    }


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    return _preflight_loaded_plan(_load_plan(args), args)


def _provider_loaded_plan(
    plan: QueryPayloadAnswerPlan,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Fill journals for any verified shared payload plan."""

    # Authorization precedes environment loading, client construction, and any
    # new provider journal.
    if args.enable_provider is not True or args.authorized_provider_calls != plan.required_calls:
        raise MatchedEvalContractError(
            f"provider-run requires exact authorization for {plan.required_calls} calls"
        )
    expected = require_sha256(
        args.expected_answer_preflight_sha256,
        "expected query-payload answer preflight",
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


def _provider(args: argparse.Namespace) -> dict[str, Any]:
    return _provider_loaded_plan(_load_plan(args), args)


def _materialize_loaded_plan(
    plan: QueryPayloadAnswerPlan,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Materialize or replay any verified shared payload plan."""

    expected = require_sha256(
        args.expected_answer_preflight_sha256,
        "expected query-payload answer preflight",
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


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    return _materialize_loaded_plan(_load_plan(args), args)


def _replay_loaded_plan(
    plan: QueryPayloadAnswerPlan,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Replay any verified shared payload plan without a client."""

    verified = replay_query_payload_answers(
        plan,
        output_root=args.output_root,
        expected_preflight_sha256=args.expected_answer_preflight_sha256,
        expected_run_sha256=args.expected_run_sha256,
        max_concurrency=args.max_concurrency,
        gateway_url=str(args.gateway_url),
    )
    semantic_binding, semantic_binding_created = (
        ensure_payload_semantic_arm_binding(
            args.output_root,
            profile=profile_for_delta_tier(plan.delta_tier),
            expected_question_count=len(plan.rows),
        )
    )
    return {
        "changed_prediction_count": len(verified.changed_rows),
        "command": "replay",
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "replay_sha256": verified.replay_sha256,
        "run_sha256": verified.run_sha256,
        "runtime_ledger_sha256": verified.runtime_ledger_sha256,
        "semantic_arm_binding_created": semantic_binding_created,
        "semantic_arm_binding_sha256": semantic_binding.sha256,
        "semantic_arm_label": semantic_binding.payload["semantic_profile"][
            "semantic_arm_label"
        ],
    }


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    return _replay_loaded_plan(_load_plan(args), args)


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
