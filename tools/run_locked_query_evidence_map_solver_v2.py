#!/usr/bin/env python3
"""Run the locked two-pass query evidence-map and conservative solver V2 arm."""

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
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    require_sha256,
)
from tools.matched_eval.payload_arm_identity import (  # noqa: E402
    QUERY_PAYLOAD_PROFILE,
    load_verified_payload_semantic_arm_binding,
)
from tools.matched_eval.population import EXPECTED_QUESTION_COUNT  # noqa: E402
from tools.matched_eval.query_evidence_map_solver_v2_live import (  # noqa: E402
    ANSWER_RUN_NAME,
    MAP_CHECKPOINT_DIR_NAME,
    MAP_OUTPUT_TOKEN_RESERVE,
    MAP_RUN_NAME,
    SOLVER_CHECKPOINT_DIR_NAME,
    EvidenceMapPlan,
    EvidenceSolverPlan,
    build_evidence_map_plan,
    build_evidence_solver_plan,
    load_map_provider_journals,
    load_map_provider_population,
    load_solver_provider_journals,
    load_solver_provider_population,
    materialize_evidence_map,
    materialize_evidence_solver,
    preflight_evidence_map,
    preflight_evidence_solver,
    replay_evidence_map,
    replay_evidence_solver,
    run_sealed_two_pass_provider,
)
from tools.matched_eval.query_payload_live import replay_query_payload_answers  # noqa: E402


DEFAULT_DIRECT_ANSWER_ROOT = payload_cli.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    payload_cli.DEFAULT_CAMPAIGN_ROOT / "s0-plus-query-evidence-map-solver-v2"
)
EXPECTED_DIRECT_ANSWER_PREFLIGHT_SHA256 = (
    "c5c705470259743ce1fb7e07bd72374ada32352f5240e44d06a17cf450f7ac9d"
)
EXPECTED_DIRECT_ANSWER_RUN_SHA256 = (
    "ab271ccb1bb830346fea64c9b11f3c7d504f048cc1ba392da39b177869106c6d"
)
EXPECTED_DIRECT_SEMANTIC_BINDING_SHA256 = (
    "3b808f7448e12518d5412aa013af54f3a7b654f05c9c14c6a6a779b6edd9757a"
)
EXPECTED_MAP_CALLS = 91


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
    parser.add_argument(
        "--expected-direct-semantic-binding-sha256",
        default=EXPECTED_DIRECT_SEMANTIC_BINDING_SHA256,
    )
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)


def _add_provider_inputs(parser: argparse.ArgumentParser, *, stage: str) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(f"--expected-{stage}-preflight-sha256", required=True)
    parser.add_argument("--enable-provider", action="store_true")
    parser.add_argument("--authorized-provider-calls", type=int, default=0)
    parser.add_argument("--api-key-env", default=live.DEFAULT_API_KEY_ENV)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--gateway-url", default=live.DEFAULT_GATEWAY_URL)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)

    map_preflight = subs.add_parser("map-preflight")
    _add_plan_inputs(map_preflight)
    map_provider = subs.add_parser("map-provider-run")
    _add_provider_inputs(map_provider, stage="map")
    map_materialize = subs.add_parser("map-materialize")
    _add_plan_inputs(map_materialize)
    map_materialize.add_argument("--expected-map-preflight-sha256", required=True)
    map_replay = subs.add_parser("map-replay")
    _add_plan_inputs(map_replay)
    map_replay.add_argument("--expected-map-preflight-sha256", required=True)
    map_replay.add_argument("--expected-map-run-sha256", required=True)

    solver_preflight = subs.add_parser("solver-preflight")
    _add_plan_inputs(solver_preflight)
    solver_preflight.add_argument("--expected-map-preflight-sha256", required=True)
    solver_preflight.add_argument("--expected-map-run-sha256", required=True)
    solver_provider = subs.add_parser("solver-provider-run")
    _add_provider_inputs(solver_provider, stage="solver")
    materialize = subs.add_parser("materialize")
    _add_plan_inputs(materialize)
    materialize.add_argument("--expected-map-preflight-sha256", required=True)
    materialize.add_argument("--expected-map-run-sha256", required=True)
    materialize.add_argument("--expected-solver-preflight-sha256", required=True)
    replay = subs.add_parser("replay")
    _add_plan_inputs(replay)
    replay.add_argument("--expected-map-preflight-sha256", required=True)
    replay.add_argument("--expected-map-run-sha256", required=True)
    replay.add_argument("--expected-solver-preflight-sha256", required=True)
    replay.add_argument("--expected-run-sha256", required=True)
    return parser


def _load_map_plan(args: argparse.Namespace) -> EvidenceMapPlan:
    output = Path(args.output_root).resolve()
    direct_root = Path(args.direct_answer_root)
    forbidden = {
        direct_root.resolve(),
        Path(args.parent_root).resolve(),
        Path(args.query_run).parent.resolve(),
    }
    if output in forbidden:
        raise MatchedEvalContractError(
            "V2 output must be isolated from every immutable source plane"
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
    load_verified_payload_semantic_arm_binding(
        direct_root,
        expected_profile=QUERY_PAYLOAD_PROFILE,
        expected_binding_sha256=args.expected_direct_semantic_binding_sha256,
        expected_question_count=EXPECTED_QUESTION_COUNT,
    )
    plan = build_evidence_map_plan(direct_plan, direct_plane)
    if len(plan.rows) != EXPECTED_QUESTION_COUNT:
        raise MatchedEvalContractError("locked V2 arm requires exactly 100 rows")
    if plan.required_calls != EXPECTED_MAP_CALLS:
        raise MatchedEvalContractError(
            f"locked V2 map requires exactly {EXPECTED_MAP_CALLS} calls"
        )
    for row in plan.rows:
        if row.dropped_query_delta_ids:
            raise MatchedEvalContractError("V2 map cannot drop fixed direct evidence")
        if row.submitted and (
            row.retained_query_delta_ids
            != row.direct_plan_row.retained_query_delta_ids
        ):
            raise MatchedEvalContractError("V2 map changed the fixed direct evidence")
    return plan


def _load_solver_plan(args: argparse.Namespace) -> EvidenceSolverPlan:
    map_plan = _load_map_plan(args)
    map_plane = replay_evidence_map(
        map_plan,
        output_root=args.output_root,
        expected_preflight_sha256=args.expected_map_preflight_sha256,
        expected_run_sha256=args.expected_map_run_sha256,
        max_concurrency=args.max_concurrency,
        gateway_url=str(args.gateway_url),
    )
    plan = build_evidence_solver_plan(map_plan, map_plane)
    if len(plan.rows) != EXPECTED_QUESTION_COUNT or plan.required_calls != EXPECTED_MAP_CALLS:
        raise MatchedEvalContractError("locked V2 solver population changed")
    return plan


def _map_preflight(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_map_plan(args)
    artifact = preflight_evidence_map(plan, output_root=args.output_root)
    max_prompt = max(
        (int(row.prompt_token_proxy) for row in plan.submitted_rows), default=0
    )
    return {
        "artifact": artifact.path.as_posix(),
        "dropped_evidence_count": sum(
            len(row.dropped_query_delta_ids) for row in plan.rows
        ),
        "eligible_route_counts": dict(
            sorted(Counter(row.route.style.value for row in plan.submitted_rows).items())
        ),
        "gold_loaded": False,
        "map_preflight_sha256": artifact.sha256,
        "max_combined_prompt_and_reserve_tokens": (
            max_prompt + MAP_OUTPUT_TOKEN_RESERVE
        ),
        "max_prompt_token_proxy": max_prompt,
        "provider_calls": 0,
        "question_count": len(plan.rows),
        "required_authorized_provider_calls": plan.required_calls,
    }


def _provider(args: argparse.Namespace, *, stage: str) -> dict[str, Any]:
    expected = getattr(args, f"expected_{stage}_preflight_sha256")
    population = (
        load_map_provider_population(
            output_root=args.output_root,
            expected_preflight_sha256=expected,
        )
        if stage == "map"
        else load_solver_provider_population(
            output_root=args.output_root,
            expected_preflight_sha256=expected,
        )
    )
    if (
        args.enable_provider is not True
        or args.authorized_provider_calls != population.required_calls
    ):
        raise MatchedEvalContractError(
            f"{stage}-provider-run requires exact authorization for "
            f"{population.required_calls} calls"
        )
    # Exact authorization and the sealed gold-blind population are verified
    # before environment loading or client creation.
    load_dotenv()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise MatchedEvalContractError(f"provider API key is empty: {args.api_key_env}")
    client = live._make_provider_client(api_key, str(args.gateway_url))
    try:
        result = run_sealed_two_pass_provider(
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
    checkpoint_name = (
        MAP_CHECKPOINT_DIR_NAME if stage == "map" else SOLVER_CHECKPOINT_DIR_NAME
    )
    checkpoint = Path(args.output_root) / checkpoint_name
    return {
        "checkpoint_hits": result.checkpoint_hits,
        "command": f"{stage}-provider-run",
        "gold_loaded": False,
        "physical_provider_calls": result.physical_provider_calls,
        "preflight_sha256": result.preflight_artifact.sha256,
        "request_journal_count": len(tuple(checkpoint.glob("*.request.json"))),
        "response_journal_count": len(tuple(checkpoint.glob("*.response.json"))),
    }


def _map_materialize(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_map_plan(args)
    expected = require_sha256(
        args.expected_map_preflight_sha256, "expected V2 map preflight"
    )
    journals = load_map_provider_journals(
        plan,
        output_root=args.output_root,
        expected_preflight_sha256=expected,
        max_concurrency=args.max_concurrency,
        gateway_url=str(args.gateway_url),
    )
    output = Path(args.output_root)
    existing = output / MAP_RUN_NAME
    if existing.exists():
        plane = replay_evidence_map(
            plan,
            output_root=output,
            expected_preflight_sha256=expected,
            expected_run_sha256=read_sealed_json(existing).sha256,
            max_concurrency=args.max_concurrency,
            gateway_url=str(args.gateway_url),
        )
        return {
            "checkpoint_hits": journals.checkpoint_hits,
            "command": "map-materialize",
            "gold_loaded": False,
            "map_run_sha256": plane.run_sha256,
            "physical_provider_calls": 0,
            "runtime_ledger_sha256": plane.runtime_ledger_sha256,
            "terminal_run_replayed": True,
        }
    result = materialize_evidence_map(
        plan,
        output_root=output,
        expected_preflight_sha256=expected,
        completion_batch=journals.batch,
        gateway_url=str(args.gateway_url),
    )
    return {
        "checkpoint_hits": journals.checkpoint_hits,
        "command": "map-materialize",
        "gold_loaded": False,
        "map_run_sha256": result.map_artifact.sha256,
        "physical_provider_calls": 0,
        "runtime_ledger_sha256": result.runtime_ledger_artifact.sha256,
        "terminal_run_replayed": False,
    }


def _map_replay(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_map_plan(args)
    plane = replay_evidence_map(
        plan,
        output_root=args.output_root,
        expected_preflight_sha256=args.expected_map_preflight_sha256,
        expected_run_sha256=args.expected_map_run_sha256,
        max_concurrency=args.max_concurrency,
        gateway_url=str(args.gateway_url),
    )
    return {
        "accepted_item_count": sum(len(row.accepted_items) for row in plane.rows),
        "command": "map-replay",
        "gold_loaded": False,
        "map_replay_sha256": plane.replay_sha256,
        "map_run_sha256": plane.run_sha256,
        "physical_provider_calls": 0,
        "runtime_ledger_sha256": plane.runtime_ledger_sha256,
    }


def _solver_preflight(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_solver_plan(args)
    artifact = preflight_evidence_solver(plan, output_root=args.output_root)
    max_prompt = max(
        (int(row.prompt_token_proxy) for row in plan.submitted_rows), default=0
    )
    return {
        "artifact": artifact.path.as_posix(),
        "gold_loaded": False,
        "max_prompt_token_proxy": max_prompt,
        "provider_calls": 0,
        "question_count": len(plan.rows),
        "required_authorized_provider_calls": plan.required_calls,
        "solver_preflight_sha256": artifact.sha256,
    }


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_solver_plan(args)
    expected = require_sha256(
        args.expected_solver_preflight_sha256, "expected V2 solver preflight"
    )
    journals = load_solver_provider_journals(
        plan,
        output_root=args.output_root,
        expected_preflight_sha256=expected,
        max_concurrency=args.max_concurrency,
        gateway_url=str(args.gateway_url),
    )
    output = Path(args.output_root)
    existing = output / ANSWER_RUN_NAME
    if existing.exists():
        plane = replay_evidence_solver(
            plan,
            output_root=output,
            expected_preflight_sha256=expected,
            expected_run_sha256=read_sealed_json(existing).sha256,
            max_concurrency=args.max_concurrency,
            gateway_url=str(args.gateway_url),
        )
        return {
            "changed_prediction_count": len(plane.changed_rows),
            "checkpoint_hits": journals.checkpoint_hits,
            "command": "materialize",
            "gold_loaded": False,
            "physical_provider_calls": 0,
            "run_sha256": plane.run_sha256,
            "runtime_ledger_sha256": plane.runtime_ledger_sha256,
            "terminal_run_replayed": True,
        }
    result = materialize_evidence_solver(
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
    plan = _load_solver_plan(args)
    plane = replay_evidence_solver(
        plan,
        output_root=args.output_root,
        expected_preflight_sha256=args.expected_solver_preflight_sha256,
        expected_run_sha256=args.expected_run_sha256,
        max_concurrency=args.max_concurrency,
        gateway_url=str(args.gateway_url),
    )
    return {
        "changed_prediction_count": len(plane.changed_rows),
        "command": "replay",
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "replay_sha256": plane.replay_sha256,
        "run_sha256": plane.run_sha256,
        "runtime_ledger_sha256": plane.runtime_ledger_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "map-preflight":
        result = _map_preflight(args)
    elif args.command == "map-provider-run":
        result = _provider(args, stage="map")
    elif args.command == "map-materialize":
        result = _map_materialize(args)
    elif args.command == "map-replay":
        result = _map_replay(args)
    elif args.command == "solver-preflight":
        result = _solver_preflight(args)
    elif args.command == "solver-provider-run":
        result = _provider(args, stage="solver")
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
