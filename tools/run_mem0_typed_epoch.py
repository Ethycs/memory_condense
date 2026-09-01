#!/usr/bin/env python3
"""Stage the sealed provider-free ``mem0-typed-v1`` campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from tools.matched_eval.contracts import canonical_json_bytes  # noqa: E402
from tools.mem0_eval.typed_epoch_campaign import (  # noqa: E402
    COMMON_INPUT_NAME,
    CONTRIBUTION_BUNDLE_NAME,
    COST_PREFLIGHT_NAME,
    PREFLIGHT_NAME,
    REPLAY_NAME,
    RETRIEVAL_BUNDLE_NAME,
    compose_campaign,
    finalize_costs,
    preflight_campaign,
    replay_campaign,
)


DEFAULT_OUTPUT_ROOT = Path("eval_results/mem0-typed-v1")


def _sha(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _add_count(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-question-count", type=int, default=100)


def _add_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and build in memory without publishing artifacts",
    )


def _add_locked_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--retrieval-bundle", type=Path, required=True)
    parser.add_argument("--expected-retrieval-bundle-sha256", required=True)
    parser.add_argument("--parent-population", type=Path, required=True)
    parser.add_argument("--expected-parent-population-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Provider-free Mem0 v3 retrieval adaptation and common-arm sealing. "
            "This command contains no Mem0, Terra, or Sol client."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--source-bridge", type=Path, required=True)
    preflight.add_argument("--parent-population", type=Path, required=True)
    preflight.add_argument("--expected-parent-population-sha256", required=True)
    preflight.add_argument("--expected-parent-run-sha256", required=True)
    preflight.add_argument("--expected-parent-replay-sha256", required=True)
    preflight.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    _add_count(preflight)
    _add_dry_run(preflight)

    compose = sub.add_parser("compose")
    _add_locked_inputs(compose)
    compose.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    _add_count(compose)
    _add_dry_run(compose)

    replay = sub.add_parser("replay")
    _add_locked_inputs(replay)
    replay.add_argument("--common-input", type=Path, required=True)
    replay.add_argument("--expected-common-input-sha256", required=True)
    replay.add_argument("--contribution-bundle", type=Path, required=True)
    replay.add_argument(
        "--expected-contribution-bundle-sha256",
        required=True,
    )
    replay.add_argument("--cost-preflight", type=Path, required=True)
    replay.add_argument("--expected-cost-preflight-sha256", required=True)
    replay.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    _add_count(replay)
    _add_dry_run(replay)

    finalize = sub.add_parser("finalize-costs")
    finalize.add_argument("--common-input", type=Path, required=True)
    finalize.add_argument("--expected-common-input-sha256", required=True)
    finalize.add_argument("--cost-preflight", type=Path, required=True)
    finalize.add_argument("--expected-cost-preflight-sha256", required=True)
    finalize.add_argument("--common-usage", type=Path, required=True)
    finalize.add_argument("--expected-common-usage-sha256", required=True)
    finalize.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    _add_count(finalize)
    _add_dry_run(finalize)
    return parser


def _summary(command: str, payloads: Sequence[dict[str, Any]], dry_run: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "command": command,
        "dry_run": dry_run,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "retained_transformer_token_state_bytes": 0,
    }
    for index, payload in enumerate(payloads):
        result[f"payload_{index + 1}_format"] = payload["format"]
        result[f"payload_{index + 1}_sha256"] = _sha(payload)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        preflight, bundle = preflight_campaign(
            source_bridge_path=args.source_bridge,
            parent_population_path=args.parent_population,
            expected_parent_population_sha256=(
                args.expected_parent_population_sha256
            ),
            expected_parent_run_sha256=args.expected_parent_run_sha256,
            expected_parent_replay_sha256=args.expected_parent_replay_sha256,
            output_root=args.output_root,
            expected_question_count=args.expected_question_count,
            dry_run=args.dry_run,
        )
        result = _summary(args.command, (preflight, bundle), args.dry_run)
        if not args.dry_run:
            result.update(
                {
                    "preflight": str(args.output_root / PREFLIGHT_NAME),
                    "retrieval_bundle": str(
                        args.output_root / RETRIEVAL_BUNDLE_NAME
                    ),
                }
            )
    elif args.command == "compose":
        common, contribution, cost = compose_campaign(
            preflight_path=args.preflight,
            expected_preflight_sha256=args.expected_preflight_sha256,
            retrieval_bundle_path=args.retrieval_bundle,
            expected_retrieval_bundle_sha256=(
                args.expected_retrieval_bundle_sha256
            ),
            parent_population_path=args.parent_population,
            expected_parent_population_sha256=(
                args.expected_parent_population_sha256
            ),
            output_root=args.output_root,
            expected_question_count=args.expected_question_count,
            dry_run=args.dry_run,
        )
        result = _summary(
            args.command,
            (common, contribution, cost),
            args.dry_run,
        )
        if not args.dry_run:
            result.update(
                {
                    "common_input": str(args.output_root / COMMON_INPUT_NAME),
                    "contribution_bundle": str(
                        args.output_root / CONTRIBUTION_BUNDLE_NAME
                    ),
                    "cost_preflight": str(
                        args.output_root / COST_PREFLIGHT_NAME
                    ),
                }
            )
    elif args.command == "replay":
        replay = replay_campaign(
            preflight_path=args.preflight,
            expected_preflight_sha256=args.expected_preflight_sha256,
            retrieval_bundle_path=args.retrieval_bundle,
            expected_retrieval_bundle_sha256=(
                args.expected_retrieval_bundle_sha256
            ),
            parent_population_path=args.parent_population,
            expected_parent_population_sha256=(
                args.expected_parent_population_sha256
            ),
            contribution_bundle_path=args.contribution_bundle,
            expected_contribution_bundle_sha256=(
                args.expected_contribution_bundle_sha256
            ),
            common_input_path=args.common_input,
            expected_common_input_sha256=args.expected_common_input_sha256,
            cost_preflight_path=args.cost_preflight,
            expected_cost_preflight_sha256=(
                args.expected_cost_preflight_sha256
            ),
            output_root=args.output_root,
            expected_question_count=args.expected_question_count,
            dry_run=args.dry_run,
        )
        result = _summary(args.command, (replay,), args.dry_run)
        if not args.dry_run:
            result["replay"] = str(args.output_root / REPLAY_NAME)
    else:
        final = finalize_costs(
            common_input_path=args.common_input,
            expected_common_input_sha256=args.expected_common_input_sha256,
            cost_preflight_path=args.cost_preflight,
            expected_cost_preflight_sha256=(
                args.expected_cost_preflight_sha256
            ),
            common_usage_path=args.common_usage,
            expected_common_usage_sha256=args.expected_common_usage_sha256,
            output_root=args.output_root,
            expected_question_count=args.expected_question_count,
            dry_run=args.dry_run,
        )
        result = _summary(args.command, (final,), args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
