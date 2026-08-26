"""Provider-free CLI for the nonofficial Mem0 direct-store scaffold."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .direct_store import (
    DIRECT_STORE_ARM_ID,
    DIRECT_STORE_ARM_LABEL,
    DIRECT_STORE_SCHEMA_VERSION,
    save_direct_store_artifact,
)
from .direct_store_report import (
    build_frozen_direct_store_population_preflight,
    merge_direct_store_retrieval_shards,
    save_direct_store_campaign_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or merge the nonofficial Mem0 infer=False direct-store arm"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "status", help="show the static provider-free execution status"
    )
    subparsers.add_parser(
        "run-shard",
        help="fail closed because no concrete Mem0 runtime is bound",
    )

    preflight = subparsers.add_parser(
        "preflight", help="reconstruct and verify the locked 100Q population"
    )
    preflight.add_argument("--benchmark-file", required=True)
    preflight.add_argument("--split-manifest", required=True)
    preflight.add_argument("--output")

    merge = subparsers.add_parser(
        "merge", help="validate and merge ten injected retrieval artifacts"
    )
    merge.add_argument("reports", nargs="+")
    merge.add_argument("--benchmark-file", required=True)
    merge.add_argument("--split-manifest", required=True)
    merge.add_argument("--output")
    return parser


def _render(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )


def _status() -> dict[str, Any]:
    return {
        "arm_id": DIRECT_STORE_ARM_ID,
        "label": DIRECT_STORE_ARM_LABEL,
        "schema_version": DIRECT_STORE_SCHEMA_VERSION,
        "official_mem0_comparison": False,
        "infer": False,
        "provider_calls_authorized": 0,
        "network_calls_authorized": 0,
        "actual_mem0_executed": False,
        "retrieval_runner": "programmatic_injected_test_double_only",
        "cli_runtime_binding": "blocked",
        "benchmark_result_eligible": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "status":
        print(_render(_status()))
        return 0
    if args.command == "run-shard":
        parser.error(
            "no concrete Mem0 runtime is bound; dynamic backend imports are "
            "intentionally disabled. Use run_injected_direct_store_shard only "
            "with a local test double."
        )
    if args.command == "preflight":
        _shards, receipt = build_frozen_direct_store_population_preflight(
            benchmark_file=args.benchmark_file,
            split_manifest=args.split_manifest,
        )
        if args.output:
            save_direct_store_artifact(receipt, args.output)
        else:
            print(_render(receipt))
        return 0
    if args.command == "merge":
        shards, _receipt = build_frozen_direct_store_population_preflight(
            benchmark_file=args.benchmark_file,
            split_manifest=args.split_manifest,
        )
        campaign = merge_direct_store_retrieval_shards(
            args.reports,
            expected_shards=shards,
        )
        if args.output:
            save_direct_store_campaign_report(campaign, args.output)
        else:
            print(_render(campaign))
        return 0
    raise AssertionError("unreachable direct-store CLI command")


if __name__ == "__main__":
    raise SystemExit(main())
