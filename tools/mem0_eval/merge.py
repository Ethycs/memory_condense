"""Command-line entry point for the strict ten-shard Mem0 merger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .report import merge_mem0_shard_reports, save_mem0_campaign_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and merge the locked ten-shard Mem0 comparison"
    )
    parser.add_argument("reports", nargs="+", help="exactly ten Stage-B reports")
    parser.add_argument("--benchmark-file", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--policy-manifest", required=True)
    parser.add_argument("--mem0-policy-manifest", required=True)
    parser.add_argument("--mem0-environment-lock", required=True)
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = merge_mem0_shard_reports(
        args.reports,
        benchmark_file=args.benchmark_file,
        split_manifest=args.split_manifest,
        policy_manifest=args.policy_manifest,
        mem0_policy_manifest=args.mem0_policy_manifest,
        mem0_environment_lock=args.mem0_environment_lock,
    )
    if args.output:
        save_mem0_campaign_report(report, Path(args.output))
    else:
        print(
            json.dumps(
                report,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
