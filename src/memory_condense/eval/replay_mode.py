"""Stateful workflow for conversation replay and parameter sweeps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

def run_replay_mode(args: argparse.Namespace, *, runtime) -> None:
    print(f"Loading conversations from {args.conversation_dir}...")
    conversations = runtime.load_directory(args.conversation_dir)
    if not conversations:
        print("No conversations found.")
        sys.exit(1)
    print(f"Found {len(conversations)} conversations")

    config = runtime.config_from_args(args)
    if config.retrieval.coverage_selection:
        raise ValueError(
            "coverage selection is currently measured through benchmark/recall "
            "packing, not self-replay"
        )

    if args.sweep:
        report = runtime.run_sweep(config, conversations)
        runtime.print_sweep_table(report)
        path = runtime.save_sweep_report(report, args.results_dir)
        print(f"\nSweep report saved to {path}")
        return

    print("\nRunning single eval...")
    result = runtime.run_eval(config, conversations)
    runtime.print_run_summary(result)
    path = runtime.save_run_result(result, args.results_dir)
    print(f"\nResult saved to {path}")

    if args.csv:
        Path(args.csv).write_text(runtime.to_csv(result), encoding="utf-8")
        print(f"Per-turn CSV written to {args.csv}")
