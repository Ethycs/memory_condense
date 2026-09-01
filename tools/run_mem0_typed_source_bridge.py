#!/usr/bin/env python3
"""Build the provider-free ten-shard Mem0 typed source bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from tools.matched_eval.contracts import canonical_json_bytes  # noqa: E402
from tools.mem0_eval.typed_source_bridge import (  # noqa: E402
    LockedSourceInputs,
    MANIFEST_NAME,
    ResumableTerminalInput,
    build_source_bridge,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate ten official resumable Mem0 terminals and rebuild "
            "their diagnostic-attribution-preserving typed retrieval exports. "
            "This command contains no Mem0 or provider client."
        )
    )
    parser.add_argument("--benchmark-file", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--mem0-policy-manifest", type=Path, required=True)
    parser.add_argument("--mem0-environment-lock", type=Path, required=True)
    parser.add_argument("--mem0-tool-root", type=Path, required=True)
    parser.add_argument(
        "--terminal-artifact", action="append", type=Path, required=True
    )
    parser.add_argument("--terminal-trace", action="append", type=Path, required=True)
    parser.add_argument("--resume-journal", action="append", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("eval_results/mem0-typed-v1")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    counts = {
        len(args.terminal_artifact),
        len(args.terminal_trace),
        len(args.resume_journal),
    }
    if counts != {10}:
        raise ValueError(
            "exactly ten terminal artifacts, traces, and journals are required"
        )
    terminals = tuple(
        ResumableTerminalInput(artifact, trace, journal)
        for artifact, trace, journal in zip(
            args.terminal_artifact,
            args.terminal_trace,
            args.resume_journal,
            strict=True,
        )
    )
    source = LockedSourceInputs(
        benchmark_file=args.benchmark_file,
        split_manifest=args.split_manifest,
        policy_manifest=args.policy_manifest,
        repository_root=args.repository_root,
        mem0_policy_manifest=args.mem0_policy_manifest,
        mem0_environment_lock=args.mem0_environment_lock,
        mem0_tool_root=args.mem0_tool_root,
    )
    manifest, exports = build_source_bridge(
        source=source,
        terminals=terminals,
        output_root=args.output_root,
        dry_run=args.dry_run,
    )
    result = {
        "dry_run": args.dry_run,
        "export_count": len(exports),
        "format": manifest["format"],
        "gold_loaded": False,
        "manifest": None if args.dry_run else str(args.output_root / MANIFEST_NAME),
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
        "physical_provider_calls": 0,
        "question_count": manifest["question_count"],
        "retained_transformer_token_state_bytes": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
