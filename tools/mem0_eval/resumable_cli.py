"""Command-line surface for sealed resumable Mem0 launch phases."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from .resumable import publish_sealed_json
from .resumable_launch import (
    MANIFEST_NAME,
    PREFLIGHT_NAME,
    PROSPECTIVE_PROVIDER_CALLS,
    REPLAY_NAME,
    LockedLaunchInputs,
    build_preflight_payload,
    load_locked_launch_context,
    materialize_launch,
    recheck_locked_launch_inputs,
    replay_launch,
    run_locked_live_segment,
)


def _add_locked_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark-file", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-policy-manifest", type=Path, required=True)
    parser.add_argument("--source-repository-root", type=Path, required=True)
    parser.add_argument("--mem0-policy-manifest", type=Path, required=True)
    parser.add_argument("--expected-mem0-policy-sha256", required=True)
    parser.add_argument("--mem0-environment-lock", type=Path, required=True)
    parser.add_argument(
        "--tool-root", type=Path, default=Path(__file__).resolve().parent
    )


def _add_preflight_reference(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)


def _add_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="rebuild and validate in memory without publishing an artifact",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight, materialize, replay, or advance one sealed validation100 "
            "Mem0 segment. Only the explicit segment action can authorize "
            "extraction provider calls; answer and judge calls remain separate."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    _add_locked_inputs(preflight)
    preflight.add_argument("--output", type=Path)
    _add_dry_run(preflight)

    materialize = sub.add_parser("materialize")
    _add_locked_inputs(materialize)
    _add_preflight_reference(materialize)
    materialize.add_argument("--run-root", type=Path, required=True)
    _add_dry_run(materialize)

    replay = sub.add_parser("replay")
    _add_locked_inputs(replay)
    _add_preflight_reference(replay)
    replay.add_argument("--run-root", type=Path, required=True)
    replay.add_argument("--expected-launch-manifest-sha256", required=True)
    replay.add_argument("--output", type=Path)
    _add_dry_run(replay)

    segment = sub.add_parser("segment")
    _add_locked_inputs(segment)
    _add_preflight_reference(segment)
    segment.add_argument("--run-root", type=Path, required=True)
    segment.add_argument("--expected-launch-manifest-sha256", required=True)
    segment.add_argument(
        "--sample-offset", type=int, choices=tuple(range(0, 100, 10)), required=True
    )
    segment.add_argument(
        "--authorize-provider-calls",
        type=int,
        required=True,
        help="exact next-segment extraction call count (256, or final tail)",
    )
    return parser


def _inputs(args: argparse.Namespace) -> LockedLaunchInputs:
    return LockedLaunchInputs(
        benchmark_file=args.benchmark_file,
        split_manifest=args.split_manifest,
        source_policy_manifest=args.source_policy_manifest,
        source_repository_root=args.source_repository_root,
        mem0_policy_manifest=args.mem0_policy_manifest,
        expected_mem0_policy_sha256=args.expected_mem0_policy_sha256,
        mem0_environment_lock=args.mem0_environment_lock,
        tool_root=args.tool_root,
    )


def _require_bootstrap_envelope(*, allow_network: bool = False) -> None:
    if not sys.flags.isolated:
        raise RuntimeError(
            "resumable launch CLI must run through bootstrap.py under Python -I"
        )
    expected_environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "MEM0_TELEMETRY": "false",
    }
    if any(os.environ.get(key) != value for key, value in expected_environment.items()):
        raise RuntimeError("provider-free bootstrap environment is incomplete")
    cache = os.environ.get("CUSTOM_TIKTOKEN_CACHE_DIR")
    if (
        not cache
        or os.environ.get("TIKTOKEN_CACHE_DIR") != cache
        or not os.environ.get("MEM0_VERIFIED_BOOTSTRAP_SOURCE_SHA256")
        or not os.environ.get("MEM0_VERIFIED_BOOTSTRAP_TOOL_SHA256")
        or os.environ.get("MEM0_VERIFIED_BOOTSTRAP_NETWORK_DENIED")
        != ("0" if allow_network else "1")
    ):
        raise RuntimeError("verified provider-free bootstrap receipt is absent")


def _require_verified_bootstrap(context, *, allow_network: bool = False) -> None:
    _require_bootstrap_envelope(allow_network=allow_network)
    if (
        os.environ.get("MEM0_VERIFIED_BOOTSTRAP_SOURCE_SHA256")
        != context.source_plan.implementation_sha256
        or os.environ.get("MEM0_VERIFIED_BOOTSTRAP_TOOL_SHA256")
        != context.mem0_tool_implementation_sha256
    ):
        raise RuntimeError("verified bootstrap tree identities do not match launch")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight" and not args.dry_run and args.output is None:
        raise ValueError("preflight requires --output unless --dry-run is set")
    live_segment = args.command == "segment"
    _require_bootstrap_envelope(allow_network=live_segment)
    inputs = _inputs(args)
    context = load_locked_launch_context(inputs)
    _require_verified_bootstrap(context, allow_network=live_segment)
    if args.command == "preflight":
        payload = build_preflight_payload(context)
        if args.dry_run:
            artifact = None
            digest = None
        else:
            receipt = publish_sealed_json(args.output, payload)
            artifact = str(receipt["path"])
            digest = receipt["sha256"]
        result = {
            "command": "preflight",
            "dry_run": args.dry_run,
            "format": payload["format"],
            "artifact": artifact,
            "artifact_sha256": digest,
            "prospective_provider_call_ceiling": PROSPECTIVE_PROVIDER_CALLS,
            "provider_call_authorization_granted": False,
            "physical_provider_calls": 0,
            "retained_transformer_token_state_bytes": 0,
        }
    elif args.command == "materialize":
        manifest, shards = materialize_launch(
            context=context,
            preflight_path=args.preflight,
            expected_preflight_sha256=args.expected_preflight_sha256,
            run_root=args.run_root,
            dry_run=args.dry_run,
        )
        result = {
            "command": "materialize",
            "dry_run": args.dry_run,
            "format": manifest["format"],
            "manifest": None if args.dry_run else str(args.run_root / MANIFEST_NAME),
            "preflight_copy": None if args.dry_run else str(args.run_root / PREFLIGHT_NAME),
            "shard_count": len(shards),
            "provider_call_authorization_granted": False,
            "physical_provider_calls": 0,
            "retained_transformer_token_state_bytes": 0,
        }
    elif args.command == "replay":
        launch_path = args.run_root / MANIFEST_NAME
        output = args.output or args.run_root / REPLAY_NAME
        payload = replay_launch(
            context=context,
            preflight_path=args.preflight,
            expected_preflight_sha256=args.expected_preflight_sha256,
            launch_manifest_path=launch_path,
            expected_launch_manifest_sha256=(
                args.expected_launch_manifest_sha256
            ),
            run_root=args.run_root,
            output_path=output,
            dry_run=args.dry_run,
        )
        result = {
            "command": "replay",
            "dry_run": args.dry_run,
            "format": payload["format"],
            "replay": None if args.dry_run else str(output),
            "shard_count": len(payload["shards"]),
            "provider_call_authorization_granted": False,
            "physical_provider_calls": 0,
            "retained_transformer_token_state_bytes": 0,
        }
    else:
        segment = run_locked_live_segment(
            inputs=inputs,
            preflight_path=args.preflight,
            expected_preflight_sha256=args.expected_preflight_sha256,
            launch_manifest_path=args.run_root / MANIFEST_NAME,
            expected_launch_manifest_sha256=(
                args.expected_launch_manifest_sha256
            ),
            run_root=args.run_root,
            sample_offset=args.sample_offset,
            authorized_provider_calls=args.authorize_provider_calls,
        )
        result = {
            "command": "segment",
            "action": segment.action,
            "sample_offset": args.sample_offset,
            "prefix_before": segment.prefix_before,
            "prefix_after": segment.prefix_after,
            "authorized_provider_calls": args.authorize_provider_calls,
            "physical_provider_calls": segment.segment_adds,
            "provider_call_authorization_granted": True,
            "checkpoint_authority_sha256": (
                segment.checkpoint_authority_sha256
            ),
            "journal_tail_sha256": segment.journal_tail_sha256,
            "receipt_sha256": segment.receipt_sha256,
            "sdk_retries": 0,
            "retained_transformer_token_state_bytes": 0,
        }
    recheck_locked_launch_inputs(inputs, context)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
