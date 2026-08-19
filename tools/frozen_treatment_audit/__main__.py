"""Command-line entry point for the frozen treatment audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .audit import AuditError, audit_frozen_treatment
from .canonical import validate_output_location


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify frozen v3 lineage and structural report consistency without "
            "authenticating provider, judge, or factual-accuracy claims."
        )
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--compiled-cache-root", type=Path, required=True)
    parser.add_argument("--causal-cache-root", type=Path, required=True)
    parser.add_argument(
        "--expected-audit-tool-sha256",
        required=True,
        help=(
            "Externally recorded SHA-256 of the tools/frozen_treatment_audit "
            "Python source package."
        ),
    )
    parser.add_argument(
        "--shard-report-root",
        type=Path,
        help="Directory containing the exact campaign input shard reports; defaults to report parent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New receipt path. Omit to write canonical JSON to stdout.",
    )
    return parser


def _publish_atomic_no_clobber(target: Path, payload: bytes) -> None:
    """Durably stage bytes and atomically link them into a new final name."""

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise AuditError(f"refusing to replace existing receipt {target}") from exc
        except OSError as exc:
            raise AuditError(f"cannot atomically publish receipt {target}: {exc}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    shard_root = args.shard_report_root or args.report.resolve().parent
    output: Path | None = None
    try:
        if args.output is not None:
            output = validate_output_location(
                args.output,
                protected_roots=(
                    args.repository_root,
                    args.compiled_cache_root,
                    args.causal_cache_root,
                    shard_root,
                ),
                protected_files=(
                    args.report,
                    args.dataset,
                    args.split_manifest,
                    args.policy,
                ),
            )
        receipt = audit_frozen_treatment(
            report_path=args.report,
            dataset_path=args.dataset,
            split_manifest_path=args.split_manifest,
            policy_path=args.policy,
            repository_root=args.repository_root,
            source_commit=args.source_commit,
            compiled_cache_root=args.compiled_cache_root,
            causal_cache_root=args.causal_cache_root,
            shard_report_root=shard_root,
            output_path=output,
            expected_audit_tool_sha256=args.expected_audit_tool_sha256,
        )
    except AuditError as exc:
        print(f"audit failed closed: {exc}", file=sys.stderr)
        return 2
    encoded = json.dumps(
        receipt,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if output is None:
        sys.stdout.write(encoded)
        return 0
    try:
        output = validate_output_location(
            output,
            protected_roots=(
                args.repository_root,
                args.compiled_cache_root,
                args.causal_cache_root,
                shard_root,
            ),
            protected_files=(
                args.report,
                args.dataset,
                args.split_manifest,
                args.policy,
            ),
        )
        _publish_atomic_no_clobber(output, encoded.encode("utf-8"))
    except (AuditError, OSError) as exc:
        print(f"cannot create receipt {output}: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
