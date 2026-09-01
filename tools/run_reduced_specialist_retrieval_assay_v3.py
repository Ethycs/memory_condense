#!/usr/bin/env python3
"""Seal the repaired reduced-specialist v3 construction and target audit.

The mechanics remain in the reusable v2 implementation.  This entry point
changes the protocol identity and filenames so algorithm repairs can never
silently overwrite or masquerade as the sealed v2 experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from tools import run_reduced_specialist_retrieval_assay as base  # noqa: E402
from tools.matched_eval.artifacts import (  # noqa: E402
    publish_sealed_json,
    read_sealed_json,
)
from tools.matched_eval.contracts import require_sha256  # noqa: E402
from tools.matched_eval.specialist_scoped_completion import (  # noqa: E402
    PROMPT_FORMAT as SPECIALIST_PROMPT_FORMAT,
    render_specialist_scoped_prompt,
)


FORMAT = "memory-condense-reduced-specialist-retrieval-assay-v3"
CONSTRUCTION_FORMAT = f"{FORMAT}-construction"
AUDIT_FORMAT = f"{FORMAT}-posthoc-target-audit"
CONSTRUCTION_NAME = "reduced-specialist-construction-v3.json"
AUDIT_NAME = "reduced-specialist-target-audit-v3.json"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / (
    "eval_results/matched_eval_100/reduced-specialist-missing10-v3"
)


def run_construct(args: argparse.Namespace) -> dict[str, Any]:
    payload = base.build_construction(
        args,
        construction_format=CONSTRUCTION_FORMAT,
        terminal_message_renderer_format=SPECIALIST_PROMPT_FORMAT,
        terminal_prompt_envelope_renderer=render_specialist_scoped_prompt,
    )
    artifact, created = publish_sealed_json(
        Path(args.output_root) / CONSTRUCTION_NAME,
        payload,
    )
    return {
        "construction_sha256": artifact.sha256,
        "created": created,
        "new_provider_calls": 0,
        "question_count": base.QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    construction = read_sealed_json(Path(args.construction))
    base._require(  # noqa: SLF001
        construction.sha256
        == require_sha256(
            args.expected_construction_sha256,
            "expected specialist v3 construction",
        ),
        "specialist v3 construction artifact changed",
    )
    base._validate_construction(  # noqa: SLF001
        construction,
        construction_format=CONSTRUCTION_FORMAT,
    )
    target_plan, plan_sha = base.reduced_cli._read_target_plan(  # noqa: SLF001
        Path(args.target_plan)
    )
    payload = base.build_target_audit(
        construction,
        target_plan,
        target_plan_file_sha256=plan_sha,
        construction_format=CONSTRUCTION_FORMAT,
        audit_format=AUDIT_FORMAT,
    )
    artifact, created = publish_sealed_json(Path(args.output), payload)
    return {
        "audit_sha256": artifact.sha256,
        "created": created,
        "new_provider_calls": 0,
        "question_count": base.QUESTION_COUNT,
        "retained_transformer_token_state_bytes": 0,
        "terminal_union_source_set_complete_questions": payload[
            "terminal_union_source_set_complete_questions"
        ],
        "terminal_union_source_target_hits": payload[
            "terminal_union_source_target_hits"
        ],
        "union_source_set_complete_questions": payload[
            "union_source_set_complete_questions"
        ],
        "union_source_target_count": payload["union_source_target_count"],
        "union_source_target_hits": payload["union_source_target_hits"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    construct = commands.add_parser("construct")
    construct.add_argument(
        "--frozen-reduced",
        type=Path,
        default=base.DEFAULT_FROZEN_REDUCED,
    )
    construct.add_argument(
        "--source-root", type=Path, default=base.DEFAULT_SOURCE_ROOT
    )
    construct.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    base._add_store_args(construct)  # noqa: SLF001

    audit = commands.add_parser("audit")
    audit.add_argument(
        "--construction",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / CONSTRUCTION_NAME,
    )
    audit.add_argument("--expected-construction-sha256", required=True)
    audit.add_argument(
        "--target-plan", type=Path, default=base.DEFAULT_TARGET_PLAN
    )
    audit.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_ROOT / AUDIT_NAME
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_construct(args) if args.command == "construct" else run_audit(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_FORMAT",
    "AUDIT_NAME",
    "CONSTRUCTION_FORMAT",
    "CONSTRUCTION_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "FORMAT",
    "build_parser",
    "main",
    "run_audit",
    "run_construct",
]
