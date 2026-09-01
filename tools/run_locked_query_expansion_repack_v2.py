#!/usr/bin/env python3
"""Materialize/replay the locked provider-free query-expansion repack v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from tools.matched_eval.contracts import MatchedEvalContractError, require_sha256
from tools.matched_eval.population import (
    EXPECTED_QUESTION_COUNT,
    EXPECTED_RETRIEVAL_SHA256,
)
from tools.matched_eval.query_expansion import (
    LockedQueryExpansionContext,
    load_locked_query_expansion_context,
    load_preflighted_query_expansion_population,
)
from tools.matched_eval.query_expansion_repack_v2 import (
    RUN_NAME,
    QueryExpansionRepackResult,
    materialize_query_expansion_repack_v2,
    replay_query_expansion_repack_v2,
)
from tools.run_locked_query_expansion import DEFAULT_RETRIEVAL, DEFAULT_STORE_ROOT


DEFAULT_PARENT_OUTPUT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/"
    "matched-eval-spine-v2/s0-plus-query-expansion-v1"
)
DEFAULT_OUTPUT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/"
    "matched-eval-spine-v2/s0-plus-query-expansion-repack-v2"
)
EXPECTED_PARENT_PREFLIGHT_SHA256 = (
    "dc357e4a4e946c541ca5cb278824c376692ba4e4a97a5947c5b18e8da86c5487"
)
EXPECTED_PARENT_RUN_SHA256 = (
    "68f7c0c073c405e33cf019c75e69db1ee5be9b9f3dd84f13cd5a427e6508ba07"
)
EXPECTED_PARENT_RUNTIME_LEDGER_SHA256 = (
    "16d5ceedee9a86d7c719d3d66538a4d8fa23cf8fbee5763097df69f28afc7c94"
)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--parent-output-root", type=Path, default=DEFAULT_PARENT_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-retrieval-sha256", default=EXPECTED_RETRIEVAL_SHA256
    )
    parser.add_argument(
        "--expected-parent-preflight-sha256",
        default=EXPECTED_PARENT_PREFLIGHT_SHA256,
    )
    parser.add_argument(
        "--expected-parent-run-sha256", default=EXPECTED_PARENT_RUN_SHA256
    )
    parser.add_argument(
        "--expected-parent-runtime-ledger-sha256",
        default=EXPECTED_PARENT_RUNTIME_LEDGER_SHA256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser(
        "materialize", help="seal the provider-free child run"
    )
    _add_common(materialize)
    replay = commands.add_parser(
        "replay", help="rebuild and require byte-identical child artifacts"
    )
    _add_common(replay)
    replay.add_argument("--expected-run-sha256", required=True)
    return parser


def _load_context(args: argparse.Namespace) -> LockedQueryExpansionContext:
    population, preflight = load_preflighted_query_expansion_population(
        args.retrieval,
        output_root=args.parent_output_root,
        expected_retrieval_sha256=args.expected_retrieval_sha256,
        expected_question_count=EXPECTED_QUESTION_COUNT,
    )
    expected_preflight = require_sha256(
        args.expected_parent_preflight_sha256,
        "expected parent preflight SHA-256",
    )
    if preflight.sha256 != expected_preflight:
        raise MatchedEvalContractError("locked parent preflight SHA-256 changed")
    context = load_locked_query_expansion_context(
        args.retrieval,
        store_root=args.store_root,
        expected_retrieval_sha256=args.expected_retrieval_sha256,
        expected_question_count=EXPECTED_QUESTION_COUNT,
        budget=population.budget,
        include_s0_evidence=population.include_s0_evidence,
    )
    if context.population.preflight_projection() != population.preflight_projection():
        raise MatchedEvalContractError(
            "locked stores differ from the sealed parent/S0 population"
        )
    context.revalidate_store_bytes()
    return context


def _summary(
    result: QueryExpansionRepackResult,
    *,
    command: str,
) -> dict[str, Any]:
    rows = result.run_artifact.payload["questions"]
    membership = result.run_artifact.payload["source_membership_coverage"]
    return {
        "admission_rescue_memberships": membership[
            "admission_rescue_memberships"
        ],
        "admission_loss_memberships": membership["admission_loss_memberships"],
        "admitted_candidate_count": sum(
            len(row["admitted_candidate_ids"]) for row in rows
        ),
        "candidate_retrieval_calls": 0,
        "command": command,
        "coverage_primary_count": sum(
            len(row["coverage_primary_candidate_ids"]) for row in rows
        ),
        "dedup_excluded_count": sum(
            len(row["dedup_excluded_candidate_ids"]) for row in rows
        ),
        "gold_loaded": False,
        "max_tokens_used": max(int(row["tokens_used"]) for row in rows),
        "mean_tokens_used": round(
            sum(int(row["tokens_used"]) for row in rows) / len(rows), 2
        ),
        "new_provider_calls": 0,
        "question_count": len(rows),
        "retained_transformer_token_state_bytes": 0,
        "run_artifact": result.run_artifact.path.as_posix(),
        "run_sha256": result.run_artifact.sha256,
        "runtime_ledger_artifact": result.runtime_ledger_artifact.path.as_posix(),
        "runtime_ledger_sha256": result.runtime_ledger_artifact.sha256,
        "selected_candidate_count": sum(
            len(row["selected_before_dedup_candidate_ids"]) for row in rows
        ),
        "selection_rescue_memberships": membership[
            "selection_rescue_memberships"
        ],
        "selection_loss_memberships": membership["selection_loss_memberships"],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    context = _load_context(args)
    common = {
        "parent_output_root": args.parent_output_root,
        "output_root": args.output_root,
        "expected_parent_preflight_sha256": (
            args.expected_parent_preflight_sha256
        ),
        "expected_parent_run_sha256": args.expected_parent_run_sha256,
        "expected_parent_runtime_ledger_sha256": (
            args.expected_parent_runtime_ledger_sha256
        ),
    }
    if args.command == "materialize":
        result = materialize_query_expansion_repack_v2(context, **common)
    elif args.command == "replay":
        expected = require_sha256(args.expected_run_sha256, "expected repack run")
        result = replay_query_expansion_repack_v2(
            context,
            expected_run_sha256=expected,
            **common,
        )
    else:  # pragma: no cover
        raise AssertionError("unreachable command")
    print(json.dumps(_summary(result, command=args.command), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
