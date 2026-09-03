#!/usr/bin/env python3
"""Materialize/replay the locked provider-free query-guided exhaustive scan."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from memory_condense.domain.integrity import file_sha256  # noqa: E402

from tools.matched_eval.artifacts import read_sealed_json  # noqa: E402
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    require_sha256,
)
from tools.matched_eval.population import (  # noqa: E402
    EXPECTED_QUESTION_COUNT,
    EXPECTED_RETRIEVAL_SHA256,
)
from tools.matched_eval.query_expansion import (  # noqa: E402
    LockedQueryExpansionContext,
    QueryExpansionPopulation,
    load_preflighted_query_expansion_population,
)
from tools.matched_eval.query_guided_scan import (  # noqa: E402
    QueryGuidedScanResult,
    materialize_query_guided_scan,
    replay_query_guided_scan,
)
DEFAULT_STORE_ROOT = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"
)
DEFAULT_RETRIEVAL = DEFAULT_STORE_ROOT / "retrieval.json"


DEFAULT_PARENT_OUTPUT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/"
    "matched-eval-spine-v2/s0-plus-query-expansion-v1"
)
DEFAULT_OUTPUT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/"
    "matched-eval-spine-v2/s0-plus-query-guided-scan-v1"
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


class LockedQueryGuidedScanError(MatchedEvalContractError):
    pass


def _require(ok: object, message: str) -> None:
    if not ok:
        raise LockedQueryGuidedScanError(message)


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
        "materialize", help="seal the cached provider-free construction arm"
    )
    _add_common(materialize)
    replay = commands.add_parser(
        "replay", help="rebuild and require byte-identical run/runtime bytes"
    )
    _add_common(replay)
    replay.add_argument("--expected-run-sha256", required=True)
    return parser


def _lightweight_store_context(
    population: QueryExpansionPopulation,
    *,
    retrieval_path: Path,
    store_root: Path,
) -> LockedQueryExpansionContext:
    """Bind sealed namespaces to stores without an extra SQLite inventory scan."""

    retrieval = read_sealed_json(retrieval_path)
    _require(
        retrieval.sha256 == population.source_population.retrieval_sha256,
        "locked retrieval changed",
    )
    raw_shards = retrieval.payload.get("shards")
    raw_questions = retrieval.payload.get("questions")
    _require(
        type(raw_shards) is list
        and type(raw_questions) is list
        and len(raw_questions) == len(population.rows)
        and all(type(row) is dict for row in (*raw_shards, *raw_questions)),
        "locked retrieval shard/question population changed",
    )
    namespace_by_receipt = {
        row.combined_store_receipt_sha256: row for row in population.namespaces
    }
    _require(
        len(namespace_by_receipt) == len(population.namespaces),
        "namespace store receipts must be unique",
    )
    store_by_namespace: dict[str, Path] = {}
    database_sha_by_namespace: dict[str, str] = {}
    index_sha_by_namespace: dict[str, str] = {}
    namespace_by_offset: dict[int, Any] = {}
    for raw in raw_shards:
        offset = raw.get("shard_offset")
        receipt_sha = raw.get("combined_store_receipt_sha256")
        receipt = raw.get("combined_store_receipt")
        _require(
            type(offset) is int
            and offset >= 0
            and offset % 10 == 0
            and type(receipt) is dict
            and receipt.get("receipt_sha256") == receipt_sha
            and receipt_sha in namespace_by_receipt,
            "frozen shard/store receipt changed",
        )
        namespace = namespace_by_receipt[str(receipt_sha)]
        _require(offset not in namespace_by_offset, "shard offset repeated")
        database_sha = require_sha256(
            receipt.get("target_database_sha256"), "frozen database SHA-256"
        )
        index_sha = require_sha256(
            receipt.get("target_index_sha256"), "frozen index SHA-256"
        )
        store = store_root / "shards" / f"offset-{offset:03d}" / "combined-store"
        database_path = store / "memory.db"
        index_path = store / "hnsw_index.bin"
        _require(
            database_path.is_file()
            and not database_path.is_symlink()
            and file_sha256(database_path) == database_sha
            and index_path.is_file()
            and not index_path.is_symlink()
            and file_sha256(index_path) == index_sha,
            f"frozen store bytes changed at offset {offset}",
        )
        namespace_by_offset[offset] = namespace
        store_by_namespace[namespace.namespace_id] = store
        database_sha_by_namespace[namespace.namespace_id] = database_sha
        index_sha_by_namespace[namespace.namespace_id] = index_sha

    offsets_by_question: dict[str, int] = {}
    for prompt, raw in zip(population.rows, raw_questions, strict=True):
        offset = raw.get("shard_offset")
        _require(
            type(offset) is int
            and offset in namespace_by_offset
            and raw.get("question_id") == prompt.source.packet.question_id
            and namespace_by_offset[offset].namespace_id
            == prompt.namespace.namespace_id,
            "question changed its frozen store binding",
        )
        offsets_by_question[prompt.source.packet.question_id] = offset
    context = LockedQueryExpansionContext(
        population=population,
        store_dirs_by_namespace=MappingProxyType(store_by_namespace),
        database_sha256_by_namespace=MappingProxyType(database_sha_by_namespace),
        index_sha256_by_namespace=MappingProxyType(index_sha_by_namespace),
        shard_offsets_by_question=MappingProxyType(offsets_by_question),
    )
    context.revalidate_store_bytes()
    return context


def _load_context(args: argparse.Namespace) -> LockedQueryExpansionContext:
    population, preflight = load_preflighted_query_expansion_population(
        args.retrieval,
        output_root=args.parent_output_root,
        expected_retrieval_sha256=args.expected_retrieval_sha256,
        expected_question_count=EXPECTED_QUESTION_COUNT,
    )
    expected = require_sha256(
        args.expected_parent_preflight_sha256, "expected parent preflight"
    )
    _require(preflight.sha256 == expected, "parent query preflight changed")
    return _lightweight_store_context(
        population,
        retrieval_path=args.retrieval,
        store_root=args.store_root,
    )


def _summary(
    result: QueryGuidedScanResult,
    *,
    command: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    payload = result.run_artifact.payload
    aggregate = payload["aggregate"]
    rows = payload["questions"]
    return {
        "admitted_candidate_count": aggregate["admitted_candidate_count"],
        "candidate_count": aggregate["candidate_count"],
        "command": command,
        "dedup_excluded_candidate_count": aggregate[
            "dedup_excluded_candidate_count"
        ],
        "elapsed_seconds": round(elapsed_seconds, 3),
        "gold_loaded": False,
        "logical_scanned_content_row_memberships": aggregate[
            "logical_scanned_content_row_memberships"
        ],
        "maximum_tokens_used": aggregate["maximum_tokens_used"],
        "mean_tokens_used": round(
            aggregate["total_tokens_used"] / len(rows), 2
        ),
        "new_provider_calls": 0,
        "physical_database_read_passes": payload[
            "physical_database_read_passes"
        ],
        "question_count": len(rows),
        "retained_transformer_token_state_bytes": 0,
        "run_artifact": result.run_artifact.path.as_posix(),
        "run_sha256": result.run_artifact.sha256,
        "runtime_ledger_artifact": result.runtime_ledger_artifact.path.as_posix(),
        "runtime_ledger_sha256": result.runtime_ledger_artifact.sha256,
        "selected_candidate_count": aggregate["selected_candidate_count"],
        "selected_second_span_count": aggregate["selected_second_span_count"],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    context = _load_context(args)
    common = {
        "parent_output_root": args.parent_output_root,
        "output_root": args.output_root,
        "expected_parent_preflight_sha256": args.expected_parent_preflight_sha256,
        "expected_parent_run_sha256": args.expected_parent_run_sha256,
        "expected_parent_runtime_ledger_sha256": (
            args.expected_parent_runtime_ledger_sha256
        ),
    }
    started = time.perf_counter()
    if args.command == "materialize":
        result = materialize_query_guided_scan(context, **common)
    elif args.command == "replay":
        result = replay_query_guided_scan(
            context,
            expected_run_sha256=require_sha256(
                args.expected_run_sha256, "expected scan run"
            ),
            **common,
        )
    else:  # pragma: no cover
        raise AssertionError("unreachable command")
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            _summary(result, command=args.command, elapsed_seconds=elapsed),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
