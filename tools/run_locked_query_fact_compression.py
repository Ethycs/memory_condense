#!/usr/bin/env python3
"""Run the split locked-100 query-expansion fact-compression lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from tools.matched_eval.contracts import MatchedEvalContractError, require_sha256  # noqa: E402
from tools.matched_eval.live import DEFAULT_API_KEY_ENV, _make_provider_client  # noqa: E402
from tools.matched_eval.population import (  # noqa: E402
    EXPECTED_QUESTION_COUNT,
    EXPECTED_RETRIEVAL_SHA256,
)
from tools.matched_eval.query_fact_adapter import (  # noqa: E402
    load_query_fact_population,
)
from tools.matched_eval.query_fact_compression import (  # noqa: E402
    DEFAULT_GATEWAY_URL,
    DEFAULT_MAX_FACTS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    QueryFactCompressionSettings,
    load_query_fact_compression_journals,
    materialize_query_fact_compression,
    preflight_query_fact_compression,
    replay_query_fact_compression,
    run_query_fact_compression_provider,
)


DEFAULT_RETRIEVAL_ROOT = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"
)
DEFAULT_RETRIEVAL = DEFAULT_RETRIEVAL_ROOT / "retrieval.json"
DEFAULT_QUERY_ROOT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/"
    "matched-eval-spine-v2/s0-plus-query-expansion-v1"
)
DEFAULT_QUERY_PREFLIGHT = DEFAULT_QUERY_ROOT / "query-expansion-preflight.json"
DEFAULT_QUERY_RUN = DEFAULT_QUERY_ROOT / "query-expansion-run.json"
DEFAULT_OUTPUT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/"
    "matched-eval-spine-v2/s0-plus-query-expansion-routed-facts-v1"
)

EXPECTED_SOURCE_POPULATION_ID = (
    "886e14025a0aedf5a9ba673be8ffc9183acc080b97645adc2b6dd003019438bf"
)
EXPECTED_QUERY_PREFLIGHT_SHA256 = (
    "dc357e4a4e946c541ca5cb278824c376692ba4e4a97a5947c5b18e8da86c5487"
)
EXPECTED_QUERY_RUN_SHA256 = (
    "68f7c0c073c405e33cf019c75e69db1ee5be9b9f3dd84f13cd5a427e6508ba07"
)
EXPECTED_QUERY_POPULATION_ID = (
    "5030a5ae9ce83be7ae39ad290b492db278c8f090303730766db21edecae33b5e"
)
EXPECTED_QUERY_PROMPT_POPULATION_SHA256 = (
    "c88a09f1817404d5f29e0cca77fdb260b1479bf004bb8339d543376a3741c02d"
)
EXPECTED_ADAPTER_POPULATION_ID = (
    "d3a36449a9fa5aefcdd2c4de243432ef939701bfa5ad558b79a175644a2624f8"
)
EXPECTED_ADAPTER_PREFLIGHT_ID = (
    "e3de19e470f26fe2bfd717af4f521dea4dac66299b52cbd6697d66990413b099"
)


def _add_population_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument(
        "--query-preflight", type=Path, default=DEFAULT_QUERY_PREFLIGHT
    )
    parser.add_argument("--query-run", type=Path, default=DEFAULT_QUERY_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-retrieval-sha256", default=EXPECTED_RETRIEVAL_SHA256
    )
    parser.add_argument(
        "--expected-source-population-id", default=EXPECTED_SOURCE_POPULATION_ID
    )
    parser.add_argument(
        "--expected-query-preflight-sha256",
        default=EXPECTED_QUERY_PREFLIGHT_SHA256,
    )
    parser.add_argument(
        "--expected-query-run-sha256", default=EXPECTED_QUERY_RUN_SHA256
    )
    parser.add_argument(
        "--expected-query-population-id", default=EXPECTED_QUERY_POPULATION_ID
    )
    parser.add_argument(
        "--expected-query-prompt-population-sha256",
        default=EXPECTED_QUERY_PROMPT_POPULATION_SHA256,
    )
    parser.add_argument(
        "--expected-adapter-population-id",
        default=EXPECTED_ADAPTER_POPULATION_ID,
    )
    parser.add_argument(
        "--expected-adapter-preflight-id",
        default=EXPECTED_ADAPTER_PREFLIGHT_ID,
    )
    parser.add_argument(
        "--expected-question-count", type=int, default=EXPECTED_QUESTION_COUNT
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--max-prompt-tokens", type=int, default=8_000)
    parser.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS
    )
    parser.add_argument("--max-facts", type=int, default=DEFAULT_MAX_FACTS)
    parser.add_argument("--max-concurrency", type=int, default=4)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="seal the adapter and routed Terra prompt population"
    )
    _add_population_arguments(preflight)
    _add_runtime_arguments(preflight)

    provider = subparsers.add_parser(
        "provider-run",
        help="fill only immutable Terra response journals; no stores or DB",
    )
    _add_population_arguments(provider)
    _add_runtime_arguments(provider)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, default=0)
    provider.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)

    materialize = subparsers.add_parser(
        "materialize",
        help="parse journals and seal exact cited facts plus runtime ledger",
    )
    _add_population_arguments(materialize)
    _add_runtime_arguments(materialize)

    replay = subparsers.add_parser(
        "replay", help="client-free byte-identical compression and ledger replay"
    )
    _add_population_arguments(replay)
    _add_runtime_arguments(replay)
    replay.add_argument("--expected-compression-sha256", required=True)
    replay.add_argument("--expected-runtime-ledger-sha256", required=True)
    return parser


def _settings(args: argparse.Namespace) -> QueryFactCompressionSettings:
    return QueryFactCompressionSettings(
        model=str(args.model),
        gateway_url=str(args.gateway_url),
        max_prompt_tokens=int(args.max_prompt_tokens),
        max_output_tokens=int(args.max_output_tokens),
        max_facts=int(args.max_facts),
        max_concurrency=int(args.max_concurrency),
    )


def _load_population(args: argparse.Namespace):
    for value, label in (
        (args.expected_adapter_population_id, "expected adapter population"),
        (args.expected_adapter_preflight_id, "expected adapter preflight"),
    ):
        require_sha256(str(value), label)
    population = load_query_fact_population(
        args.retrieval,
        query_preflight_path=args.query_preflight,
        query_run_path=args.query_run,
        expected_retrieval_sha256=str(args.expected_retrieval_sha256),
        expected_source_population_id=str(args.expected_source_population_id),
        expected_query_preflight_sha256=str(
            args.expected_query_preflight_sha256
        ),
        expected_query_run_sha256=str(args.expected_query_run_sha256),
        expected_query_population_id=str(args.expected_query_population_id),
        expected_query_prompt_population_sha256=str(
            args.expected_query_prompt_population_sha256
        ),
        expected_question_count=int(args.expected_question_count),
        max_prompt_tokens=int(args.max_prompt_tokens),
    )
    if population.population_id != str(args.expected_adapter_population_id):
        raise MatchedEvalContractError("query-fact adapter population ID changed")
    if population.preflight_identity_sha256 != str(
        args.expected_adapter_preflight_id
    ):
        raise MatchedEvalContractError("query-fact adapter preflight identity changed")
    return population


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    population = _load_population(args)
    artifact = preflight_query_fact_compression(
        population,
        output_root=args.output_root,
        settings=_settings(args),
    )
    return {
        "command": "preflight",
        "preflight_path": str(artifact.path),
        "preflight_sha256": artifact.sha256,
        "required_authorized_provider_calls": artifact.payload[
            "required_authorized_provider_calls"
        ],
        "provider_calls": 0,
    }


def _provider(args: argparse.Namespace) -> dict[str, Any]:
    # The locked command knows the expected unique population before loading
    # any artifact, so malformed authorization fails before journal creation.
    if not args.enable_provider:
        raise MatchedEvalContractError("provider-run requires --enable-provider")
    if args.authorized_provider_calls != args.expected_question_count:
        raise MatchedEvalContractError(
            "--authorized-provider-calls must exactly equal "
            f"{args.expected_question_count}"
        )
    population = _load_population(args)
    load_dotenv()
    api_key = os.environ.get(str(args.api_key_env), "").strip()
    if not api_key:
        raise RuntimeError(f"provider API key is empty: {args.api_key_env}")
    client = _make_provider_client(api_key, str(args.gateway_url))
    result = run_query_fact_compression_provider(
        population,
        output_root=args.output_root,
        enable_provider=True,
        authorized_provider_calls=int(args.authorized_provider_calls),
        client=client,
        settings=_settings(args),
    )
    return {
        "checkpoint_hits": result.checkpoint_hits,
        "command": "provider-run",
        "physical_provider_calls": result.physical_provider_calls,
        "preflight_sha256": result.preflight_artifact.sha256,
        "required_authorized_provider_calls": (
            population.compression_prompt_population.unique_prompt_count
        ),
    }


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    population = _load_population(args)
    settings = _settings(args)
    completion = load_query_fact_compression_journals(
        population,
        output_root=args.output_root,
        settings=settings,
    )
    result = materialize_query_fact_compression(
        population,
        output_root=args.output_root,
        completion_batch=completion.batch,
        settings=settings,
    )
    return {
        "checkpoint_hits": result.checkpoint_hits,
        "command": "materialize",
        "compression_sha256": result.compression_artifact.sha256,
        "physical_provider_calls": result.physical_provider_calls,
        "runtime_ledger_sha256": result.runtime_ledger_artifact.sha256,
        "status_counts": result.compression_artifact.payload["status_counts"],
    }


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    population = _load_population(args)
    result = replay_query_fact_compression(
        population,
        output_root=args.output_root,
        expected_compression_sha256=str(args.expected_compression_sha256),
        expected_runtime_ledger_sha256=str(args.expected_runtime_ledger_sha256),
        settings=_settings(args),
    )
    return {
        "checkpoint_hits": result.checkpoint_hits,
        "command": "replay",
        "compression_sha256": result.compression_artifact.sha256,
        "physical_provider_calls": result.physical_provider_calls,
        "runtime_ledger_sha256": result.runtime_ledger_artifact.sha256,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {
        "preflight": _preflight,
        "provider-run": _provider,
        "materialize": _materialize,
        "replay": _replay,
    }
    result = handlers[args.command](args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
