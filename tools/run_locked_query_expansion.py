#!/usr/bin/env python3
"""Split provider and store phases for locked-100 query expansion."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repository / "src"), str(repository)]

from dotenv import load_dotenv  # noqa: E402

from tools.matched_eval.artifacts import read_sealed_json  # noqa: E402
from tools.matched_eval.contracts import (  # noqa: E402
    MatchedEvalContractError,
    require_sha256,
)
from tools.matched_eval.live import (  # noqa: E402
    DEFAULT_API_KEY_ENV,
    _make_provider_client,
)
from tools.matched_eval.population import (  # noqa: E402
    EXPECTED_QUESTION_COUNT,
    EXPECTED_RETRIEVAL_SHA256,
)
from tools.matched_eval.query_expansion import (  # noqa: E402
    CHECKPOINT_DIR_NAME,
    DEFAULT_GATEWAY_URL,
    RUN_NAME,
    ExistingPartitionHybridSearch,
    FrozenPartitionSearch,
    LockedQueryExpansionContext,
    QueryExpansionPopulation,
    load_locked_query_expansion_context,
    load_preflighted_query_expansion_population,
    load_query_expansion_provider_journals,
    materialize_query_expansion,
    preflight_query_expansion,
    replay_query_expansion,
    run_query_expansion_provider,
)


DEFAULT_STORE_ROOT = Path(
    "eval_results/longmemeval-1m-recall-guarded-cumulative-validation-20260822"
)
DEFAULT_RETRIEVAL = DEFAULT_STORE_ROOT / "retrieval.json"
DEFAULT_OUTPUT = Path(
    "eval_results/longmemeval-1m-locked-retrieval-mechanism-arms-20260826/"
    "matched-eval-spine-v2/s0-plus-query-expansion-v1"
)
DEFAULT_POLICY = Path(
    "docs/10 - Research Log/data/"
    "longmemeval-qwen-choice-coverage-operational-validation-v3.json"
)
DEFAULT_QWEN_PREFIX = Path(".cache/models/Qwen3-8B")
EXPECTED_NAMESPACE_COUNT = 10
EXPECTED_PREFLIGHT_SHA256 = (
    "dc357e4a4e946c541ca5cb278824c376692ba4e4a97a5947c5b18e8da86c5487"
)


def _add_prompt_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-retrieval-sha256",
        default=EXPECTED_RETRIEVAL_SHA256,
    )


def _add_runtime_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)


def _add_expected_preflight_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--expected-preflight-sha256",
        default=EXPECTED_PREFLIGHT_SHA256,
    )


def _add_store_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--qwen-prefix",
        type=Path,
        default=DEFAULT_QWEN_PREFIX,
        help="locator required by the pinned embedding binding; Qwen is not loaded",
    )
    parser.add_argument("--device", default="cuda")
    _add_runtime_identity_arguments(parser)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight",
        help="seal exact prompts and complete store namespaces without a provider",
    )
    _add_prompt_arguments(preflight)
    preflight.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    preflight.add_argument("--include-s0-evidence", action="store_true")

    provider = subparsers.add_parser(
        "provider-run",
        help="fill completion journals; never open memory.db, ANN, or BGE",
    )
    _add_prompt_arguments(provider)
    _add_expected_preflight_argument(provider)
    _add_runtime_identity_arguments(provider)
    provider.add_argument("--enable-provider", action="store_true")
    provider.add_argument("--authorized-provider-calls", type=int, default=0)
    provider.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)

    for name in ("materialize", "run"):
        materialize = subparsers.add_parser(
            name,
            help="client-free store materialization from 100 response journals",
        )
        _add_prompt_arguments(materialize)
        _add_expected_preflight_argument(materialize)
        _add_store_arguments(materialize)

    replay = subparsers.add_parser(
        "replay",
        help="client-free reconstruction of the sealed run and ledger",
    )
    _add_prompt_arguments(replay)
    _add_expected_preflight_argument(replay)
    _add_store_arguments(replay)
    replay.add_argument("--expected-run-sha256", required=True)
    return parser


def _validate_population(population: QueryExpansionPopulation) -> None:
    if (
        len(population.rows) != EXPECTED_QUESTION_COUNT
        or population.prompt_population.logical_prompt_count
        != EXPECTED_QUESTION_COUNT
        or population.prompt_population.unique_prompt_count
        != EXPECTED_QUESTION_COUNT
        or len(population.namespaces) != EXPECTED_NAMESPACE_COUNT
    ):
        raise MatchedEvalContractError(
            "locked query expansion requires exactly 100 unique prompts "
            "across ten frozen stores"
        )


def _load_preflight_population(
    args: argparse.Namespace,
) -> tuple[QueryExpansionPopulation, Any]:
    population, artifact = load_preflighted_query_expansion_population(
        args.retrieval,
        output_root=args.output_root,
        expected_retrieval_sha256=args.expected_retrieval_sha256,
        expected_question_count=EXPECTED_QUESTION_COUNT,
    )
    _validate_population(population)
    expected_preflight = require_sha256(
        args.expected_preflight_sha256,
        "expected preflight SHA-256",
    )
    if artifact.sha256 != expected_preflight:
        raise MatchedEvalContractError("sealed query-expansion preflight SHA-256 changed")
    if artifact.payload.get("required_authorized_provider_calls") != 100:
        raise MatchedEvalContractError("sealed preflight is not the exact 100-call run")
    return population, artifact


def _load_store_context(
    args: argparse.Namespace,
    population: QueryExpansionPopulation,
) -> LockedQueryExpansionContext:
    context = load_locked_query_expansion_context(
        args.retrieval,
        store_root=args.store_root,
        expected_retrieval_sha256=args.expected_retrieval_sha256,
        expected_question_count=EXPECTED_QUESTION_COUNT,
        budget=population.budget,
        include_s0_evidence=population.include_s0_evidence,
    )
    _validate_population(context.population)
    if context.population.preflight_projection() != population.preflight_projection():
        raise MatchedEvalContractError(
            "store-backed population differs from the sealed prompt population"
        )
    context.revalidate_store_bytes()
    return context


def _require_provider_authorization(args: argparse.Namespace) -> None:
    if args.enable_provider is not True:
        raise MatchedEvalContractError(
            "locked query expansion requires --enable-provider"
        )
    if (
        type(args.authorized_provider_calls) is not int
        or args.authorized_provider_calls != EXPECTED_QUESTION_COUNT
    ):
        raise MatchedEvalContractError(
            "--authorized-provider-calls must exactly equal 100"
        )


@contextmanager
def _open_locked_retrievers(
    context: LockedQueryExpansionContext,
    *,
    policy_path: Path,
    qwen_prefix: Path,
    device: str,
) -> Iterator[Mapping[str, FrozenPartitionSearch]]:
    """Open ten verified stores read-only over one shared live embedder."""

    from memory_condense.application.condenser import MemoryCondenser
    from memory_condense.eval._recall_guarded_cumulative_validation_shard import (
        load_frozen_validation_policy,
    )
    from memory_condense.eval.recall_guarded_cumulative import (
        causal_graph_context_budget,
    )
    from memory_condense.eval.recall_guarded_cumulative_1m_source import (
        current_source_binding,
    )

    context.revalidate_store_bytes()
    policy = load_frozen_validation_policy(policy_path, device=str(device))
    _source_config, binding = current_source_binding(
        policy.config,
        qwen_model_dir=Path(qwen_prefix),
    )
    embedder = binding.embedder
    condensers: list[Any] = []
    retrievers: dict[str, FrozenPartitionSearch] = {}
    try:
        graph_budget = causal_graph_context_budget(policy.config.retrieval)
        for namespace in context.population.namespaces:
            condenser = MemoryCondenser(
                data_dir=context.store_dirs_by_namespace[namespace.namespace_id],
                chunker_min_tokens=policy.config.chunker.min_tokens,
                chunker_max_tokens=policy.config.chunker.max_tokens,
                auto_extract=False,
                budget=graph_budget,
                embedder=embedder,
                persist_index_on_close=False,
                retriever_max_elements=max(1, len(namespace.chunk_to_source)),
                read_only=True,
            )
            condensers.append(condenser)
            retrievers[namespace.namespace_id] = ExistingPartitionHybridSearch(
                condenser,
                namespace,
            )
        if set(retrievers) != set(context.store_dirs_by_namespace):
            raise MatchedEvalContractError(
                "opened retrievers do not cover every frozen namespace"
            )
        context.revalidate_store_bytes()
        yield MappingProxyType(retrievers)
    finally:
        cleanup_errors: list[BaseException] = []
        for condenser in reversed(condensers):
            try:
                condenser.close()
            except BaseException as exc:  # pragma: no cover
                cleanup_errors.append(exc)
        try:
            embedder.close()
        except BaseException as exc:  # pragma: no cover
            cleanup_errors.append(exc)
        if cleanup_errors and sys.exc_info()[0] is None:
            raise RuntimeError("failed to close query-expansion resources") from (
                cleanup_errors[0]
            )


def _provider_locked(args: argparse.Namespace) -> dict[str, Any]:
    # This gate precedes population loading, client construction, and journals.
    _require_provider_authorization(args)
    population, preflight = _load_preflight_population(args)
    load_dotenv()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise MatchedEvalContractError(
            f"provider API key is empty: {args.api_key_env}"
        )
    client = _make_provider_client(api_key, str(args.gateway_url))
    try:
        result = run_query_expansion_provider(
            population,
            output_root=args.output_root,
            enable_provider=True,
            authorized_provider_calls=EXPECTED_QUESTION_COUNT,
            client=client,
            max_concurrency=args.max_concurrency,
            gateway_url=str(args.gateway_url),
        )
    except BaseException:
        close = getattr(client, "close", None)
        if callable(close):
            close()
        raise
    checkpoint = Path(args.output_root) / CHECKPOINT_DIR_NAME
    return {
        "checkpoint_hits": result.checkpoint_hits,
        "checkpoint_root": checkpoint.as_posix(),
        "command": "provider-run",
        "gold_loaded": False,
        "memory_store_opened": False,
        "physical_provider_calls": result.physical_provider_calls,
        "preflight_sha256": preflight.sha256,
        "request_journal_count": len(tuple(checkpoint.glob("*.request.json"))),
        "response_journal_count": len(tuple(checkpoint.glob("*.response.json"))),
    }


def _materialize_locked(args: argparse.Namespace) -> dict[str, Any]:
    population, preflight = _load_preflight_population(args)
    # Require every immutable response before reading a store or loading BGE.
    completions = load_query_expansion_provider_journals(
        population,
        output_root=args.output_root,
        max_concurrency=args.max_concurrency,
        gateway_url=str(args.gateway_url),
    )
    context = _load_store_context(args, population)
    output = Path(args.output_root)
    existing = output / RUN_NAME
    with _open_locked_retrievers(
        context,
        policy_path=args.policy,
        qwen_prefix=args.qwen_prefix,
        device=args.device,
    ) as retrievers:
        if existing.exists():
            source = read_sealed_json(existing)
            result = replay_query_expansion(
                context.population,
                output_root=output,
                retrievers_by_namespace=retrievers,
                expected_run_sha256=source.sha256,
                max_concurrency=args.max_concurrency,
                gateway_url=str(args.gateway_url),
            )
            terminal_replay = True
        else:
            result = materialize_query_expansion(
                context.population,
                output_root=output,
                retrievers_by_namespace=retrievers,
                completion_batch=completions.batch,
            )
            terminal_replay = False
    return {
        "checkpoint_hits": completions.checkpoint_hits,
        "command": args.command,
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "preflight_sha256": preflight.sha256,
        "provider_client_created": False,
        "run_artifact": result.run_artifact.path.as_posix(),
        "run_sha256": result.run_artifact.sha256,
        "runtime_ledger_artifact": result.runtime_ledger_artifact.path.as_posix(),
        "runtime_ledger_sha256": result.runtime_ledger_artifact.sha256,
        "terminal_run_replayed": terminal_replay,
    }


def _replay_locked(args: argparse.Namespace) -> dict[str, Any]:
    expected = require_sha256(args.expected_run_sha256, "expected run SHA-256")
    population, preflight = _load_preflight_population(args)
    completions = load_query_expansion_provider_journals(
        population,
        output_root=args.output_root,
        max_concurrency=args.max_concurrency,
        gateway_url=str(args.gateway_url),
    )
    context = _load_store_context(args, population)
    with _open_locked_retrievers(
        context,
        policy_path=args.policy,
        qwen_prefix=args.qwen_prefix,
        device=args.device,
    ) as retrievers:
        result = replay_query_expansion(
            context.population,
            output_root=args.output_root,
            retrievers_by_namespace=retrievers,
            expected_run_sha256=expected,
            max_concurrency=args.max_concurrency,
            gateway_url=str(args.gateway_url),
        )
    return {
        "checkpoint_hits": completions.checkpoint_hits,
        "command": "replay",
        "gold_loaded": False,
        "physical_provider_calls": 0,
        "preflight_sha256": preflight.sha256,
        "provider_client_created": False,
        "run_sha256": result.run_artifact.sha256,
        "runtime_ledger_sha256": result.runtime_ledger_artifact.sha256,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        context = load_locked_query_expansion_context(
            args.retrieval,
            store_root=args.store_root,
            expected_retrieval_sha256=args.expected_retrieval_sha256,
            expected_question_count=EXPECTED_QUESTION_COUNT,
            include_s0_evidence=args.include_s0_evidence,
        )
        _validate_population(context.population)
        artifact = preflight_query_expansion(
            context.population,
            output_root=args.output_root,
        )
        payload = {
            "artifact": artifact.path.as_posix(),
            "gold_loaded": False,
            "namespace_count": len(context.population.namespaces),
            "preflight_sha256": artifact.sha256,
            "provider_calls": 0,
            "question_count": len(context.population.rows),
            "required_authorized_provider_calls": 100,
        }
    elif args.command == "provider-run":
        payload = _provider_locked(args)
    elif args.command in {"materialize", "run"}:
        payload = _materialize_locked(args)
    elif args.command == "replay":
        payload = _replay_locked(args)
    else:  # pragma: no cover
        raise AssertionError("unreachable command")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
