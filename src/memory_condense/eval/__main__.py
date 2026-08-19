"""Executable compatibility facade for the evaluation pipeline.

The CLI implementations live in focused modules.  This facade intentionally
retains the historical helper names because the evaluation safety tests patch
them here to prove provider-call budgets, immutable provenance, and blind
cache preparation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from memory_condense.eval import benchmark_mode as _benchmark_mode
from memory_condense.eval import cache_mode as _cache_mode
from memory_condense.eval import cli_config as _cli_config
from memory_condense.eval import cli_parser as _cli_parser
from memory_condense.eval import offline_modes as _offline_modes
from memory_condense.eval import policy_gate as _policy_gate
from memory_condense.eval import provider_runtime as _provider_runtime
from memory_condense.eval import replay_mode as _replay_mode
from memory_condense.eval import runtime_controls as _runtime_controls
from memory_condense.eval.analysis import (
    compare_runs,
    load_run,
    print_comparison,
    to_csv,
)
from memory_condense.eval.benchmark import (
    print_benchmark_summary,
    run_benchmark,
    save_benchmark_report,
)
from memory_condense.eval.compiled_cache import sample_sha256
from memory_condense.eval.recall import print_recall_report, run_recall
from memory_condense.eval.report import (
    print_run_summary,
    print_sweep_table,
    save_run_result,
    save_sweep_report,
)
from memory_condense.eval.reproducibility import (
    environment_lock_sha256,
    file_sha256,
    implementation_sha256,
    project_root,
)
from memory_condense.eval.runner import run_eval
from memory_condense.eval.schemas import EvalConfig
from memory_condense.eval.sufficiency import (
    print_sufficiency_report,
    run_sufficiency_audit,
)
from memory_condense.eval.sweep import run_sweep
from memory_condense.ingest.loader import load_benchmark, load_directory


# Parser/config compatibility exports.
_StoreExplicitValue = _cli_parser._StoreExplicitValue
build_parser = _cli_parser.build_parser
config_from_args = _cli_config.config_from_args


# Provider helpers.  The factories inject the facade client constructor at
# call time so monkeypatching this module continues to govern every route.
_content = _provider_runtime._content
_BINARY_JUDGE_VERDICT = _provider_runtime._BINARY_JUDGE_VERDICT
_parse_binary_judge_verdict = _provider_runtime._parse_binary_judge_verdict
_make_central_dev_client = _provider_runtime._make_central_dev_client


def _make_answer_fn(model: str, *, retries: int = 0):
    return _provider_runtime._make_answer_fn(
        model,
        retries=retries,
        client_factory=_make_central_dev_client,
    )


def _make_judge_fn(model: str, *, retries: int = 0):
    return _provider_runtime._make_judge_fn(
        model,
        retries=retries,
        client_factory=_make_central_dev_client,
    )


def _make_sufficiency_fn(model: str, *, retries: int = 0):
    return _provider_runtime._make_sufficiency_fn(
        model,
        retries=retries,
        client_factory=_make_central_dev_client,
    )


# Locked split and frozen-policy helpers.
_apply_locked_split = _policy_gate._apply_locked_split
_apply_sample_offset = _policy_gate._apply_sample_offset
_planned_provider_calls = _policy_gate._planned_provider_calls
_benchmark_evaluation_identity = _policy_gate._benchmark_evaluation_identity
_coverage_prefix_policy_identity = _policy_gate._coverage_prefix_policy_identity
_policy_retrieval_identity = _policy_gate._policy_retrieval_identity
_optional_policy_sha256 = _policy_gate._optional_policy_sha256
_policy_repository_file = _policy_gate._policy_repository_file


def _verified_policy_sha256(
    path: Path | None,
    *,
    config: EvalConfig,
    dataset_sha256: str,
    split_manifest: str | None,
    active_split: str | None = None,
    active_implementation_sha256: str | None = None,
    active_environment_lock_sha256: str | None = None,
    repository_root: str | Path | None = None,
    evaluation_identity: dict[str, object] | None = None,
    prepare_only: bool = False,
) -> str:
    return _policy_gate._verified_policy_sha256(
        path,
        config=config,
        dataset_sha256=dataset_sha256,
        split_manifest=split_manifest,
        active_split=active_split,
        active_implementation_sha256=active_implementation_sha256,
        active_environment_lock_sha256=active_environment_lock_sha256,
        repository_root=repository_root,
        evaluation_identity=evaluation_identity,
        prepare_only=prepare_only,
        implementation_sha256_fn=implementation_sha256,
        environment_lock_sha256_fn=environment_lock_sha256,
    )


def _assert_implementation_unchanged(expected_sha256: str) -> None:
    if implementation_sha256() != expected_sha256:
        raise RuntimeError("implementation changed during benchmark run")


# Ingest and transient-model controls.
_benchmark_ingest_fn = _runtime_controls._benchmark_ingest_fn
_load_candidate_reranker = _runtime_controls._load_candidate_reranker
_attach_candidate_reranker = _runtime_controls._attach_candidate_reranker
_LazyQwenPrefixCoverageSelector = (
    _runtime_controls._LazyQwenPrefixCoverageSelector
)
_LazyLocalINICoverageSelector = (
    _runtime_controls._LazyLocalINICoverageSelector
)
_LazyCrossEncoderCoverageSelector = (
    _runtime_controls._LazyCrossEncoderCoverageSelector
)
_load_coverage_selector = _runtime_controls._load_coverage_selector
_attach_coverage_selector = _runtime_controls._attach_coverage_selector
_attach_runtime_controls = _runtime_controls._attach_runtime_controls
_reserve_embedding_device_for_transient_models = (
    _runtime_controls._reserve_embedding_device_for_transient_models
)


# Blind cache helpers.
_validate_prepare_cache_args = _cache_mode._validate_prepare_cache_args
_validated_blind_cache_receipts = (
    _cache_mode._validated_blind_cache_receipts
)
_causal_count_timing_metadata = _cache_mode._causal_count_timing_metadata


def _facade():
    """Return the live facade namespace used for patch-compatible injection."""

    return sys.modules[__name__]


def run_compare(args: argparse.Namespace) -> None:
    return _offline_modes.run_compare(args, runtime=_facade())


def run_answer_recall(args: argparse.Namespace) -> None:
    return _offline_modes.run_answer_recall(args, runtime=_facade())


def run_sufficiency_mode(args: argparse.Namespace) -> None:
    return _offline_modes.run_sufficiency_mode(args, runtime=_facade())


def run_prepare_cache_only(args: argparse.Namespace) -> dict[str, object]:
    return _cache_mode.run_prepare_cache_only(args, runtime=_facade())


def run_benchmark_mode(args: argparse.Namespace) -> None:
    return _benchmark_mode.run_benchmark_mode(args, runtime=_facade())


def run_replay_mode(args: argparse.Namespace) -> None:
    return _replay_mode.run_replay_mode(args, runtime=_facade())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    modes = [
        bool(args.compare),
        bool(args.answer_recall),
        bool(args.sufficiency_audit),
        bool(args.benchmark_file),
        bool(args.conversation_dir),
    ]
    if sum(modes) > 1:
        parser.error(
            "--compare, --answer-recall, --sufficiency-audit, --benchmark-file, "
            "and --conversation-dir are mutually exclusive"
        )
    if args.prepare_cache_only and not args.benchmark_file:
        parser.error("--prepare-cache-only requires --benchmark-file")

    if args.compare:
        run_compare(args)
    elif args.answer_recall:
        run_answer_recall(args)
    elif args.sufficiency_audit:
        run_sufficiency_mode(args)
    elif args.benchmark_file:
        if args.prepare_cache_only:
            run_prepare_cache_only(args)
        else:
            run_benchmark_mode(args)
    elif args.conversation_dir:
        run_replay_mode(args)
    else:
        parser.error(
            "one of --conversation-dir, --benchmark-file, --compare, "
            "--answer-recall, or --sufficiency-audit is required"
        )


if __name__ == "__main__":
    main()
