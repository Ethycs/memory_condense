"""Public CLI facade for locked cumulative 1M validation retrieval."""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections.abc import Sequence
from pathlib import Path

from memory_condense.eval._recall_guarded_cumulative_validation_campaign import (
    ReconstructedValidationShardSet,
    merge_locked_validation_retrievals,
    merged_question_store_receipts,
    reconstruct_and_validate_locked_validation_retrievals,
    run_locked_validation_shard_retrieval,
    validate_merged_validation_retrieval,
)
from memory_condense.eval._recall_guarded_cumulative_validation_shard import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POLICY,
    DEFAULT_QWEN_CHOICE,
    DEFAULT_QWEN_PREFIX,
    DEFAULT_SPLIT,
    LOCKED_100Q_OFFSETS,
    LOCKED_VALIDATION_POLICY_MANIFEST_SHA256,
    VALIDATION_CAMPAIGN_FORMAT,
    VALIDATION_EXECUTION_POLICY_FORMAT,
    VALIDATION_EXTERNAL_RECONSTRUCTION_FORMAT,
    VALIDATION_MERGED_QUESTION_FORMAT,
    VALIDATION_MERGED_RETRIEVAL_FORMAT,
    VALIDATION_POLICY_ATTESTATION_FORMAT,
    VALIDATION_PREFLIGHT_FORMAT,
    VALIDATION_SHARD_QUESTION_FORMAT,
    VALIDATION_SHARD_REFERENCE_FORMAT,
    VALIDATION_SHARD_RETRIEVAL_FORMAT,
    FrozenValidationPolicy,
    ValidationShardPreflight,
    _load_shared_qwen,
    _read_canonical_json,
    load_frozen_validation_policy,
    preflight_locked_validation_shard,
    prepare_validation_source,
    prepare_validation_store,
    shard_output_root,
    validate_validation_shard_retrieval,
)

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or merge the exact ten-shard/100Q LongMemEval validation "
            "cumulative retrieval campaign"
        )
    )
    parser.add_argument(
        "--phase",
        choices=("preflight", "source", "build", "retrieve", "all", "merge"),
        default="preflight",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=None,
        help="defaults to OUTPUT_ROOT/retrieval.json",
    )
    parser.add_argument(
        "--sample-offset",
        type=int,
        choices=LOCKED_100Q_OFFSETS,
        default=0,
    )
    parser.add_argument(
        "--shard-retrieval",
        type=Path,
        action="append",
        default=None,
        help=(
            "ordered shard retrieval path; repeat exactly ten times, or omit "
            "to use OUTPUT_ROOT/shards/offset-NNN/retrieval.json"
        ),
    )
    parser.add_argument(
        "--qwen-prefix-model-dir", type=Path, default=DEFAULT_QWEN_PREFIX
    )
    parser.add_argument(
        "--qwen-choice-model-dir", type=Path, default=DEFAULT_QWEN_CHOICE
    )
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.phase == "merge":
        merge_locked_validation_retrievals(
            dataset_path=args.dataset,
            split_manifest_path=args.split_manifest,
            policy_path=args.policy_manifest,
            output_root=args.output_root,
            output_path=args.merged_output,
            shard_retrieval_paths=args.shard_retrieval,
            device=args.device,
        )
        return 0

    preflight = preflight_locked_validation_shard(
        dataset_path=args.dataset,
        split_manifest_path=args.split_manifest,
        policy_path=args.policy_manifest,
        output_root=args.output_root,
        sample_offset=args.sample_offset,
        qwen_prefix_model_dir=args.qwen_prefix_model_dir,
        qwen_choice_model_dir=args.qwen_choice_model_dir,
        device=args.device,
    )
    if args.phase == "preflight":
        print(
            json.dumps(
                preflight.public_report(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
            flush=True,
        )
        return 0

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.phase == "source":
        binding, _database, receipt, mode = prepare_validation_source(preflight)
        try:
            print(
                f"Current exact-span source: {mode}; receipt "
                f"{receipt['receipt_sha256']}",
                flush=True,
            )
            return 0
        finally:
            binding.embedder.close()

    retrieval_path = preflight.shard_root / "retrieval.json"
    if retrieval_path.exists() and args.phase in {"retrieve", "all"}:
        retrieval, digest = _read_canonical_json(retrieval_path)
        validate_validation_shard_retrieval(retrieval, preflight=preflight)
        print(
            f"Validation shard retrieval already complete: {retrieval_path} "
            f"({digest})",
            flush=True,
        )
        return 0

    prepared, embedder, build_mode, source_receipt, source_mode = (
        prepare_validation_store(preflight)
    )
    try:
        print(
            f"Current exact-span source: {source_mode}; receipt "
            f"{source_receipt['receipt_sha256']}\n"
            f"Combined store: {build_mode}; receipt "
            f"{prepared.receipt.receipt_sha256}",
            flush=True,
        )
        if args.phase == "build":
            return 0
        embedder.close()
        selector, representative_linker = _load_shared_qwen(
            preflight.policy.config,
            preflight.qwen_prefix_model_dir,
            preflight.qwen_choice_model_dir,
        )
        try:
            run_locked_validation_shard_retrieval(
                prepared=prepared,
                preflight=preflight,
                selector=selector,
                representative_linker=representative_linker,
                source_store_receipt=source_receipt,
                source_store_mode=source_mode,
                combined_store_mode=build_mode,
            )
        finally:
            selector.close()
        del representative_linker, selector
        gc.collect()
        return 0
    finally:
        prepared.close()
        close = getattr(embedder, "close", None)
        if callable(close):
            close()


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "LOCKED_VALIDATION_POLICY_MANIFEST_SHA256",
    "VALIDATION_CAMPAIGN_FORMAT",
    "VALIDATION_EXECUTION_POLICY_FORMAT",
    "VALIDATION_EXTERNAL_RECONSTRUCTION_FORMAT",
    "VALIDATION_MERGED_QUESTION_FORMAT",
    "VALIDATION_MERGED_RETRIEVAL_FORMAT",
    "VALIDATION_POLICY_ATTESTATION_FORMAT",
    "VALIDATION_PREFLIGHT_FORMAT",
    "VALIDATION_SHARD_QUESTION_FORMAT",
    "VALIDATION_SHARD_REFERENCE_FORMAT",
    "VALIDATION_SHARD_RETRIEVAL_FORMAT",
    "FrozenValidationPolicy",
    "ValidationShardPreflight",
    "ReconstructedValidationShardSet",
    "load_frozen_validation_policy",
    "main",
    "merge_locked_validation_retrievals",
    "merged_question_store_receipts",
    "preflight_locked_validation_shard",
    "reconstruct_and_validate_locked_validation_retrievals",
    "prepare_validation_source",
    "prepare_validation_store",
    "run_locked_validation_shard_retrieval",
    "shard_output_root",
    "validate_merged_validation_retrieval",
    "validate_validation_shard_retrieval",
]
