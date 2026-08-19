"""CLI entry point for the evaluation pipeline.

Four modes, selected by flags:

    # 1. Self-replay on your own exported conversations (the default)
    pixi run python -m memory_condense.eval --conversation-dir <path>

    # 2. Parameter sweep over chunker/retrieval settings
    pixi run python -m memory_condense.eval --conversation-dir <path> --sweep

    # 3. Public benchmark (LongMemEval / LoCoMo) QA probes
    pixi run python -m memory_condense.eval --benchmark-file longmemeval_oracle.json

    # 4. Offline analysis of two saved runs (no API calls, no cost)
    pixi run python -m memory_condense.eval --compare baseline.json treatment.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from memory_condense._tokenizer import tokenizer_proxy_identity
from memory_condense.eval.analysis import (
    compare_runs,
    load_run,
    print_comparison,
    to_csv,
)
from memory_condense.eval.benchmark import (
    BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE,
    build_judge_prompt,
    ingest_sample,
    print_benchmark_summary,
    run_benchmark,
    save_benchmark_report,
)
from memory_condense.eval.cache_receipts import validated_cache_receipts
from memory_condense.eval.compiled_cache import (
    compiled_store_ingest_fn,
    sample_sha256,
)
from memory_condense.eval.judge import JUDGE_MAX_TOKENS
from memory_condense.eval.locked_split import load_split_manifest, select_locked_split
from memory_condense.eval.recall import print_recall_report, run_recall
from memory_condense.eval.reproducibility import (
    environment_lock_sha256,
    file_sha256,
    implementation_sha256,
    project_root,
)
from memory_condense.eval.report import (
    print_run_summary,
    print_sweep_table,
    save_run_result,
    save_sweep_report,
)
from memory_condense.eval.runner import run_eval
from memory_condense.eval.schemas import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_RESPONDER_MODEL,
    ChunkerConfig,
    EvalConfig,
    RetrievalConfig,
    UsageStats,
)
from memory_condense.eval.sweep import run_sweep
from memory_condense.eval.validation_profile import (
    claimed_validation_profile,
    validate_longmemeval_claim_profile,
)
from memory_condense.loader import load_benchmark, load_directory


class _StoreExplicitValue(argparse.Action):
    """Store an option while retaining whether the caller supplied it.

    Cache preparation must reject responder/judge options even when a caller
    explicitly repeats the normal default.  Comparing parsed values cannot
    distinguish that from an omitted option, so those two model flags use this
    tiny action instead of inspecting process-global ``sys.argv``.
    """

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        del parser, option_string
        setattr(namespace, self.dest, values)
        setattr(namespace, f"_{self.dest}_explicit", True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m memory_condense.eval",
        description="Evaluate memory_condense retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    parser.add_argument(
        "--conversation-dir",
        help="Directory of .txt/.md conversation exports (self-replay mode)",
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Run the full parameter sweep"
    )
    parser.add_argument(
        "--benchmark-file",
        help="LongMemEval/LoCoMo JSON or JSONL file (benchmark QA mode)",
    )
    parser.add_argument(
        "--prepare-cache-only",
        action="store_true",
        help=(
            "Blindly prepare the locked benchmark's compiled and causal "
            "stores; performs no retrieval, answer scoring, or model calls"
        ),
    )
    parser.add_argument(
        "--benchmark-format",
        default="auto",
        choices=["auto", "longmemeval", "locomo"],
        help="Benchmark format (default: auto-detect)",
    )
    parser.add_argument(
        "--benchmark-split-manifest",
        help="Dataset-hash-verified locked split manifest (benchmark modes)",
    )
    parser.add_argument(
        "--benchmark-split",
        help="Named partition from --benchmark-split-manifest",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "TREATMENT"),
        help="Compare two saved eval_results JSON files offline (no API calls)",
    )
    parser.add_argument(
        "--answer-recall",
        metavar="BENCHMARK_FILE",
        help=(
            "Measure whether the gold answer is even reachable from the "
            "assembled context, and whether it survives simulated decay. "
            "Ingests and retrieves locally; makes no API calls"
        ),
    )
    parser.add_argument(
        "--sufficiency-audit",
        metavar="BENCHMARK_FILE",
        help=(
            "Compare retrieved context with the benchmark's gold-source "
            "oracle; deterministic by default, optional semantic judge"
        ),
    )

    # Models — import the defaults so the CLI can never drift from the schema.
    parser.add_argument(
        "--judge-model",
        action=_StoreExplicitValue,
        default=DEFAULT_JUDGE_MODEL,
        help="LLM model for judging",
    )
    parser.add_argument(
        "--responder-model",
        action=_StoreExplicitValue,
        default=DEFAULT_RESPONDER_MODEL,
        help="LLM model for response generation",
    )
    parser.add_argument(
        "--embedding-device",
        choices=("cpu", "cuda"),
        default=None,
        help="Force the local embedding model onto CPU or CUDA",
    )
    parser.add_argument(
        "--compiled-store-cache",
        type=Path,
        help=(
            "Content-addressed cache for reusable per-sample SQLite/HNSW "
            "stores; verified on every hit"
        ),
    )
    parser.add_argument(
        "--causal-store-cache",
        type=Path,
        help=(
            "Content-addressed cache for learned sample-local causal graphs; "
            "write-policy keyed and hash-verified on every hit"
        ),
    )
    parser.add_argument(
        "--policy-manifest",
        type=Path,
        help="Frozen retrieval selection manifest; hash and config are verified",
    )
    parser.add_argument(
        "--local-qwen-model-dir",
        type=Path,
        help="Use a local full Qwen checkpoint as the benchmark responder",
    )
    parser.add_argument(
        "--local-qwen-max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens for the local Qwen responder",
    )
    parser.add_argument("--local-qwen-gpu-memory", default="4GiB")
    parser.add_argument("--local-qwen-cpu-memory", default="24GiB")
    parser.add_argument(
        "--local-qwen-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--qwen-rerank-model-dir",
        type=Path,
        help=(
            "Use a bounded local Qwen prefix as a QK/OV candidate reranker; "
            "requires --source-local-search and a compatible retrieval mode"
        ),
    )
    parser.add_argument(
        "--qwen-rerank-cav-report",
        type=Path,
        default=Path("eval_results/qwen3_prefix_cav_probe.json"),
    )
    parser.add_argument(
        "--qwen-rerank-cav-vectors",
        type=Path,
        default=Path("eval_results/qwen3_prefix_cav_probe.safetensors"),
    )
    parser.add_argument("--qwen-rerank-prefix-layers", type=int, default=2)
    parser.add_argument("--qwen-rerank-attention-layer", type=int, default=1)
    parser.add_argument("--qwen-rerank-cav-layer", type=int, default=5)
    parser.add_argument(
        "--qwen-rerank-use-cav",
        action="store_true",
        help="Also capture the configured source CAV signature (off by default)",
    )
    parser.add_argument("--qwen-rerank-device", default="cuda")
    parser.add_argument("--qwen-rerank-dtype", default="bfloat16")
    parser.add_argument("--qwen-rerank-candidate-pool", type=int, default=64)
    parser.add_argument("--qwen-rerank-slots", type=int, default=6)
    parser.add_argument("--qwen-rerank-group-size", type=int, default=8)
    parser.add_argument("--qwen-rerank-beam-per-group", type=int, default=2)
    parser.add_argument("--qwen-rerank-candidate-tokens", type=int, default=64)
    parser.add_argument("--qwen-rerank-query-tokens", type=int, default=96)
    parser.add_argument("--qwen-rerank-score-weight", type=float, default=0.35)
    parser.add_argument("--qwen-rerank-max-workspace-tokens", type=int, default=1024)
    parser.add_argument(
        "--qwen-feedback",
        action="store_true",
        help=(
            "Attend over first-round evidence and use it for one bounded "
            "second retrieval round instead of direct Qwen reranking"
        ),
    )
    parser.add_argument("--qwen-feedback-candidate-pool", type=int, default=32)
    parser.add_argument("--qwen-feedback-seed-slots", type=int, default=6)
    parser.add_argument("--qwen-feedback-slots", type=int, default=12)
    parser.add_argument("--qwen-feedback-evidence-tokens", type=int, default=48)
    parser.add_argument("--qwen-feedback-query-tokens", type=int, default=384)
    parser.add_argument(
        "--coverage-selector-local-model-dir",
        type=Path,
        help=(
            "Use a small full local model to partition the bounded expansion "
            "set into existing/new/null events before context packing"
        ),
    )
    parser.add_argument(
        "--coverage-selector-qwen-prefix-model-dir",
        type=Path,
        help=(
            "Use only the configured prefix of a full-size Qwen checkpoint for "
            "transient QK/OV event grouping; no LM head or generation"
        ),
    )
    parser.add_argument(
        "--coverage-selector-choice-model-dir",
        type=Path,
        help=(
            "Add a staged local causal checkpoint that scores direct answer "
            "evidence with forced-choice likelihoods; requires the Qwen "
            "prefix selector and never generates text"
        ),
    )
    parser.add_argument("--coverage-selector-choice-model-id", default="")
    parser.add_argument(
        "--coverage-selector-choice-model-revision",
        dest="coverage_selector_choice_revision",
        default="",
    )
    parser.add_argument(
        "--coverage-selector-choice-checkpoint-sha256",
        default="",
    )
    parser.add_argument(
        "--coverage-selector-choice-device",
        default="cuda",
    )
    parser.add_argument(
        "--coverage-selector-choice-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--coverage-selector-choice-batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--coverage-selector-choice-max-candidates",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--coverage-selector-choice-query-tokens",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--coverage-selector-choice-candidate-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--coverage-selector-choice-max-prompt-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--coverage-selector-choice-max-workspace-tokens",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--coverage-selector-cross-encoder-model-dir",
        type=Path,
        help=(
            "Rerank the bounded expansion set with the pinned provider-free "
            "MS MARCO MiniLM cross-encoder; combine with the Qwen prefix flag "
            "to group duplicate events after semantic ranking"
        ),
    )
    parser.add_argument(
        "--coverage-selector-cross-encoder-device",
        default="cuda",
    )
    parser.add_argument(
        "--coverage-selector-cross-encoder-candidate-pool",
        type=int,
        default=None,
        help=(
            "Semantic candidates scored before optional Qwen grouping "
            "(default: at least the complete routed expansion union)"
        ),
    )
    parser.add_argument(
        "--coverage-selector-cross-encoder-semantic-rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Globally reorder the expansion union with MS MARCO; disable for "
            "source-companion selection only (default true)"
        ),
    )
    parser.add_argument(
        "--coverage-selector-cross-encoder-score-only",
        action="store_true",
        help=(
            "Score the full bounded frontier and expose transient logits to "
            "the downstream posterior without changing baseline order"
        ),
    )
    parser.add_argument(
        "--coverage-selector-cross-encoder-batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--coverage-selector-cross-encoder-max-length",
        type=int,
        default=256,
    )
    parser.add_argument("--coverage-selector-candidate-pool", type=int, default=64)
    parser.add_argument("--coverage-selector-candidate-tokens", type=int, default=96)
    parser.add_argument("--coverage-selector-query-tokens", type=int, default=192)
    parser.add_argument(
        "--coverage-selector-max-workspace-tokens",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--coverage-selector-max-new-tokens",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--coverage-selector-null-threshold",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--coverage-selector-uncertainty-entropy",
        type=float,
        default=0.95,
    )
    parser.add_argument("--coverage-selector-prefix-layers", type=int, default=6)
    parser.add_argument("--coverage-selector-attention-layer", type=int, default=5)
    parser.add_argument("--coverage-selector-prefix-device", default="cuda")
    parser.add_argument(
        "--coverage-selector-merge-similarity",
        type=float,
        default=0.985,
    )
    parser.add_argument(
        "--coverage-selector-same-source-merge-similarity",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--coverage-selector-strict",
        action="store_true",
        help="Raise on malformed classifier output instead of recall-safe fallback",
    )
    parser.add_argument(
        "--allow-selected-scope-fixed-k-closure",
        action="store_true",
        help=(
            "Frozen policy exception: allow an exhaustive typed FIXED-K scan "
            "of approximately selected partitions to close the prompt tail; "
            "global recall remains explicitly unguaranteed"
        ),
    )
    parser.add_argument("--coverage-selector-gpu-memory", default="4GiB")
    parser.add_argument("--coverage-selector-cpu-memory", default="24GiB")
    parser.add_argument(
        "--coverage-selector-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )

    parser.add_argument(
        "--results-dir", default="./eval_results", help="Output directory"
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Limit number of conversations evaluated (self-replay mode)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of benchmark samples evaluated",
    )
    parser.add_argument(
        "--stress-context-tokens",
        type=int,
        default=None,
        help=(
            "Combine complete benchmark histories into one memory containing "
            "at least this many content tokens before answer-recall"
        ),
    )
    parser.add_argument(
        "--stress-questions",
        type=int,
        default=10,
        help="Questions issued against a combined context-stress memory",
    )
    parser.add_argument(
        "--stress-question-offset",
        type=int,
        default=0,
        help="Skip this many locked questions while retaining the full stress memory",
    )
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help=(
            "Skip this many samples after locked-split selection; enables "
            "non-overlapping resumable/parallel benchmark shards"
        ),
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=4,
        help="Number of recent turns to include in context",
    )
    parser.add_argument(
        "--use-judge",
        action="store_true",
        help="Also grade benchmark answers with an LLM judge (doubles API cost)",
    )
    parser.add_argument(
        "--max-provider-calls",
        type=int,
        default=0,
        help=(
            "Required logical-call ceiling for remote benchmark models; "
            "default 0 refuses paid calls"
        ),
    )
    parser.add_argument(
        "--provider-retries",
        type=int,
        default=0,
        help="Automatic retries per remote provider call (default 0)",
    )
    parser.add_argument(
        "--accuracy-target",
        type=float,
        default=0.95,
        help="Judge-accuracy target for long-chat runs (default 0.95)",
    )
    parser.add_argument(
        "--min-target-questions",
        type=int,
        default=100,
        help="Minimum judged questions required to pass the accuracy target",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=8000,
        help=(
            "Hard local responder prompt-token-proxy cap per question "
            "(cl100k_base plus explicit chat framing; default 8000)"
        ),
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        help="Write per-turn results as CSV (with --compare, writes the treatment run)",
    )

    # Retrieval / chunker params
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=250)
    parser.add_argument("--k", type=int, default=10, help="Chunks retrieved (0 = no-memory baseline)")
    parser.add_argument("--ef-search", type=int, default=50)
    parser.add_argument(
        "--mode",
        choices=[
            "dense",
            "hybrid",
            "memory",
            "span",
            "source",
            "anchored_source",
            "hybrid_source",
            "hybrid_graph",
            "hybrid_neighbor",
            "causal_consolidation",
            "causal_graph",
        ],
        default="dense",
        help=(
            "What the responder is given: dense chunks (default), "
            "hybrid BM25+dense chunks, the packed memory context "
            "(memory-item header + budgeted expansions), or pooled spans of "
            "contiguous chunks (best on short-turn dialogue), or complete "
            "provenance sources/sessions, hybrid-anchored source expansion, "
            "bounded reranking inside hybrid-activated sources, their "
            "transition/source graph union, "
            "bounded source-local expansion around hybrid anchors, or a "
            "sample-local causal consolidation graph, or its bounded "
            "transition/source-union candidate front end"
        ),
    )
    parser.add_argument(
        "--span-levels",
        default="110,220",
        help="Comma-separated token targets per span level for --mode span (default 110,220)",
    )
    parser.add_argument(
        "--k-per-level",
        type=int,
        default=2,
        help="Spans taken from each level in --mode span (default 2)",
    )
    parser.add_argument(
        "--k-sources",
        type=int,
        default=4,
        help="Complete sources/sessions retrieved in --mode source (default 4)",
    )
    parser.add_argument(
        "--source-slots",
        type=int,
        default=24,
        help="Extra chunks from hybrid-activated sources (default 24)",
    )
    parser.add_argument(
        "--source-candidate-pool",
        type=int,
        default=200,
        help="Candidate pool for --mode hybrid_source (default 200)",
    )
    parser.add_argument(
        "--source-activation-k",
        type=int,
        default=None,
        help="Pool prefix allowed to activate source links (default: --k)",
    )
    parser.add_argument(
        "--query-facet-retrieval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reserve bounded retrieval slots for each explicit list facet "
            "after a question colon (default false)"
        ),
    )
    parser.add_argument("--query-facet-slots", type=int, default=6)
    parser.add_argument("--query-facet-max", type=int, default=4)
    parser.add_argument(
        "--role-aware-retrieval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Prefer user turns over assistant suggestions for first-person "
            "questions (default false)"
        ),
    )
    parser.add_argument("--role-user-weight", type=float, default=1.25)
    parser.add_argument("--role-assistant-weight", type=float, default=0.75)
    parser.add_argument("--role-system-weight", type=float, default=0.50)
    parser.add_argument(
        "--multi-fact-source-diversity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Round-robin candidates by source for order/set questions "
            "(default false)"
        ),
    )
    parser.add_argument(
        "--source-tfisf-activation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add bounded live TF-ISF source activation (default false)",
    )
    parser.add_argument("--source-tfisf-slots", type=int, default=8)
    parser.add_argument(
        "--source-hsc-activation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Expand source seeds through bounded pairwise contraction",
    )
    parser.add_argument("--source-hsc-slots", type=int, default=8)
    parser.add_argument("--source-hsc-hops", type=int, default=2)
    parser.add_argument("--source-hsc-chunk-slots", type=int, default=8)
    parser.add_argument(
        "--source-local-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Search inside activated sources instead of filtering the global "
            "candidate pool (default false for historical-arm reproducibility)"
        ),
    )
    parser.add_argument(
        "--source-partition-routing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Route through hierarchical partition::source IDs before chunk "
            "competition (default false)"
        ),
    )
    parser.add_argument("--source-partition-slots", type=int, default=3)
    parser.add_argument("--source-partition-separator", default="::")
    parser.add_argument(
        "--neighbor-radius",
        type=int,
        default=1,
        help="Source-local chunk shells in --mode hybrid_neighbor (default 1)",
    )
    parser.add_argument(
        "--neighbor-slots",
        type=int,
        default=5,
        help="Hard extra-chunk budget in --mode hybrid_neighbor (default 5)",
    )
    parser.add_argument(
        "--neighbor-replacement-slots",
        type=int,
        default=0,
        help=(
            "Replace this many weakest anchors with transition candidates "
            "in --mode hybrid_neighbor (default 0)"
        ),
    )
    parser.add_argument(
        "--neighbor-direction",
        choices=("both", "previous", "next"),
        default="both",
        help="Transition direction for graph retrieval (default both)",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Deprecated alias for --mode hybrid",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Dense weight when blending (1.0 = pure dense)",
    )
    parser.add_argument(
        "--k-memories",
        type=int,
        default=8,
        help="Memory items requested for the header in --mode memory",
    )
    parser.add_argument("--consolidation-chunk-slots", type=int, default=3)
    parser.add_argument("--consolidation-hops", type=int, default=2)
    parser.add_argument("--consolidation-candidates", type=int, default=128)
    parser.add_argument("--consolidation-diffusion-width", type=int, default=32)
    parser.add_argument("--consolidation-min-count", type=int, default=2)
    parser.add_argument("--consolidation-expansion-tokens", type=int, default=1600)
    parser.add_argument(
        "--consolidation-training-expansion-tokens",
        type=int,
        default=1600,
    )
    parser.add_argument("--consolidation-training-k", type=int, default=10)
    parser.add_argument("--consolidation-max-event-nodes", type=int, default=9)
    parser.add_argument("--consolidation-new-event-nodes", type=int, default=5)
    parser.add_argument(
        "--consolidation-max-training-prompt-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--consolidation-budget-aware-packing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--consolidation-source-diverse-packing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--consolidation-query-aware-sentence-packing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--consolidation-max-sentences-per-expansion",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--consolidation-information-gain-packing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--consolidation-min-information-gain-per-token",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--consolidation-source-metadata-packing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Bind source/session timestamps to their selected excerpts instead "
            "of packing anonymous timestamp-only chunks"
        ),
    )

    return parser


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    # --hybrid predates --mode and is kept so the commands in
    # `docs/02 - Implementation/01` keep working.
    if args.qwen_feedback and not args.qwen_rerank_model_dir:
        raise ValueError("--qwen-feedback requires --qwen-rerank-model-dir")
    choice_dir = args.coverage_selector_choice_model_dir
    cross_encoder_dir = args.coverage_selector_cross_encoder_model_dir
    qwen_prefix_dir = args.coverage_selector_qwen_prefix_model_dir
    local_ini_dir = args.coverage_selector_local_model_dir
    if choice_dir and not qwen_prefix_dir:
        raise ValueError(
            "--coverage-selector-choice-model-dir requires "
            "--coverage-selector-qwen-prefix-model-dir"
        )
    if local_ini_dir and (qwen_prefix_dir or cross_encoder_dir or choice_dir):
        raise ValueError(
            "choose either the Qwen prefix/MS MARCO coverage path or the local "
            "INI classifier, not both"
        )
    if choice_dir and cross_encoder_dir:
        raise ValueError(
            "the forced-choice and MS MARCO score providers are separate "
            "coverage arms"
        )

    choice_model_id = ""
    choice_revision = ""
    choice_checkpoint_sha256 = ""
    if choice_dir:
        from memory_condense.causal_choice_scorer import (
            QWEN_CHOICE_MODEL_ID,
            QWEN_CHOICE_MODEL_REVISION,
            QWEN_CHOICE_WEIGHTS_SHA256,
            SMOLLM_CHOICE_MODEL_ID,
            SMOLLM_CHOICE_MODEL_REVISION,
            SMOLLM_CHOICE_WEIGHTS_SHA256,
        )

        explicit_identity = (
            args.coverage_selector_choice_model_id,
            args.coverage_selector_choice_revision,
            args.coverage_selector_choice_checkpoint_sha256,
        )
        if any(explicit_identity):
            if not all(explicit_identity):
                raise ValueError(
                    "explicit choice identity requires model id, revision, "
                    "and checkpoint SHA-256"
                )
            (
                choice_model_id,
                choice_revision,
                choice_checkpoint_sha256,
            ) = explicit_identity
        elif choice_dir.name.casefold() == "qwen3-0.6b".casefold():
            choice_model_id = QWEN_CHOICE_MODEL_ID
            choice_revision = QWEN_CHOICE_MODEL_REVISION
            choice_checkpoint_sha256 = QWEN_CHOICE_WEIGHTS_SHA256
        elif choice_dir.name.casefold() == "smollm2-360m-instruct".casefold():
            choice_model_id = SMOLLM_CHOICE_MODEL_ID
            choice_revision = SMOLLM_CHOICE_MODEL_REVISION
            choice_checkpoint_sha256 = SMOLLM_CHOICE_WEIGHTS_SHA256
        else:
            raise ValueError(
                "unknown choice checkpoint directory; provide exact "
                "--coverage-selector-choice-model-id, "
                "--coverage-selector-choice-model-revision, and "
                "--coverage-selector-choice-checkpoint-sha256"
            )

    coverage_selection = bool(
        cross_encoder_dir or qwen_prefix_dir or local_ini_dir or choice_dir
    )
    if choice_dir and qwen_prefix_dir:
        coverage_backend = "qwen_prefix_choice"
        coverage_model = f"{qwen_prefix_dir.name}+{choice_dir.name}"
    elif cross_encoder_dir and qwen_prefix_dir:
        coverage_backend = "cross_encoder_qwen_prefix"
        coverage_model = f"{cross_encoder_dir.name}+{qwen_prefix_dir.name}"
    elif cross_encoder_dir:
        coverage_backend = "cross_encoder"
        coverage_model = cross_encoder_dir.name
    elif qwen_prefix_dir:
        coverage_backend = "qwen_prefix"
        coverage_model = qwen_prefix_dir.name
    else:
        coverage_backend = "local_ini"
        coverage_model = local_ini_dir.name if local_ini_dir else ""
    if cross_encoder_dir:
        from memory_condense.cross_encoder_selector import (
            MS_MARCO_MODEL_ID,
            MS_MARCO_MODEL_REVISION,
            MS_MARCO_WEIGHTS_SHA256,
        )
    else:
        MS_MARCO_MODEL_ID = ""
        MS_MARCO_MODEL_REVISION = ""
        MS_MARCO_WEIGHTS_SHA256 = ""
    prefix_model_id = ""
    prefix_revision = ""
    prefix_checkpoint_sha256 = ""
    prefix_device = ""
    prefix_dtype = ""
    if qwen_prefix_dir:
        import torch

        from memory_condense.eval.local_qwen import resolve_local_qwen_dtype
        from memory_condense.qwen_prefix import (
            DEFAULT_MODEL_ID,
            DEFAULT_MODEL_REVISION,
            expected_prefix_checkpoint_sha256,
        )

        prefix_model_id = DEFAULT_MODEL_ID
        prefix_revision = DEFAULT_MODEL_REVISION
        prefix_checkpoint_sha256 = expected_prefix_checkpoint_sha256(
            args.coverage_selector_prefix_layers
        )
        prefix_device = str(args.coverage_selector_prefix_device)
        _prefix_torch_dtype, prefix_dtype = resolve_local_qwen_dtype(
            torch,
            args.coverage_selector_dtype,
            device=prefix_device,
        )
    mode = "hybrid" if args.hybrid and args.mode == "dense" else args.mode
    routed_expansion_union = args.k + args.consolidation_chunk_slots
    if mode == "causal_graph":
        routed_expansion_union += args.neighbor_slots + args.source_slots
    cross_encoder_candidate_pool = (
        args.coverage_selector_cross_encoder_candidate_pool
        if args.coverage_selector_cross_encoder_candidate_pool is not None
        else max(
            128,
            routed_expansion_union,
            args.coverage_selector_candidate_pool,
        )
    )
    cross_encoder_semantic_rerank = bool(
        args.coverage_selector_cross_encoder_semantic_rerank
        and not args.coverage_selector_cross_encoder_score_only
    )
    return EvalConfig(
        chunker=ChunkerConfig(min_tokens=args.min_tokens, max_tokens=args.max_tokens),
        retrieval=RetrievalConfig(
            k=args.k,
            ef_search=args.ef_search,
            mode=mode,
            hybrid=args.hybrid,
            alpha=args.alpha,
            k_memories=args.k_memories,
            span_levels=tuple(
                int(x) for x in str(args.span_levels).split(",") if x.strip()
            ),
            k_per_level=args.k_per_level,
            k_sources=args.k_sources,
            source_slots=args.source_slots,
            source_candidate_pool=args.source_candidate_pool,
            source_activation_k=args.source_activation_k,
            query_facet_retrieval=args.query_facet_retrieval,
            query_facet_slots=args.query_facet_slots,
            query_facet_max=args.query_facet_max,
            role_aware_retrieval=args.role_aware_retrieval,
            role_user_weight=args.role_user_weight,
            role_assistant_weight=args.role_assistant_weight,
            role_system_weight=args.role_system_weight,
            multi_fact_source_diversity=args.multi_fact_source_diversity,
            source_tfisf_activation=args.source_tfisf_activation,
            source_tfisf_slots=args.source_tfisf_slots,
            source_hsc_activation=args.source_hsc_activation,
            source_hsc_slots=args.source_hsc_slots,
            source_hsc_hops=args.source_hsc_hops,
            source_hsc_chunk_slots=args.source_hsc_chunk_slots,
            source_local_search=args.source_local_search,
            source_partition_routing=args.source_partition_routing,
            source_partition_slots=args.source_partition_slots,
            source_partition_separator=args.source_partition_separator,
            qwen_rerank=(
                bool(args.qwen_rerank_model_dir) and not args.qwen_feedback
            ),
            qwen_rerank_candidate_pool=args.qwen_rerank_candidate_pool,
            qwen_rerank_slots=args.qwen_rerank_slots,
            qwen_rerank_group_size=args.qwen_rerank_group_size,
            qwen_rerank_beam_per_group=args.qwen_rerank_beam_per_group,
            qwen_rerank_candidate_tokens=args.qwen_rerank_candidate_tokens,
            qwen_rerank_query_tokens=args.qwen_rerank_query_tokens,
            qwen_rerank_score_weight=args.qwen_rerank_score_weight,
            qwen_rerank_model=(
                args.qwen_rerank_model_dir.name
                if args.qwen_rerank_model_dir
                else ""
            ),
            qwen_rerank_prefix_layers=args.qwen_rerank_prefix_layers,
            qwen_rerank_attention_layer=args.qwen_rerank_attention_layer,
            qwen_rerank_use_cav=args.qwen_rerank_use_cav,
            qwen_rerank_cav_layer=args.qwen_rerank_cav_layer,
            qwen_rerank_max_workspace_tokens=(
                args.qwen_rerank_max_workspace_tokens
            ),
            qwen_feedback=args.qwen_feedback,
            qwen_feedback_candidate_pool=args.qwen_feedback_candidate_pool,
            qwen_feedback_seed_slots=args.qwen_feedback_seed_slots,
            qwen_feedback_slots=args.qwen_feedback_slots,
            qwen_feedback_evidence_tokens=args.qwen_feedback_evidence_tokens,
            qwen_feedback_query_tokens=args.qwen_feedback_query_tokens,
            coverage_selection=coverage_selection,
            coverage_selector_backend=coverage_backend,
            coverage_selector_model=coverage_model,
            coverage_selector_dtype=args.coverage_selector_dtype,
            coverage_selector_prefix_model_id=prefix_model_id,
            coverage_selector_prefix_revision=prefix_revision,
            coverage_selector_prefix_checkpoint_sha256=(
                prefix_checkpoint_sha256
            ),
            coverage_selector_prefix_device=prefix_device,
            coverage_selector_prefix_dtype=prefix_dtype,
            coverage_selector_candidate_pool=(
                args.coverage_selector_candidate_pool
            ),
            coverage_selector_candidate_tokens=(
                args.coverage_selector_candidate_tokens
            ),
            coverage_selector_query_tokens=args.coverage_selector_query_tokens,
            coverage_selector_max_workspace_tokens=(
                args.coverage_selector_max_workspace_tokens
            ),
            coverage_selector_max_new_tokens=(
                args.coverage_selector_max_new_tokens
            ),
            coverage_selector_cross_encoder_model_id=MS_MARCO_MODEL_ID,
            coverage_selector_cross_encoder_revision=MS_MARCO_MODEL_REVISION,
            coverage_selector_cross_encoder_checkpoint_sha256=(
                MS_MARCO_WEIGHTS_SHA256
            ),
            coverage_selector_cross_encoder_device=(
                args.coverage_selector_cross_encoder_device
            ),
            coverage_selector_cross_encoder_candidate_pool=(
                cross_encoder_candidate_pool
            ),
            coverage_selector_cross_encoder_semantic_rerank=(
                cross_encoder_semantic_rerank
            ),
            coverage_selector_cross_encoder_score_only=(
                args.coverage_selector_cross_encoder_score_only
            ),
            coverage_selector_cross_encoder_batch_size=(
                args.coverage_selector_cross_encoder_batch_size
            ),
            coverage_selector_cross_encoder_max_length=(
                args.coverage_selector_cross_encoder_max_length
            ),
            coverage_selector_choice_model_id=choice_model_id,
            coverage_selector_choice_revision=choice_revision,
            coverage_selector_choice_checkpoint_sha256=(
                choice_checkpoint_sha256
            ),
            coverage_selector_choice_device=(
                args.coverage_selector_choice_device
            ),
            coverage_selector_choice_dtype=args.coverage_selector_choice_dtype,
            coverage_selector_choice_batch_size=(
                args.coverage_selector_choice_batch_size
            ),
            coverage_selector_choice_max_candidates=(
                args.coverage_selector_choice_max_candidates
            ),
            coverage_selector_choice_query_tokens=(
                args.coverage_selector_choice_query_tokens
            ),
            coverage_selector_choice_candidate_tokens=(
                args.coverage_selector_choice_candidate_tokens
            ),
            coverage_selector_choice_max_prompt_tokens=(
                args.coverage_selector_choice_max_prompt_tokens
            ),
            coverage_selector_choice_max_workspace_tokens=(
                args.coverage_selector_choice_max_workspace_tokens
            ),
            coverage_selector_null_threshold=(
                args.coverage_selector_null_threshold
            ),
            coverage_selector_uncertainty_entropy=(
                args.coverage_selector_uncertainty_entropy
            ),
            coverage_selector_prefix_layers=args.coverage_selector_prefix_layers,
            coverage_selector_attention_layer=(
                args.coverage_selector_attention_layer
            ),
            coverage_selector_merge_similarity=(
                args.coverage_selector_merge_similarity
            ),
            coverage_selector_same_source_merge_similarity=(
                args.coverage_selector_same_source_merge_similarity
            ),
            allow_selected_scope_fixed_k_closure=(
                args.allow_selected_scope_fixed_k_closure
            ),
            coverage_selector_strict=args.coverage_selector_strict,
            neighbor_radius=args.neighbor_radius,
            neighbor_slots=args.neighbor_slots,
            neighbor_replacement_slots=args.neighbor_replacement_slots,
            neighbor_direction=args.neighbor_direction,
            consolidation_chunk_slots=args.consolidation_chunk_slots,
            consolidation_hops=args.consolidation_hops,
            consolidation_candidates=args.consolidation_candidates,
            consolidation_diffusion_width=args.consolidation_diffusion_width,
            consolidation_min_count=args.consolidation_min_count,
            consolidation_expansion_tokens=args.consolidation_expansion_tokens,
            consolidation_training_expansion_tokens=(
                args.consolidation_training_expansion_tokens
            ),
            consolidation_budget_aware_packing=(
                args.consolidation_budget_aware_packing
            ),
            consolidation_source_diverse_packing=(
                args.consolidation_source_diverse_packing
            ),
            consolidation_query_aware_sentence_packing=(
                args.consolidation_query_aware_sentence_packing
            ),
            consolidation_max_sentences_per_expansion=(
                args.consolidation_max_sentences_per_expansion
            ),
            consolidation_information_gain_packing=(
                args.consolidation_information_gain_packing
            ),
            consolidation_min_information_gain_per_token=(
                args.consolidation_min_information_gain_per_token
            ),
            consolidation_source_metadata_packing=(
                args.consolidation_source_metadata_packing
            ),
            consolidation_training_k=args.consolidation_training_k,
            consolidation_max_event_nodes=args.consolidation_max_event_nodes,
            consolidation_new_event_nodes=args.consolidation_new_event_nodes,
            consolidation_max_training_prompt_tokens=(
                args.consolidation_max_training_prompt_tokens
            ),
        ),
        judge_model=args.judge_model,
        responder_model=args.responder_model,
        embedding_device=args.embedding_device,
        conversation_dir=(
            args.conversation_dir
            or args.benchmark_file
            or args.answer_recall
            or args.sufficiency_audit
            or ""
        ),
        results_dir=args.results_dir,
        max_conversations=args.max_conversations,
        recent_window=args.recent_window,
        accuracy_target=args.accuracy_target,
        min_target_questions=args.min_target_questions,
        max_prompt_tokens=args.max_prompt_tokens,
    )


def _content(response) -> str:
    """The assistant text, or "" if the provider returned none.

    A refusal, a content filter, or a `max_tokens` stop before any visible text
    all yield ``content=None``. Reaching ``.strip()`` on that raises
    ``AttributeError`` deep in a paid run, after every preceding call has
    already been billed.
    """
    try:
        return (response.choices[0].message.content or "").strip()
    except (AttributeError, IndexError, TypeError):
        return ""


_BINARY_JUDGE_VERDICT = re.compile(
    r"^\s*(CORRECT|INCORRECT)\b(?P<remainder>.*)$",
    re.IGNORECASE | re.DOTALL,
)


def _parse_binary_judge_verdict(text: str) -> bool:
    """Parse one unambiguous judge label and reject provider/protocol noise."""

    match = _BINARY_JUDGE_VERDICT.match(text or "")
    if match is None:
        raise RuntimeError("judge returned an empty or malformed verdict")
    remainder = match.group("remainder").lstrip(" \t\r\n,.:;-—")
    if remainder.casefold().startswith("or ") or remainder.startswith("/"):
        raise RuntimeError("judge returned an ambiguous verdict")
    return match.group(1).casefold() == "correct"


def _make_central_dev_client(model: str):
    """Return the trusted OpenAI-compatible client for central-dev routes.

    LiteLLM otherwise constructs its own certifi-backed transport, which does
    not see the internal Caddy CA installed in the Windows trust store.  Keep
    this in one place so answer, judge, and sufficiency calls cannot silently
    diverge onto different transports.
    """
    api_base = os.environ.get("OPENAI_API_BASE", "") or os.environ.get(
        "LITELLM_API_BASE", ""
    )
    api_key = os.environ.get("OPENAI_API_KEY", "") or os.environ.get(
        "LITELLM_KEY", ""
    )
    # The codex_sdk namespace is served by the central-dev v1 gateway. A
    # checked-in command should work with the gateway-native LITELLM_KEY name;
    # requiring callers to duplicate it into OPENAI_API_KEY made normal pixi
    # runs fall through to LiteLLM's unconfigured generic OpenAI transport.
    if not api_base and model.startswith("openai/codex_sdk/") and api_key:
        api_base = "https://central-dev.zt:4000/v1"
    if not model.startswith("openai/") or "central-dev.zt" not in api_base:
        return None
    if not api_key:
        return None

    import ssl

    import httpx
    import truststore
    from openai import OpenAI

    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return OpenAI(
        api_key=api_key,
        base_url=api_base,
        http_client=httpx.Client(verify=ssl_context),
        # ``num_retries`` is controlled and budgeted by this harness. Letting
        # the nested SDK retry too would exceed --max-provider-calls silently.
        max_retries=0,
    )


def _make_answer_fn(model: str, *, retries: int = 0):
    """Answer a benchmark question. Short, deterministic answers — F1/EM depend on it."""
    import litellm

    central_dev_client = _make_central_dev_client(model)

    def answer_fn(
        messages: list[dict[str, str]],
    ) -> tuple[str, UsageStats]:
        started = time.perf_counter()
        request = {
            "model": model,
            "messages": messages,
            "max_tokens": BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE,
            "num_retries": retries,
        }
        # Codex GPT-5 routes reject non-default temperature values. Omitting
        # the field keeps the central-dev codex_sdk gateway compatible while
        # preserving deterministic temperature=0 for historical model routes.
        if "codex_sdk/" not in model:
            request["temperature"] = 0.0
        if central_dev_client is not None:
            request["client"] = central_dev_client
        response = litellm.completion(**request)
        content = _content(response)
        if not content:
            raise RuntimeError("responder returned no answer text")
        return content, UsageStats.from_litellm(
            response,
            time.perf_counter() - started,
        )

    return answer_fn


def _make_judge_fn(model: str, *, retries: int = 0):
    """Semantic-equivalence grading, for answers that F1 scores unfairly.

    ``max_tokens`` is JUDGE_MAX_TOKENS for the reason spelled out in
    ``judge.py``: the default judge is Sonnet 5, which runs adaptive thinking,
    and ``max_tokens`` caps thinking + visible text together. A tight 256 spends
    the whole budget on thinking and returns an empty verdict — which this path
    then scored as INCORRECT for every answer. The replay judge got this fix;
    this one did not, so it is deliberately expressed as the same constant.
    """
    import litellm

    central_dev_client = _make_central_dev_client(model)

    def judge_fn(
        question: str,
        gold: str,
        prediction: str,
    ) -> tuple[bool, str, UsageStats]:
        started = time.perf_counter()
        request = {
            "model": model,
            "messages": build_judge_prompt(question, gold, prediction),
            "max_tokens": JUDGE_MAX_TOKENS,
            "num_retries": retries,
        }
        if central_dev_client is not None:
            request["client"] = central_dev_client
        response = litellm.completion(**request)
        text = _content(response)
        return (
            _parse_binary_judge_verdict(text),
            text,
            UsageStats.from_litellm(response, time.perf_counter() - started),
        )

    return judge_fn


def _make_sufficiency_fn(model: str, *, retries: int = 0):
    """Judge whether excerpts can derive the gold answer, not an answer string."""

    import litellm

    from memory_condense.eval.sufficiency import build_sufficiency_prompt

    central_dev_client = _make_central_dev_client(model)

    def sufficiency_fn(
        question: str,
        gold: str,
        context: list[str],
    ) -> tuple[bool, str, UsageStats]:
        started = time.perf_counter()
        request = {
            "model": model,
            "messages": build_sufficiency_prompt(question, gold, context),
            "max_tokens": JUDGE_MAX_TOKENS,
            "num_retries": retries,
        }
        if central_dev_client is not None:
            request["client"] = central_dev_client
        response = litellm.completion(**request)
        verdict = _content(response)
        return (
            verdict.upper().startswith("SUFFICIENT"),
            verdict,
            UsageStats.from_litellm(response, time.perf_counter() - started),
        )

    return sufficiency_fn


def _apply_locked_split(
    args: argparse.Namespace,
    samples,
    *,
    verbose: bool = True,
):
    manifest_path = args.benchmark_split_manifest
    split = args.benchmark_split
    if bool(manifest_path) != bool(split):
        raise ValueError(
            "--benchmark-split-manifest and --benchmark-split must be used together"
        )
    if not manifest_path:
        return samples
    dataset_path = (
        args.answer_recall or args.sufficiency_audit or args.benchmark_file
    )
    manifest = load_split_manifest(manifest_path)
    selected = select_locked_split(
        samples,
        dataset_path=dataset_path,
        manifest=manifest,
        split=split,
    )
    if verbose:
        print(
            f"Locked split {split!r}: {len(selected)} / {len(samples)} samples "
            f"(dataset sha256 {manifest.dataset_sha256[:12]}...)"
        )
    return selected


def _apply_sample_offset(
    args: argparse.Namespace,
    samples,
    *,
    verbose: bool = True,
):
    offset = int(args.sample_offset)
    if offset < 0:
        raise ValueError("--sample-offset must be non-negative")
    if offset >= len(samples) and offset:
        raise ValueError(
            f"--sample-offset {offset} is outside the {len(samples)} samples"
        )
    if offset and verbose:
        print(f"Sample shard starts at locked-split offset {offset}")
    if offset:
        return samples[offset:]
    return samples


def _planned_provider_calls(
    samples,
    *,
    max_samples: int | None,
    local_answerer: bool,
    use_judge: bool,
    provider_retries: int = 0,
) -> int:
    if provider_retries < 0:
        raise ValueError("provider_retries must be non-negative")
    selected = samples[:max_samples] if max_samples is not None else samples
    questions = sum(len(sample.questions) for sample in selected)
    logical_calls = (0 if local_answerer else questions) + (
        questions if use_judge else 0
    )
    return logical_calls * (provider_retries + 1)


def _benchmark_evaluation_identity(
    args: argparse.Namespace,
    config: EvalConfig,
) -> dict[str, object]:
    """Execution controls that a frozen validation policy must precommit.

    Retrieval and chunking already live in the policy's ``retrieval`` object.
    These values cover the answer/judge protocol and the exact resumable stress
    shard, none of which are represented by :class:`EvalConfig.retrieval`.
    """

    return {
        "responder_model": config.responder_model,
        "judge_model": config.judge_model,
        "embedding_device": config.embedding_device,
        "benchmark_format": str(args.benchmark_format),
        "use_judge": bool(args.use_judge),
        "provider_retries": int(args.provider_retries),
        "max_provider_calls": int(args.max_provider_calls),
        "max_prompt_tokens": config.max_prompt_tokens,
        "prompt_cap_semantics": (
            "local_prompt_token_proxy_with_provider_usage_postcheck_v1"
        ),
        "prompt_token_proxy_identity": tokenizer_proxy_identity(),
        "responder_output_token_reserve": (
            int(args.local_qwen_max_new_tokens)
            if args.local_qwen_model_dir
            else BENCHMARK_RESPONDER_OUTPUT_TOKEN_RESERVE
        ),
        "recent_window": config.recent_window,
        "accuracy_target": config.accuracy_target,
        "min_target_questions": config.min_target_questions,
        "stress_context_tokens": getattr(args, "stress_context_tokens", None),
        "stress_questions": int(getattr(args, "stress_questions", 10)),
        "stress_question_offset": int(
            getattr(args, "stress_question_offset", 0)
        ),
        "max_samples": args.max_samples,
        "sample_offset": int(args.sample_offset),
    }


def _coverage_prefix_policy_identity(config: EvalConfig) -> dict[str, str]:
    """Exact runtime/checkpoint identity for prefix-backed coverage arms."""

    if config.retrieval.coverage_selector_backend not in {
        "qwen_prefix",
        "qwen_prefix_choice",
        "cross_encoder_qwen_prefix",
    }:
        return {}
    return {
        "coverage_selector_prefix_model_id": (
            config.retrieval.coverage_selector_prefix_model_id
        ),
        "coverage_selector_prefix_revision": (
            config.retrieval.coverage_selector_prefix_revision
        ),
        "coverage_selector_prefix_checkpoint_sha256": (
            config.retrieval.coverage_selector_prefix_checkpoint_sha256
        ),
        "coverage_selector_prefix_device": (
            config.retrieval.coverage_selector_prefix_device
        ),
        "coverage_selector_prefix_dtype": (
            config.retrieval.coverage_selector_prefix_dtype
        ),
    }


def _policy_retrieval_identity(config: EvalConfig) -> dict[str, object]:
    """Return the exact conditional retrieval identity enforced by policy files.

    Keeping construction separate from file verification lets a selected
    development command generate its manifest through the same code path that
    later rejects drift. Disabled arms stay out of the identity; every active
    arm includes both its explicit values and runtime defaults.
    """

    expected = {
        "mode": config.retrieval.mode,
        "k": config.retrieval.k,
        "ef_search": config.retrieval.ef_search,
        "alpha": config.retrieval.alpha,
        "candidates": config.retrieval.candidates,
        "neighbor_radius": config.retrieval.neighbor_radius,
        "neighbor_slots": config.retrieval.neighbor_slots,
        "neighbor_replacement_slots": (
            config.retrieval.neighbor_replacement_slots
        ),
        "max_prompt_tokens": config.max_prompt_tokens,
        "chunker_min_tokens": config.chunker.min_tokens,
        "chunker_max_tokens": config.chunker.max_tokens,
    }
    if config.retrieval.mode in {
        "hybrid_source",
        "hybrid_graph",
        "causal_graph",
    }:
        expected.update(
            {
                "source_slots": config.retrieval.source_slots,
                "source_activation_k": (
                    config.retrieval.source_activation_k or config.retrieval.k
                ),
                "source_candidate_pool": config.retrieval.source_candidate_pool,
            }
        )
        if config.retrieval.source_local_search:
            expected["source_local_search"] = True
        if config.retrieval.source_tfisf_activation:
            expected["source_tfisf_activation"] = True
            expected["source_tfisf_slots"] = config.retrieval.source_tfisf_slots
        if config.retrieval.source_hsc_activation:
            expected.update(
                {
                    "source_hsc_activation": True,
                    "source_hsc_slots": config.retrieval.source_hsc_slots,
                    "source_hsc_hops": config.retrieval.source_hsc_hops,
                    "source_hsc_chunk_slots": (
                        config.retrieval.source_hsc_chunk_slots
                    ),
                }
            )
        if config.retrieval.source_partition_routing:
            expected.update(
                {
                    "source_partition_routing": True,
                    "source_partition_slots": (
                        config.retrieval.source_partition_slots
                    ),
                    "source_partition_separator": (
                        config.retrieval.source_partition_separator
                    ),
                }
            )
        if config.retrieval.qwen_rerank:
            expected.update(
                {
                    "qwen_rerank": True,
                    "qwen_rerank_candidate_pool": (
                        config.retrieval.qwen_rerank_candidate_pool
                    ),
                    "qwen_rerank_slots": config.retrieval.qwen_rerank_slots,
                    "qwen_rerank_group_size": (
                        config.retrieval.qwen_rerank_group_size
                    ),
                    "qwen_rerank_beam_per_group": (
                        config.retrieval.qwen_rerank_beam_per_group
                    ),
                    "qwen_rerank_candidate_tokens": (
                        config.retrieval.qwen_rerank_candidate_tokens
                    ),
                    "qwen_rerank_query_tokens": (
                        config.retrieval.qwen_rerank_query_tokens
                    ),
                    "qwen_rerank_score_weight": (
                        config.retrieval.qwen_rerank_score_weight
                    ),
                    "qwen_rerank_model": config.retrieval.qwen_rerank_model,
                    "qwen_rerank_prefix_layers": (
                        config.retrieval.qwen_rerank_prefix_layers
                    ),
                    "qwen_rerank_attention_layer": (
                        config.retrieval.qwen_rerank_attention_layer
                    ),
                    "qwen_rerank_use_cav": config.retrieval.qwen_rerank_use_cav,
                    "qwen_rerank_cav_layer": (
                        config.retrieval.qwen_rerank_cav_layer
                    ),
                    "qwen_rerank_max_workspace_tokens": (
                        config.retrieval.qwen_rerank_max_workspace_tokens
                    ),
                }
            )
        if config.retrieval.qwen_feedback:
            expected.update(
                {
                    "qwen_feedback": True,
                    "qwen_feedback_candidate_pool": (
                        config.retrieval.qwen_feedback_candidate_pool
                    ),
                    "qwen_feedback_seed_slots": (
                        config.retrieval.qwen_feedback_seed_slots
                    ),
                    "qwen_feedback_slots": config.retrieval.qwen_feedback_slots,
                    "qwen_feedback_evidence_tokens": (
                        config.retrieval.qwen_feedback_evidence_tokens
                    ),
                    "qwen_feedback_query_tokens": (
                        config.retrieval.qwen_feedback_query_tokens
                    ),
                    "qwen_rerank_group_size": (
                        config.retrieval.qwen_rerank_group_size
                    ),
                    "qwen_rerank_beam_per_group": (
                        config.retrieval.qwen_rerank_beam_per_group
                    ),
                    "qwen_rerank_candidate_tokens": (
                        config.retrieval.qwen_rerank_candidate_tokens
                    ),
                    "qwen_rerank_query_tokens": (
                        config.retrieval.qwen_rerank_query_tokens
                    ),
                    "qwen_rerank_model": config.retrieval.qwen_rerank_model,
                    "qwen_rerank_prefix_layers": (
                        config.retrieval.qwen_rerank_prefix_layers
                    ),
                    "qwen_rerank_attention_layer": (
                        config.retrieval.qwen_rerank_attention_layer
                    ),
                    "qwen_rerank_use_cav": config.retrieval.qwen_rerank_use_cav,
                    "qwen_rerank_cav_layer": (
                        config.retrieval.qwen_rerank_cav_layer
                    ),
                    "qwen_rerank_max_workspace_tokens": (
                        config.retrieval.qwen_rerank_max_workspace_tokens
                    ),
                }
            )
    if config.retrieval.mode in {"hybrid_graph", "causal_graph"}:
        expected["neighbor_direction"] = config.retrieval.neighbor_direction
        if config.retrieval.query_facet_retrieval:
            expected.update(
                {
                    "query_facet_retrieval": True,
                    "query_facet_slots": config.retrieval.query_facet_slots,
                    "query_facet_max": config.retrieval.query_facet_max,
                }
            )
        if config.retrieval.role_aware_retrieval:
            expected.update(
                {
                    "role_aware_retrieval": True,
                    "role_user_weight": config.retrieval.role_user_weight,
                    "role_assistant_weight": (
                        config.retrieval.role_assistant_weight
                    ),
                    "role_system_weight": config.retrieval.role_system_weight,
                }
            )
        if config.retrieval.multi_fact_source_diversity:
            expected["multi_fact_source_diversity"] = True
    if config.retrieval.mode in {"causal_consolidation", "causal_graph"}:
        expected.update(
            {
                "consolidation_chunk_slots": (
                    config.retrieval.consolidation_chunk_slots
                ),
                "consolidation_hops": config.retrieval.consolidation_hops,
                "consolidation_candidates": (
                    config.retrieval.consolidation_candidates
                ),
                "consolidation_diffusion_width": (
                    config.retrieval.consolidation_diffusion_width
                ),
                "consolidation_min_count": (
                    config.retrieval.consolidation_min_count
                ),
                "consolidation_expansion_tokens": (
                    config.retrieval.consolidation_expansion_tokens
                ),
                "consolidation_training_expansion_tokens": (
                    config.retrieval.consolidation_training_expansion_tokens
                ),
                "consolidation_budget_aware_packing": (
                    config.retrieval.consolidation_budget_aware_packing
                ),
                "consolidation_training_k": (
                    config.retrieval.consolidation_training_k
                ),
                "consolidation_max_event_nodes": (
                    config.retrieval.consolidation_max_event_nodes
                ),
                "consolidation_new_event_nodes": (
                    config.retrieval.consolidation_new_event_nodes
                ),
                "consolidation_max_training_prompt_tokens": (
                    config.retrieval.consolidation_max_training_prompt_tokens
                ),
            }
        )
        if config.retrieval.consolidation_source_diverse_packing:
            expected["consolidation_source_diverse_packing"] = True
        if config.retrieval.consolidation_query_aware_sentence_packing:
            expected["consolidation_query_aware_sentence_packing"] = True
            expected["consolidation_max_sentences_per_expansion"] = (
                config.retrieval.consolidation_max_sentences_per_expansion
            )
        if config.retrieval.consolidation_information_gain_packing:
            expected["consolidation_information_gain_packing"] = True
            expected["consolidation_min_information_gain_per_token"] = (
                config.retrieval.consolidation_min_information_gain_per_token
            )
        if config.retrieval.consolidation_source_metadata_packing:
            expected["consolidation_source_metadata_packing"] = True
        if config.retrieval.coverage_selection:
            expected.update(
                {
                    "coverage_selection": True,
                    "coverage_selector_backend": (
                        config.retrieval.coverage_selector_backend
                    ),
                    "coverage_selector_model": (
                        config.retrieval.coverage_selector_model
                    ),
                    "coverage_selector_dtype": (
                        config.retrieval.coverage_selector_dtype
                    ),
                    "coverage_selector_candidate_pool": (
                        config.retrieval.coverage_selector_candidate_pool
                    ),
                    "coverage_selector_candidate_tokens": (
                        config.retrieval.coverage_selector_candidate_tokens
                    ),
                    "coverage_selector_query_tokens": (
                        config.retrieval.coverage_selector_query_tokens
                    ),
                    "coverage_selector_max_workspace_tokens": (
                        config.retrieval.coverage_selector_max_workspace_tokens
                    ),
                    "coverage_selector_max_new_tokens": (
                        config.retrieval.coverage_selector_max_new_tokens
                    ),
                    "coverage_selector_null_threshold": (
                        config.retrieval.coverage_selector_null_threshold
                    ),
                    "coverage_selector_uncertainty_entropy": (
                        config.retrieval.coverage_selector_uncertainty_entropy
                    ),
                    "coverage_selector_prefix_layers": (
                        config.retrieval.coverage_selector_prefix_layers
                    ),
                    "coverage_selector_attention_layer": (
                        config.retrieval.coverage_selector_attention_layer
                    ),
                    "coverage_selector_merge_similarity": (
                        config.retrieval.coverage_selector_merge_similarity
                    ),
                    "coverage_selector_same_source_merge_similarity": (
                        config.retrieval.coverage_selector_same_source_merge_similarity
                    ),
                    "coverage_selector_strict": (
                        config.retrieval.coverage_selector_strict
                    ),
                }
            )
            if config.retrieval.allow_selected_scope_fixed_k_closure:
                expected["allow_selected_scope_fixed_k_closure"] = True
            if config.retrieval.coverage_selector_backend in {
                "cross_encoder",
                "cross_encoder_qwen_prefix",
            }:
                expected.update(
                    {
                        "coverage_selector_cross_encoder_model_id": (
                            config.retrieval.coverage_selector_cross_encoder_model_id
                        ),
                        "coverage_selector_cross_encoder_revision": (
                            config.retrieval.coverage_selector_cross_encoder_revision
                        ),
                        "coverage_selector_cross_encoder_checkpoint_sha256": (
                            config.retrieval.coverage_selector_cross_encoder_checkpoint_sha256
                        ),
                        "coverage_selector_cross_encoder_device": (
                            config.retrieval.coverage_selector_cross_encoder_device
                        ),
                        "coverage_selector_cross_encoder_candidate_pool": (
                            config.retrieval.coverage_selector_cross_encoder_candidate_pool
                        ),
                        "coverage_selector_cross_encoder_semantic_rerank": (
                            config.retrieval.coverage_selector_cross_encoder_semantic_rerank
                        ),
                        "coverage_selector_cross_encoder_score_only": (
                            config.retrieval.coverage_selector_cross_encoder_score_only
                        ),
                        "coverage_selector_cross_encoder_batch_size": (
                            config.retrieval.coverage_selector_cross_encoder_batch_size
                        ),
                        "coverage_selector_cross_encoder_max_length": (
                            config.retrieval.coverage_selector_cross_encoder_max_length
                        ),
                    }
                )
            if config.retrieval.coverage_selector_backend in {
                "qwen_prefix",
                "qwen_prefix_choice",
                "cross_encoder_qwen_prefix",
            }:
                expected.update(_coverage_prefix_policy_identity(config))
            if (
                config.retrieval.coverage_selector_backend
                == "qwen_prefix_choice"
            ):
                expected.update(
                    {
                        "coverage_selector_choice_model_id": (
                            config.retrieval.coverage_selector_choice_model_id
                        ),
                        "coverage_selector_choice_revision": (
                            config.retrieval.coverage_selector_choice_revision
                        ),
                        "coverage_selector_choice_checkpoint_sha256": (
                            config.retrieval.coverage_selector_choice_checkpoint_sha256
                        ),
                        "coverage_selector_choice_device": (
                            config.retrieval.coverage_selector_choice_device
                        ),
                        "coverage_selector_choice_dtype": (
                            config.retrieval.coverage_selector_choice_dtype
                        ),
                        "coverage_selector_choice_batch_size": (
                            config.retrieval.coverage_selector_choice_batch_size
                        ),
                        "coverage_selector_choice_max_candidates": (
                            config.retrieval.coverage_selector_choice_max_candidates
                        ),
                        "coverage_selector_choice_query_tokens": (
                            config.retrieval.coverage_selector_choice_query_tokens
                        ),
                        "coverage_selector_choice_candidate_tokens": (
                            config.retrieval.coverage_selector_choice_candidate_tokens
                        ),
                        "coverage_selector_choice_max_prompt_tokens": (
                            config.retrieval.coverage_selector_choice_max_prompt_tokens
                        ),
                        "coverage_selector_choice_max_workspace_tokens": (
                            config.retrieval.coverage_selector_choice_max_workspace_tokens
                        ),
                    }
                )
    return expected


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
    if path is None:
        return ""
    policy_bytes = path.read_bytes()
    payload = json.loads(policy_bytes)
    if not isinstance(payload, dict):
        raise ValueError("policy manifest must be a JSON object")
    raw_status = payload.get("status")
    status = raw_status if isinstance(raw_status, str) else ""
    if not status or status.startswith("superseded"):
        raise ValueError(f"policy manifest is not active: {path}")
    locked_validation = active_split == "validation"
    if payload.get("dataset_sha256") != dataset_sha256:
        raise ValueError("policy manifest dataset SHA-256 mismatch")
    if split_manifest is None or payload.get("split_manifest") != Path(
        split_manifest
    ).name:
        raise ValueError("policy manifest locked-split identity mismatch")
    if locked_validation and "split" not in payload:
        raise ValueError("validation policy manifest must bind the active split")
    if "split" in payload:
        expected_split = payload["split"]
        if (
            not isinstance(expected_split, str)
            or not expected_split
            or expected_split != active_split
        ):
            raise ValueError("policy manifest active split mismatch")
    if locked_validation:
        if payload.get("format") != "memory-condense-retrieval-policy-v1":
            raise ValueError("validation policy manifest format mismatch")
        if status != "validation_frozen":
            raise ValueError(
                "validation requires a policy with status 'validation_frozen'"
            )
    expected_split_sha256 = _optional_policy_sha256(
        payload,
        "split_manifest_sha256",
    )
    if locked_validation and expected_split_sha256 is None:
        raise ValueError("validation policy must bind split_manifest_sha256")
    if expected_split_sha256 is not None:
        actual_split_sha256 = file_sha256(split_manifest)
        if actual_split_sha256 != expected_split_sha256:
            raise ValueError("policy manifest locked-split SHA-256 mismatch")

    expected_implementation_sha256 = _optional_policy_sha256(
        payload,
        "implementation_sha256",
    )
    if locked_validation and expected_implementation_sha256 is None:
        raise ValueError("validation policy must bind implementation_sha256")
    if expected_implementation_sha256 is not None:
        actual_implementation_sha256 = (
            active_implementation_sha256 or implementation_sha256()
        ).casefold()
        if actual_implementation_sha256 != expected_implementation_sha256:
            raise ValueError("policy manifest implementation SHA-256 mismatch")

    expected_environment_sha256 = _optional_policy_sha256(
        payload,
        "environment_lock_sha256",
    )
    if locked_validation and expected_environment_sha256 is None:
        raise ValueError("validation policy must bind environment_lock_sha256")
    if expected_environment_sha256 is not None:
        actual_environment_sha256 = (
            active_environment_lock_sha256 or environment_lock_sha256()
        ).casefold()
        if actual_environment_sha256 != expected_environment_sha256:
            raise ValueError("policy manifest environment-lock SHA-256 mismatch")

    selection_artifact_required = payload.get(
        "selection_artifact_required",
        False,
    )
    if not isinstance(selection_artifact_required, bool):
        raise ValueError(
            "policy manifest selection_artifact_required must be boolean"
        )
    if locked_validation and not selection_artifact_required:
        raise ValueError(
            "validation policy must require its development selection artifact"
        )
    if selection_artifact_required:
        has_selection_artifact = "selection_artifact" in payload
        has_selection_sha256 = "selection_artifact_sha256" in payload
        if not has_selection_artifact or not has_selection_sha256:
            raise ValueError(
                "policy manifest selection artifact and SHA-256 must be "
                "provided together when required"
            )
        selection_sha256 = _optional_policy_sha256(
            payload,
            "selection_artifact_sha256",
        )
        assert selection_sha256 is not None
        selection_path = _policy_repository_file(
            payload["selection_artifact"],
            field="selection_artifact",
            repository_root=repository_root,
        )
        if file_sha256(selection_path) != selection_sha256:
            raise ValueError("policy manifest selection artifact SHA-256 mismatch")

    if locked_validation:
        frozen_evaluation = payload.get("evaluation")
        if not isinstance(frozen_evaluation, dict):
            raise ValueError("validation policy must contain an evaluation object")
        frozen_evaluation = dict(frozen_evaluation)
        claim_profile = claimed_validation_profile(payload)
        if claim_profile:
            validate_longmemeval_claim_profile(payload, frozen_evaluation)
        sample_offsets = frozen_evaluation.pop("sample_offsets", None)
        if (
            not isinstance(sample_offsets, list)
            or not sample_offsets
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in sample_offsets
            )
            or len(set(sample_offsets)) != len(sample_offsets)
        ):
            raise ValueError(
                "validation evaluation.sample_offsets must be unique "
                "non-negative integers"
            )
        if frozen_evaluation.get("use_judge") is not True:
            raise ValueError("validation evaluation must enable the judge")
        if frozen_evaluation.get("provider_retries") != 0:
            raise ValueError("validation evaluation must freeze provider_retries=0")
        stress_target = frozen_evaluation.get("stress_context_tokens")
        stress_questions = frozen_evaluation.get("stress_questions")
        if (
            isinstance(stress_target, bool)
            or not isinstance(stress_target, int)
            or stress_target < 1
        ):
            raise ValueError(
                "validation evaluation must set a positive stress_context_tokens"
            )
        if (
            isinstance(stress_questions, bool)
            or not isinstance(stress_questions, int)
            or stress_questions < 1
        ):
            raise ValueError(
                "validation evaluation must set a positive stress_questions"
            )
        if frozen_evaluation.get("stress_question_offset") != 0:
            raise ValueError(
                "validation evaluation must freeze stress_question_offset=0"
            )
        if frozen_evaluation.get("max_samples") != 1:
            raise ValueError("validation evaluation must freeze max_samples=1")
        for field in ("responder_model", "judge_model", "embedding_device"):
            value = frozen_evaluation.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"validation evaluation must bind a non-empty {field}"
                )
        if frozen_evaluation.get("benchmark_format") != "longmemeval":
            raise ValueError(
                "validation evaluation must freeze "
                "benchmark_format='longmemeval'"
            )
        max_prompt_tokens = frozen_evaluation.get("max_prompt_tokens")
        if (
            isinstance(max_prompt_tokens, bool)
            or not isinstance(max_prompt_tokens, int)
            or max_prompt_tokens < 1
        ):
            raise ValueError(
                "validation evaluation must set a positive max_prompt_tokens"
            )
        accuracy_target = frozen_evaluation.get("accuracy_target")
        if (
            isinstance(accuracy_target, bool)
            or not isinstance(accuracy_target, (int, float))
            or not math.isfinite(float(accuracy_target))
            or not 0.0 <= float(accuracy_target) <= 1.0
        ):
            raise ValueError(
                "validation evaluation must set accuracy_target in [0, 1]"
            )
        min_target_questions = frozen_evaluation.get("min_target_questions")
        if (
            isinstance(min_target_questions, bool)
            or not isinstance(min_target_questions, int)
            or min_target_questions < 1
        ):
            raise ValueError(
                "validation evaluation must set a positive min_target_questions"
            )
        recent_window = frozen_evaluation.get("recent_window")
        if (
            isinstance(recent_window, bool)
            or not isinstance(recent_window, int)
            or recent_window < 0
        ):
            raise ValueError(
                "validation evaluation must set a non-negative recent_window"
            )
        authorization = frozen_evaluation.get("max_provider_calls")
        if (
            isinstance(authorization, bool)
            or not isinstance(authorization, int)
            or authorization != 2 * stress_questions
        ):
            raise ValueError(
                "validation evaluation must authorize exactly one responder "
                "and one judge call per question"
            )
        if evaluation_identity is None:
            raise ValueError("validation policy requires active evaluation identity")
        active_evaluation = dict(evaluation_identity)
        active_offset = active_evaluation.pop("sample_offset", None)
        if (
            isinstance(active_offset, bool)
            or not isinstance(active_offset, int)
            or active_offset not in sample_offsets
        ):
            raise ValueError("validation shard sample_offset is not in the policy")
        if prepare_only:
            cache_shaping_fields = (
                "embedding_device",
                "benchmark_format",
                "stress_context_tokens",
                "stress_questions",
                "stress_question_offset",
                "max_samples",
            )
            expected_prepare = {
                field: frozen_evaluation.get(field)
                for field in cache_shaping_fields
            }
            actual_prepare = {
                field: active_evaluation.get(field)
                for field in cache_shaping_fields
            }
            if actual_prepare != expected_prepare:
                raise ValueError(
                    "policy manifest cache-preparation config mismatch: expected "
                    f"{expected_prepare}, got {actual_prepare}"
                )
        else:
            if active_evaluation != frozen_evaluation:
                raise ValueError(
                    "policy manifest evaluation config mismatch: expected "
                    f"{frozen_evaluation}, got {active_evaluation}"
                )

    retrieval = payload.get("retrieval", {})
    expected = _policy_retrieval_identity(config)
    if retrieval != expected:
        raise ValueError(
            f"policy manifest retrieval config mismatch: expected {retrieval}, "
            f"got {expected}"
        )
    return hashlib.sha256(policy_bytes).hexdigest()


def _optional_policy_sha256(payload: dict, field: str) -> str | None:
    if field not in payload:
        return None
    value = payload[field]
    if not isinstance(value, str):
        raise ValueError(f"policy manifest {field} must be a SHA-256 digest")
    normalized = value.casefold()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"policy manifest {field} must be a SHA-256 digest")
    return normalized


def _policy_repository_file(
    value: object,
    *,
    field: str,
    repository_root: str | Path | None,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"policy manifest {field} must be a repository-relative path"
        )
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"policy manifest {field} must be a safe repository-relative path"
        )
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else project_root().resolve()
    )
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"policy manifest {field} must stay within the repository"
        ) from exc
    if not candidate.is_file():
        raise ValueError(f"policy manifest {field} does not name an existing file")
    return candidate


def _assert_implementation_unchanged(expected_sha256: str) -> None:
    if implementation_sha256() != expected_sha256:
        raise RuntimeError("implementation changed during benchmark run")


def _benchmark_ingest_fn(
    args: argparse.Namespace,
    config: EvalConfig,
    *,
    prepare_only: bool = False,
):
    """Select an isolated benchmark writer without mutating compiled stores."""

    require_cache_hit = bool(
        not prepare_only and getattr(args, "benchmark_split", None) == "validation"
    )
    if require_cache_hit and config.retrieval.mode not in {
        "causal_consolidation",
        "causal_graph",
    }:
        raise ValueError(
            "locked validation requires causal compiled+learned cache receipts"
        )
    if config.retrieval.mode in {"causal_consolidation", "causal_graph"}:
        from memory_condense.eval.causal_benchmark import (
            causal_consolidation_ingest_fn,
        )

        return causal_consolidation_ingest_fn(
            args.compiled_store_cache,
            causal_cache_root=args.causal_store_cache,
            device=config.embedding_device,
            prepare_only=prepare_only,
            require_cache_hit=require_cache_hit,
        )
    if args.compiled_store_cache:
        return compiled_store_ingest_fn(
            args.compiled_store_cache,
            device=config.embedding_device,
            require_cache_hit=require_cache_hit,
        )
    return ingest_sample


def _load_candidate_reranker(args: argparse.Namespace, config: EvalConfig):
    """Load one shared, bounded Qwen control plane for a benchmark run."""

    if not (config.retrieval.qwen_rerank or config.retrieval.qwen_feedback):
        return None
    if args.qwen_rerank_model_dir is None:
        raise ValueError("Qwen attention requires --qwen-rerank-model-dir")
    from memory_condense.association_store import AssociationArtifact
    from memory_condense.qwen_consolidation import load_qwen_linker
    from memory_condense.qwen_rerank import QwenCandidateReranker

    print(f"Loading bounded Qwen reranker from {args.qwen_rerank_model_dir}...")
    linker = load_qwen_linker(
        args.qwen_rerank_model_dir,
        prefix_layers=config.retrieval.qwen_rerank_prefix_layers,
        attention_layer=config.retrieval.qwen_rerank_attention_layer,
        cav_report=(
            args.qwen_rerank_cav_report
            if config.retrieval.qwen_rerank_use_cav
            else None
        ),
        cav_vectors=(
            args.qwen_rerank_cav_vectors
            if config.retrieval.qwen_rerank_use_cav
            else None
        ),
        cav_layer=config.retrieval.qwen_rerank_cav_layer,
        device=args.qwen_rerank_device,
        dtype=args.qwen_rerank_dtype,
        max_candidates=8,
        max_workspace_tokens=config.retrieval.qwen_rerank_max_workspace_tokens,
    )
    artifact = None
    if linker.cav_bank is not None:
        report_payload = json.loads(
            Path(args.qwen_rerank_cav_report).read_text(encoding="utf-8")
        )
        index_path = Path(args.qwen_rerank_model_dir) / "model.safetensors.index.json"
        artifact = AssociationArtifact.create(
            model_id=str(report_payload.get("model", "Qwen/Qwen3-8B")),
            checkpoint_id=f"safetensors-index:{file_sha256(index_path)}",
            prefix_layers=config.retrieval.qwen_rerank_prefix_layers,
            head_layer=config.retrieval.qwen_rerank_attention_layer,
            cav_layer=config.retrieval.qwen_rerank_cav_layer,
            concept_names=linker.cav_bank.names,
            head_count=int(linker.encoder.config.num_attention_heads),
            metadata={
                "cav_dataset_sha256": report_payload.get("dataset_sha256"),
                "cav_vectors_sha256": file_sha256(args.qwen_rerank_cav_vectors),
                "pooling": "conceptual-span-max-v1",
            },
        )
    return QwenCandidateReranker(
        linker,
        candidate_pool=(
            config.retrieval.qwen_feedback_candidate_pool
            if config.retrieval.qwen_feedback
            else config.retrieval.qwen_rerank_candidate_pool
        ),
        qwen_slots=(
            config.retrieval.qwen_feedback_seed_slots
            if config.retrieval.qwen_feedback
            else config.retrieval.qwen_rerank_slots
        ),
        group_size=config.retrieval.qwen_rerank_group_size,
        beam_per_group=config.retrieval.qwen_rerank_beam_per_group,
        candidate_tokens=config.retrieval.qwen_rerank_candidate_tokens,
        query_tokens=(
            config.retrieval.qwen_feedback_query_tokens
            if config.retrieval.qwen_feedback
            else config.retrieval.qwen_rerank_query_tokens
        ),
        score_weight=config.retrieval.qwen_rerank_score_weight,
        association_artifact=artifact,
    )


def _attach_candidate_reranker(ingest_fn, reranker):
    if reranker is None:
        return ingest_fn

    def attached(sample, config, data_dir):
        condenser = ingest_fn(sample, config, data_dir)
        condenser.set_source_candidate_reranker(reranker)
        artifact = getattr(reranker, "association_artifact", None)
        if artifact is not None:
            report = condenser.compile_indexed_cav_signatures(
                reranker.linker,
                artifact,
                batch_size=32,
                roles=("user",),
            )
            print(
                "  CAV concept index: "
                f"{report['compiled']} compiled, {report['reused']} reused, "
                f"{report['compiled_spans']} spans, "
                f"width={report['signature_width']}"
            )
        return condenser

    return attached


class _LazyQwenPrefixCoverageSelector:
    """Load the full-width prefix only after BGE has left the shared GPU."""

    requires_staged_gpu = True
    requires_baseline_ranking = True
    requires_complete_frontier = True

    def __init__(
        self,
        load,
        *,
        strict=False,
        allow_selected_scope_fixed_k_closure=False,
    ):
        self._load = load
        self._selector = None
        self.strict = bool(strict)
        self.allow_selected_scope_fixed_k_closure = bool(
            allow_selected_scope_fixed_k_closure
        )
        self.last_report = None
        self.last_source_companion_report = None
        self.last_candidate_trace = []
        self.load_elapsed_s = 0.0

    @property
    def loaded(self) -> bool:
        return self._selector is not None

    @staticmethod
    def requires_complete_frontier_for(query: str) -> bool:
        """Resolve query shape without loading either staged checkpoint."""

        from memory_condense.coverage_selector import compile_set_program

        return bool(compile_set_program(query).requires_completeness)

    def _ensure_loaded(self):
        if self._selector is None:
            started = time.perf_counter()
            self._selector = self._load()
            self.load_elapsed_s += time.perf_counter() - started
        return self._selector

    def select(self, query, candidates, **kwargs):
        self.last_candidate_trace = []
        selector = self._ensure_loaded()
        selected = selector.select(query, candidates, **kwargs)
        self.last_report = selector.last_report
        self.last_candidate_trace = list(
            getattr(selector, "last_candidate_trace", ())
        )
        return selected

    def select_source_companions(
        self,
        query,
        candidates_by_source,
        *,
        source_timestamps=None,
    ):
        self.last_source_companion_report = None
        selector = self._ensure_loaded()
        choose = getattr(selector, "select_source_companions", None)
        report_owner = selector
        if not callable(choose):
            score_provider = getattr(selector, "score_provider", None)
            choose = getattr(score_provider, "select_source_companions", None)
            report_owner = score_provider
        if not callable(choose):
            return {
                str(source_id): candidates[0]
                for source_id, candidates in candidates_by_source.items()
                if candidates
            }
        if source_timestamps is None:
            selected = choose(query, candidates_by_source)
        else:
            selected = choose(
                query,
                candidates_by_source,
                source_timestamps=source_timestamps,
            )
        self.last_source_companion_report = getattr(
            report_owner,
            "last_source_companion_report",
            None,
        )
        return selected

    def close(self) -> None:
        selector = self._selector
        self._selector = None
        self.last_report = None
        self.last_source_companion_report = None
        if selector is not None:
            selector.close()


class _LazyLocalINICoverageSelector(_LazyQwenPrefixCoverageSelector):
    """Load the local INI classifier only after BGE leaves the shared GPU."""


class _LazyCrossEncoderCoverageSelector(_LazyQwenPrefixCoverageSelector):
    """Load semantic reranking only after BGE leaves the shared GPU."""

    requires_baseline_ranking = False

    def __init__(
        self,
        load,
        *,
        strict=False,
        semantic_rerank=True,
        semantic_score_only=False,
        allow_selected_scope_fixed_k_closure=False,
    ):
        super().__init__(
            load,
            strict=strict,
            allow_selected_scope_fixed_k_closure=(
                allow_selected_scope_fixed_k_closure
            ),
        )
        self.strict = bool(strict)
        self.semantic_rerank = bool(semantic_rerank)
        self.semantic_score_only = bool(semantic_score_only)
        self.requires_baseline_ranking = not self.semantic_rerank

    def select_source_companions(self, query, candidates_by_source):
        self.last_source_companion_report = None
        selector = self._ensure_loaded()
        selected = selector.select_source_companions(
            query,
            candidates_by_source,
        )
        self.last_source_companion_report = getattr(
            selector,
            "last_source_companion_report",
            None,
        )
        return selected


def _load_coverage_selector(args: argparse.Namespace, config: EvalConfig):
    """Load one shared transient coverage backend for a benchmark run."""

    if not config.retrieval.coverage_selection:
        return None
    if args.qwen_rerank_model_dir is not None:
        raise ValueError(
            "coverage selection and the source Qwen reranker are separate arms; "
            "measure them in separate processes"
        )

    def load_prefix_selector(*, score_provider=None):
        if args.coverage_selector_qwen_prefix_model_dir is None:
            raise ValueError(
                "Qwen prefix coverage selection requires "
                "--coverage-selector-qwen-prefix-model-dir"
            )

        import torch

        from memory_condense.coverage_selector import QwenPrefixCoverageSelector
        from memory_condense.eval.local_qwen import resolve_local_qwen_dtype
        from memory_condense.head_memory import QwenMemoryLinker
        from memory_condense.qwen_prefix import Qwen3PrefixEncoder

        _torch_dtype, dtype_name = resolve_local_qwen_dtype(
            torch,
            config.retrieval.coverage_selector_prefix_dtype
            or config.retrieval.coverage_selector_dtype,
            device=config.retrieval.coverage_selector_prefix_device
            or args.coverage_selector_prefix_device,
        )
        print(
            "Loading staged Qwen3-8B prefix coverage selector from "
            f"{args.coverage_selector_qwen_prefix_model_dir}...",
            flush=True,
        )
        encoder = Qwen3PrefixEncoder(
            args.coverage_selector_qwen_prefix_model_dir,
            layers=config.retrieval.coverage_selector_prefix_layers,
            device=config.retrieval.coverage_selector_prefix_device,
            dtype=dtype_name,
            model_id=config.retrieval.coverage_selector_prefix_model_id,
            model_revision=config.retrieval.coverage_selector_prefix_revision,
            expected_checkpoint_sha256=(
                config.retrieval.coverage_selector_prefix_checkpoint_sha256
            ),
        )
        linker = QwenMemoryLinker(
            encoder,
            layer=config.retrieval.coverage_selector_attention_layer,
            max_candidates=config.retrieval.coverage_selector_candidate_pool,
            max_workspace_tokens=(
                config.retrieval.coverage_selector_max_workspace_tokens
            ),
        )
        print(
            "  loaded layers: 0.."
            f"{config.retrieval.coverage_selector_prefix_layers - 1}; "
            "QK/OV readout layer: "
            f"{config.retrieval.coverage_selector_attention_layer}; "
            f"dtype: {dtype_name}; checkpoint: "
            f"{config.retrieval.coverage_selector_prefix_checkpoint_sha256[:12]}...; "
            "LM head: absent",
            flush=True,
        )
        return QwenPrefixCoverageSelector(
            linker,
            score_provider=score_provider,
            candidate_pool=config.retrieval.coverage_selector_candidate_pool,
            candidate_tokens=config.retrieval.coverage_selector_candidate_tokens,
            query_tokens=config.retrieval.coverage_selector_query_tokens,
            merge_similarity=config.retrieval.coverage_selector_merge_similarity,
            same_source_merge_similarity=(
                config.retrieval.coverage_selector_same_source_merge_similarity
            ),
            null_threshold=config.retrieval.coverage_selector_null_threshold,
            uncertainty_entropy=(
                config.retrieval.coverage_selector_uncertainty_entropy
            ),
            allow_selected_scope_fixed_k_closure=(
                config.retrieval.allow_selected_scope_fixed_k_closure
            ),
            strict=config.retrieval.coverage_selector_strict,
        )

    if config.retrieval.coverage_selector_backend == "qwen_prefix":

        return _LazyQwenPrefixCoverageSelector(
            load_prefix_selector,
            strict=config.retrieval.coverage_selector_strict,
            allow_selected_scope_fixed_k_closure=(
                config.retrieval.allow_selected_scope_fixed_k_closure
            ),
        )

    if config.retrieval.coverage_selector_backend == "qwen_prefix_choice":
        if args.coverage_selector_choice_model_dir is None:
            raise ValueError(
                "forced-choice coverage selection requires "
                "--coverage-selector-choice-model-dir"
            )

        def load_choice_selector():
            from memory_condense.causal_choice_scorer import CausalChoiceScorer

            print(
                "Loading staged generation-free choice scorer from "
                f"{args.coverage_selector_choice_model_dir} "
                f"({config.retrieval.coverage_selector_choice_model_id}@"
                f"{config.retrieval.coverage_selector_choice_revision[:12]}, "
                "K/V cache disabled)...",
                flush=True,
            )
            scorer = CausalChoiceScorer.from_local_checkpoint(
                args.coverage_selector_choice_model_dir,
                model_id=config.retrieval.coverage_selector_choice_model_id,
                model_revision=(
                    config.retrieval.coverage_selector_choice_revision
                ),
                expected_weights_sha256=(
                    config.retrieval.coverage_selector_choice_checkpoint_sha256
                ),
                device=config.retrieval.coverage_selector_choice_device,
                dtype=config.retrieval.coverage_selector_choice_dtype,
                batch_size=config.retrieval.coverage_selector_choice_batch_size,
                max_candidates=(
                    config.retrieval.coverage_selector_choice_max_candidates
                ),
                query_tokens=(
                    config.retrieval.coverage_selector_choice_query_tokens
                ),
                candidate_tokens=(
                    config.retrieval.coverage_selector_choice_candidate_tokens
                ),
                max_prompt_tokens=(
                    config.retrieval.coverage_selector_choice_max_prompt_tokens
                ),
                max_workspace_tokens=(
                    config.retrieval.coverage_selector_choice_max_workspace_tokens
                ),
                require_single_token_labels=True,
                strict=config.retrieval.coverage_selector_strict,
            )
            try:
                return load_prefix_selector(score_provider=scorer)
            except BaseException:
                scorer.close()
                raise

        return _LazyQwenPrefixCoverageSelector(
            load_choice_selector,
            strict=config.retrieval.coverage_selector_strict,
            allow_selected_scope_fixed_k_closure=(
                config.retrieval.allow_selected_scope_fixed_k_closure
            ),
        )

    if config.retrieval.coverage_selector_backend in {
        "cross_encoder",
        "cross_encoder_qwen_prefix",
    }:
        if args.coverage_selector_cross_encoder_model_dir is None:
            raise ValueError(
                "MS MARCO coverage selection requires "
                "--coverage-selector-cross-encoder-model-dir"
            )

        def load_cross_encoder_selector():
            import gc

            import torch
            from sentence_transformers import CrossEncoder

            from memory_condense.cross_encoder_selector import (
                MS_MARCO_MODEL_ID,
                MS_MARCO_MODEL_REVISION,
                MSMarcoCrossEncoderSelector,
                verify_ms_marco_checkpoint,
            )

            checkpoint_sha256 = verify_ms_marco_checkpoint(
                args.coverage_selector_cross_encoder_model_dir
            )
            print(
                "Loading staged MS MARCO cross-encoder from "
                f"{args.coverage_selector_cross_encoder_model_dir} "
                f"({MS_MARCO_MODEL_ID}@{MS_MARCO_MODEL_REVISION[:12]}, "
                f"sha256={checkpoint_sha256[:12]}...)...",
                flush=True,
            )
            encoder = CrossEncoder(
                str(args.coverage_selector_cross_encoder_model_dir),
                device=config.retrieval.coverage_selector_cross_encoder_device,
                local_files_only=True,
                trust_remote_code=False,
                max_length=(
                    config.retrieval.coverage_selector_cross_encoder_max_length
                ),
                model_kwargs={"use_safetensors": True},
            )
            duplicate_grouper = None
            try:
                if (
                    config.retrieval.coverage_selector_backend
                    == "cross_encoder_qwen_prefix"
                ):
                    duplicate_grouper = load_prefix_selector()
                return MSMarcoCrossEncoderSelector(
                    encoder,
                    candidate_pool=(
                        config.retrieval.coverage_selector_cross_encoder_candidate_pool
                    ),
                    candidate_tokens=(
                        config.retrieval.coverage_selector_candidate_tokens
                    ),
                    query_tokens=config.retrieval.coverage_selector_query_tokens,
                    batch_size=(
                        config.retrieval.coverage_selector_cross_encoder_batch_size
                    ),
                    max_length=(
                        config.retrieval.coverage_selector_cross_encoder_max_length
                    ),
                    max_workspace_tokens=(
                        config.retrieval.coverage_selector_max_workspace_tokens
                    ),
                    duplicate_grouper=duplicate_grouper,
                    checkpoint_sha256=checkpoint_sha256,
                    semantic_rerank=(
                        config.retrieval.coverage_selector_cross_encoder_semantic_rerank
                    ),
                    semantic_score_only=(
                        config.retrieval.coverage_selector_cross_encoder_score_only
                    ),
                    strict=config.retrieval.coverage_selector_strict,
                )
            except BaseException:
                if duplicate_grouper is not None:
                    duplicate_grouper.close()
                del encoder
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise

        return _LazyCrossEncoderCoverageSelector(
            load_cross_encoder_selector,
            strict=config.retrieval.coverage_selector_strict,
            semantic_rerank=(
                config.retrieval.coverage_selector_cross_encoder_semantic_rerank
            ),
            semantic_score_only=(
                config.retrieval.coverage_selector_cross_encoder_score_only
            ),
            allow_selected_scope_fixed_k_closure=(
                config.retrieval.allow_selected_scope_fixed_k_closure
            ),
        )

    if config.retrieval.coverage_selector_backend != "local_ini":
        raise ValueError(
            "unsupported coverage selector backend: "
            f"{config.retrieval.coverage_selector_backend}"
        )
    if args.coverage_selector_local_model_dir is None:
        raise ValueError(
            "local INI coverage selection requires --coverage-selector-local-model-dir"
        )

    def load_local_ini_selector():
        from memory_condense.coverage_selector import QueryConditionedCoverageSelector
        from memory_condense.eval.local_qwen import LocalQwenAnswerer

        print(
            "Loading staged local INI coverage selector from "
            f"{args.coverage_selector_local_model_dir}...",
            flush=True,
        )
        answerer = LocalQwenAnswerer(
            args.coverage_selector_local_model_dir,
            max_new_tokens=config.retrieval.coverage_selector_max_new_tokens,
            gpu_memory=args.coverage_selector_gpu_memory,
            cpu_memory=args.coverage_selector_cpu_memory,
            dtype=config.retrieval.coverage_selector_dtype,
            stop_strings=("[end]",),
        )
        print(
            f"  selector generation dtype: {answerer.dtype_name}",
            flush=True,
        )
        return QueryConditionedCoverageSelector(
            answerer,
            candidate_pool=config.retrieval.coverage_selector_candidate_pool,
            candidate_tokens=config.retrieval.coverage_selector_candidate_tokens,
            query_tokens=config.retrieval.coverage_selector_query_tokens,
            max_workspace_tokens=(
                config.retrieval.coverage_selector_max_workspace_tokens
            ),
            null_threshold=config.retrieval.coverage_selector_null_threshold,
            uncertainty_entropy=(
                config.retrieval.coverage_selector_uncertainty_entropy
            ),
            strict=config.retrieval.coverage_selector_strict,
        )

    return _LazyLocalINICoverageSelector(
        load_local_ini_selector,
        allow_selected_scope_fixed_k_closure=(
            config.retrieval.allow_selected_scope_fixed_k_closure
        ),
    )


def _attach_coverage_selector(ingest_fn, selector):
    if selector is None:
        return ingest_fn

    def attached(sample, config, data_dir):
        staged = bool(getattr(selector, "requires_staged_gpu", False))
        if staged and getattr(selector, "loaded", False):
            # A later sample needs BGE first; shed the prior sample's transient
            # selector before its frozen query-vector batch is prepared.
            selector.close()
        condenser = ingest_fn(sample, config, data_dir)
        if staged:
            release_embedder = getattr(ingest_fn, "release_embedder", None)
            if callable(release_embedder):
                print(
                    "Staging GPU: frozen query vectors ready; releasing BGE "
                    "before coverage-selector load.",
                    flush=True,
                )
                release_embedder()
        condenser.set_context_candidate_selector(selector)
        return condenser

    return attached


def _attach_runtime_controls(ingest_fn, *, reranker=None, selector=None):
    """Compose independent transient controls around one benchmark writer."""

    return _attach_coverage_selector(
        _attach_candidate_reranker(ingest_fn, reranker),
        selector,
    )


def _reserve_embedding_device_for_transient_models(
    args: argparse.Namespace,
) -> None:
    """Keep BGE on GPU when a causal run can release it before selection."""

    coverage_selector = bool(
        args.coverage_selector_local_model_dir
        or args.coverage_selector_qwen_prefix_model_dir
        or args.coverage_selector_cross_encoder_model_dir
    )
    staged_coverage = coverage_selector and args.mode in {
        "causal_consolidation",
        "causal_graph",
    }
    if (
        args.embedding_device is None
        and (args.qwen_rerank_model_dir or (coverage_selector and not staged_coverage))
    ):
        # Non-causal stores still need their live BGE embedder during search.
        args.embedding_device = "cpu"


def run_compare(args: argparse.Namespace) -> None:
    baseline_path, treatment_path = args.compare
    baseline = load_run(baseline_path)
    treatment = load_run(treatment_path)
    report = compare_runs(baseline, treatment)
    print_comparison(report)

    if args.csv:
        Path(args.csv).write_text(to_csv(treatment), encoding="utf-8")
        print(f"\nPer-turn CSV written to {args.csv}")


def run_answer_recall(args: argparse.Namespace) -> None:
    """Offline: is the gold answer even reachable from the assembled context?

    Free, keyless, and the cheap predictor of the paid comparison — if the
    memory arm's context holds the answer less often than the dense arm's, no
    responder can recover the difference.
    """
    print(f"Loading benchmark from {args.answer_recall}...")
    samples = load_benchmark(args.answer_recall, args.benchmark_format)
    if not samples:
        print("No samples parsed. Check --benchmark-format.")
        return

    samples = _apply_sample_offset(args, _apply_locked_split(args, samples))
    stress_tokens = getattr(args, "stress_context_tokens", None)
    stress_question_offset = 0
    stress_question_count = None
    if stress_tokens is not None:
        from memory_condense.eval.context_stress import (
            compose_context_stress_sample,
            transcript_tokens,
        )

        stress_question_offset = int(
            getattr(args, "stress_question_offset", 0)
        )
        stress_question_count = int(getattr(args, "stress_questions", 10))
        if stress_question_offset < 0:
            raise ValueError("--stress-question-offset must be non-negative")
        if stress_question_count < 1:
            raise ValueError("--stress-questions must be positive")
        # Keep the canonical ten-question stress sample as the causal-store
        # cache identity. Question sharding happens only after that immutable
        # store is opened; held-out questions do not change its learned graph.
        stress_question_pool = max(
            10,
            stress_question_offset + stress_question_count,
        )
        samples = [
            compose_context_stress_sample(
                samples,
                target_tokens=stress_tokens,
                max_questions=stress_question_pool,
            )
        ]
        actual_tokens = transcript_tokens(samples[0])
        print(
            f"Context stress memory: {actual_tokens:,} tokens, "
            f"{len(samples[0].turns):,} turns, "
            f"{len(samples[0].questions)} questions"
        )
    _reserve_embedding_device_for_transient_models(args)
    config = config_from_args(args)
    print(
        f"{len(samples)} sample(s); measuring "
        f"{args.max_samples or len(samples)} in {config.retrieval.mode} mode. "
        "No API calls will be made."
    )
    reranker = _load_candidate_reranker(args, config)
    selector = _load_coverage_selector(args, config)
    try:
        report = run_recall(
            samples,
            config,
            benchmark=Path(args.answer_recall).stem,
            max_samples=1 if stress_tokens is not None else args.max_samples,
            ingest_fn=_attach_runtime_controls(
                _benchmark_ingest_fn(args, config),
                reranker=reranker,
                selector=selector,
            ),
            question_offset=stress_question_offset,
            max_questions=stress_question_count,
        )
    finally:
        if reranker is not None:
            reranker.close()
        if selector is not None:
            selector.close()
    print_recall_report(report)

    if args.csv:
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            [
                "question_id",
                "category",
                "in_haystack",
                "in_context",
                "best_f1",
                "in_header",
                "in_expansions",
                "context_tokens",
                "evidence_source_recall",
                "all_evidence_sources",
                "retrieved_source_ids",
                "raw_evidence_source_recall",
                "raw_all_evidence_sources",
                "raw_retrieved_source_ids",
                "answer_value_components_expected",
                "answer_value_components_found",
                "answer_value_component_recall",
                "all_answer_value_components",
                "answer_value_component_hit_mask",
                "answer_value_metric_kind",
                "source_companion_requested",
                "source_companion_hydrated",
                "source_companion_orphans",
                "source_companion_direct_date_retained",
                "source_companion_candidates_before",
                "source_companion_candidates_after",
                "selected_partitions",
                "partition_ranking",
                "direct_chunks",
                "consolidation_chunks",
                "causal_events",
                "causal_graph_edges",
                "causal_write_s",
                "qwen_rerank_passes",
                "qwen_candidate_inspections",
                "qwen_max_workspace_candidates",
                "qwen_max_workspace_tokens",
                "qwen_candidates_added",
                "qwen_feedback_rounds",
                "qwen_feedback_seed_sources",
                "qwen_feedback_candidates_added",
                "qwen_feedback_activation_candidates",
                "qwen_feedback_query_tokens",
                "coverage_selector_inspected",
                "coverage_selector_classified",
                "coverage_selector_clusters",
                "coverage_selector_null",
                "coverage_selector_uncertain",
                "coverage_selector_output",
                "coverage_selector_representatives",
                "coverage_selector_workspace_tokens",
                "coverage_selector_elapsed_s",
                "coverage_selector_operator",
                "coverage_selector_cardinality",
                "coverage_selector_quantifier",
                "coverage_selector_ordering",
                "coverage_selector_query_timestamp",
                "coverage_selector_temporal_window_days",
                "coverage_selector_posterior_kind",
                "coverage_selector_semantic_score_kind",
                "coverage_selector_answerability_score_kind",
                "coverage_selector_frontier_candidates",
                "coverage_selector_frontier_attempted",
                "coverage_selector_frontier_uninspected",
                "coverage_selector_frontier_exhaustive",
                "coverage_selector_frontier_batches",
                "coverage_selector_routed_frontier_exhaustive",
                "coverage_selector_active_partition_total",
                "coverage_selector_active_partition_inspected",
                "coverage_selector_active_partition_exhaustive",
                "coverage_selector_active_partition_sources_total",
                "coverage_selector_active_partition_structural_rows",
                "coverage_selector_active_partition_structural_hypotheses",
                "coverage_selector_active_partition_candidates_admitted",
                "coverage_selector_active_partition_candidates_already_present",
                "coverage_selector_active_partition_candidates_replaced",
                "coverage_selector_active_partition_candidates_truncated",
                "coverage_selector_active_partition_structural_overflow",
                "coverage_selector_active_partition_scan_contract",
                "coverage_selector_active_partition_semantically_complete",
                "coverage_selector_partition_scope_kind",
                "coverage_selector_partition_inventory_total",
                "coverage_selector_selected_partition_count",
                "coverage_selector_partition_scope_exhaustive",
                "coverage_selector_selected_scope_structurally_complete",
                "coverage_selector_global_semantic_complete",
                "coverage_selector_allow_selected_scope_fixed_k_closure",
                "closure_applied",
                "closure_scope",
                "closure_global_recall_guaranteed",
                "coverage_selector_cardinality_deficit",
                "coverage_selector_credible_clusters",
                "coverage_selector_reserved_representatives",
                "coverage_selector_structural_eligible_clusters",
                "coverage_selector_structural_reserved_representatives",
                "coverage_selector_score_provider_fallback",
                "coverage_selector_score_provider_model_id",
                "coverage_selector_score_provider_model_revision",
                "coverage_selector_score_provider_checkpoint_sha256",
                "coverage_selector_score_provider_device",
                "coverage_selector_score_provider_dtype",
                "coverage_selector_score_provider_forward_passes",
                "coverage_selector_score_provider_peak_workspace_tokens",
                "coverage_selector_score_provider_total_workspace_tokens",
                "coverage_selector_score_provider_elapsed_s",
                "coverage_selector_score_provider_retained_state_bytes",
                "coverage_selector_prefix_model_id",
                "coverage_selector_prefix_model_revision",
                "coverage_selector_prefix_checkpoint_sha256",
                "coverage_selector_prefix_device",
                "coverage_selector_prefix_dtype",
                "coverage_selector_prefix_layers",
                "coverage_selector_prefix_attention_layer",
                "coverage_selector_model_id",
                "coverage_selector_model_revision",
                "coverage_selector_checkpoint_sha256",
                "coverage_selector_semantic_inspected",
                "coverage_selector_semantic_workspace_tokens",
                "coverage_selector_semantic_elapsed_s",
                "coverage_selector_retained_state_bytes",
                "coverage_selector_status",
                "coverage_selector_bypass_reason",
                "coverage_selector_fallback_reason",
                "coverage_candidate_trace",
            ]
        )
        for question in report.questions:
            writer.writerow(
                [
                    question.question_id,
                    question.category,
                    int(question.in_haystack),
                    int(question.in_context),
                    f"{question.best_f1:.4f}",
                    int(question.in_memory_header),
                    int(question.in_expansions),
                    question.context_tokens,
                    "" if question.evidence_source_recall is None else (
                        f"{question.evidence_source_recall:.4f}"
                    ),
                    "" if question.all_evidence_sources is None else (
                        int(question.all_evidence_sources)
                    ),
                    "|".join(question.retrieved_source_ids),
                    "" if question.raw_evidence_source_recall is None else (
                        f"{question.raw_evidence_source_recall:.4f}"
                    ),
                    "" if question.raw_all_evidence_sources is None else (
                        int(question.raw_all_evidence_sources)
                    ),
                    "|".join(question.raw_retrieved_source_ids),
                    (
                        ""
                        if question.answer_value_components_expected is None
                        else question.answer_value_components_expected
                    ),
                    (
                        ""
                        if question.answer_value_components_found is None
                        else question.answer_value_components_found
                    ),
                    (
                        ""
                        if question.answer_value_component_recall is None
                        else f"{question.answer_value_component_recall:.4f}"
                    ),
                    (
                        ""
                        if question.all_answer_value_components is None
                        else int(question.all_answer_value_components)
                    ),
                    "|".join(
                        "1" if hit else "0"
                        for hit in question.answer_value_component_hit_mask
                    ),
                    question.answer_value_metric_kind,
                    "|".join(question.source_companion_requested),
                    "|".join(question.source_companion_hydrated),
                    "|".join(question.source_companion_orphans),
                    question.source_companion_direct_date_retained,
                    question.source_companion_candidates_before,
                    question.source_companion_candidates_after,
                    "|".join(question.selected_partitions),
                    json.dumps(question.partition_ranking, separators=(",", ":")),
                    question.direct_chunks,
                    question.consolidation_chunks,
                    question.causal_events,
                    question.causal_graph_edges,
                    f"{question.causal_write_s:.4f}",
                    question.qwen_rerank_passes,
                    question.qwen_candidate_inspections,
                    question.qwen_max_workspace_candidates,
                    question.qwen_max_workspace_tokens,
                    question.qwen_candidates_added,
                    question.qwen_feedback_rounds,
                    question.qwen_feedback_seed_sources,
                    question.qwen_feedback_candidates_added,
                    question.qwen_feedback_activation_candidates,
                    question.qwen_feedback_query_tokens,
                    question.coverage_selector_inspected,
                    question.coverage_selector_classified,
                    question.coverage_selector_clusters,
                    question.coverage_selector_null,
                    question.coverage_selector_uncertain,
                    question.coverage_selector_output,
                    question.coverage_selector_representatives,
                    question.coverage_selector_workspace_tokens,
                    f"{question.coverage_selector_elapsed_s:.4f}",
                    question.coverage_selector_operator,
                    (
                        ""
                        if question.coverage_selector_cardinality is None
                        else question.coverage_selector_cardinality
                    ),
                    question.coverage_selector_quantifier,
                    question.coverage_selector_ordering,
                    question.coverage_selector_query_timestamp or "",
                    (
                        ""
                        if question.coverage_selector_temporal_window_days is None
                        else question.coverage_selector_temporal_window_days
                    ),
                    question.coverage_selector_posterior_kind,
                    question.coverage_selector_semantic_score_kind,
                    question.coverage_selector_answerability_score_kind,
                    question.coverage_selector_frontier_candidates,
                    question.coverage_selector_frontier_attempted,
                    question.coverage_selector_frontier_uninspected,
                    int(question.coverage_selector_frontier_exhaustive),
                    question.coverage_selector_frontier_batches,
                    (
                        ""
                        if question.coverage_selector_routed_frontier_exhaustive
                        is None
                        else int(
                            question.coverage_selector_routed_frontier_exhaustive
                        )
                    ),
                    (
                        ""
                        if question.coverage_selector_active_partition_total is None
                        else question.coverage_selector_active_partition_total
                    ),
                    (
                        ""
                        if question.coverage_selector_active_partition_inspected
                        is None
                        else question.coverage_selector_active_partition_inspected
                    ),
                    (
                        ""
                        if question.coverage_selector_active_partition_exhaustive
                        is None
                        else int(
                            question.coverage_selector_active_partition_exhaustive
                        )
                    ),
                    (
                        ""
                        if question.coverage_selector_active_partition_sources_total
                        is None
                        else question.coverage_selector_active_partition_sources_total
                    ),
                    question.coverage_selector_active_partition_structural_rows,
                    question.coverage_selector_active_partition_structural_hypotheses,
                    question.coverage_selector_active_partition_candidates_admitted,
                    (
                        question.coverage_selector_active_partition_candidates_already_present
                    ),
                    question.coverage_selector_active_partition_candidates_replaced,
                    question.coverage_selector_active_partition_candidates_truncated,
                    question.coverage_selector_active_partition_structural_overflow,
                    question.coverage_selector_active_partition_scan_contract,
                    (
                        ""
                        if question.coverage_selector_active_partition_semantically_complete
                        is None
                        else int(
                            question.coverage_selector_active_partition_semantically_complete
                        )
                    ),
                    question.coverage_selector_partition_scope_kind,
                    (
                        ""
                        if question.coverage_selector_partition_inventory_total
                        is None
                        else question.coverage_selector_partition_inventory_total
                    ),
                    (
                        ""
                        if question.coverage_selector_selected_partition_count
                        is None
                        else question.coverage_selector_selected_partition_count
                    ),
                    (
                        ""
                        if question.coverage_selector_partition_scope_exhaustive
                        is None
                        else int(
                            question.coverage_selector_partition_scope_exhaustive
                        )
                    ),
                    (
                        ""
                        if question.coverage_selector_selected_scope_structurally_complete
                        is None
                        else int(
                            question.coverage_selector_selected_scope_structurally_complete
                        )
                    ),
                    (
                        ""
                        if question.coverage_selector_global_semantic_complete
                        is None
                        else int(
                            question.coverage_selector_global_semantic_complete
                        )
                    ),
                    int(
                        question.coverage_selector_allow_selected_scope_fixed_k_closure
                    ),
                    int(question.closure_applied),
                    question.closure_scope,
                    (
                        ""
                        if question.closure_global_recall_guaranteed is None
                        else int(question.closure_global_recall_guaranteed)
                    ),
                    question.coverage_selector_cardinality_deficit,
                    question.coverage_selector_credible_clusters,
                    question.coverage_selector_reserved_representatives,
                    question.coverage_selector_structural_eligible_clusters,
                    (
                        question.coverage_selector_structural_reserved_representatives
                    ),
                    question.coverage_selector_score_provider_fallback,
                    question.coverage_selector_score_provider_model_id,
                    question.coverage_selector_score_provider_model_revision,
                    question.coverage_selector_score_provider_checkpoint_sha256,
                    question.coverage_selector_score_provider_device,
                    question.coverage_selector_score_provider_dtype,
                    question.coverage_selector_score_provider_forward_passes,
                    (
                        question.coverage_selector_score_provider_peak_workspace_tokens
                    ),
                    (
                        question.coverage_selector_score_provider_total_workspace_tokens
                    ),
                    f"{question.coverage_selector_score_provider_elapsed_s:.4f}",
                    (
                        question.coverage_selector_score_provider_retained_state_bytes
                    ),
                    question.coverage_selector_prefix_model_id,
                    question.coverage_selector_prefix_model_revision,
                    question.coverage_selector_prefix_checkpoint_sha256,
                    question.coverage_selector_prefix_device,
                    question.coverage_selector_prefix_dtype,
                    question.coverage_selector_prefix_layers,
                    question.coverage_selector_prefix_attention_layer,
                    question.coverage_selector_model_id,
                    question.coverage_selector_model_revision,
                    question.coverage_selector_checkpoint_sha256,
                    question.coverage_selector_semantic_inspected,
                    question.coverage_selector_semantic_workspace_tokens,
                    f"{question.coverage_selector_semantic_elapsed_s:.4f}",
                    question.coverage_selector_retained_state_bytes,
                    question.coverage_selector_status,
                    question.coverage_selector_bypass_reason,
                    question.coverage_selector_fallback_reason,
                    json.dumps(
                        question.coverage_candidate_trace,
                        separators=(",", ":"),
                    ),
                ]
            )
        Path(args.csv).write_text(output.getvalue(), encoding="utf-8")
        print(f"Per-question CSV written to {args.csv}")


def run_sufficiency_mode(args: argparse.Namespace) -> None:
    """Audit retrieval separately from whether labelled evidence is answerable."""

    from memory_condense.eval.sufficiency import (
        print_sufficiency_report,
        run_sufficiency_audit,
    )

    print(f"Loading benchmark from {args.sufficiency_audit}...")
    samples = load_benchmark(args.sufficiency_audit, args.benchmark_format)
    if not samples:
        print("No samples parsed. Check --benchmark-format.")
        return
    samples = _apply_sample_offset(args, _apply_locked_split(args, samples))
    selected = samples[: args.max_samples] if args.max_samples is not None else samples
    labeled_questions = sum(
        bool(question.evidence_sources)
        for sample in selected
        for question in sample.questions
    )
    if args.provider_retries < 0:
        raise ValueError("--provider-retries must be non-negative")
    planned_calls = (
        2 * labeled_questions * (args.provider_retries + 1)
        if args.use_judge
        else 0
    )
    remote_calls = 0 if args.local_qwen_model_dir else planned_calls
    if remote_calls > args.max_provider_calls:
        raise ValueError(
            f"planned remote provider calls ({remote_calls}) exceed "
            f"--max-provider-calls ({args.max_provider_calls}); explicit "
            "authorization is required"
        )
    if args.qwen_rerank_model_dir and args.local_qwen_model_dir and args.use_judge:
        raise ValueError(
            "the full local judge and prefix reranker cannot share this GPU in "
            "one process; run the deterministic retrieval audit first"
        )
    if (
        (
            args.coverage_selector_local_model_dir
            or args.coverage_selector_qwen_prefix_model_dir
            or args.coverage_selector_cross_encoder_model_dir
        )
        and args.local_qwen_model_dir
        and args.use_judge
    ):
        raise ValueError(
            "the transient coverage selector and local sufficiency judge "
            "cannot share this process"
        )
    _reserve_embedding_device_for_transient_models(args)
    config = config_from_args(args)
    policy_hash = _verified_policy_sha256(
        args.policy_manifest,
        config=config,
        dataset_sha256=file_sha256(args.sufficiency_audit),
        split_manifest=args.benchmark_split_manifest,
        active_split=args.benchmark_split,
    )
    if policy_hash:
        print(f"Verified retrieval policy sha256 {policy_hash[:12]}...")

    local_judge = None
    sufficiency_fn = None
    if args.use_judge and args.local_qwen_model_dir:
        from memory_condense.eval.local_qwen import LocalQwenAnswerer

        local_judge = LocalQwenAnswerer(
            args.local_qwen_model_dir,
            max_new_tokens=args.local_qwen_max_new_tokens,
            gpu_memory=args.local_qwen_gpu_memory,
            cpu_memory=args.local_qwen_cpu_memory,
            dtype=args.local_qwen_dtype,
        )

        from memory_condense.eval.sufficiency import build_sufficiency_prompt

        def local_sufficiency(question, gold, context):
            started = time.perf_counter()
            verdict = local_judge(
                build_sufficiency_prompt(question, gold, context)
            )
            return (
                verdict.upper().startswith("SUFFICIENT"),
                verdict,
                UsageStats(calls=1, elapsed_s=time.perf_counter() - started),
            )

        sufficiency_fn = local_sufficiency
    elif args.use_judge:
        sufficiency_fn = _make_sufficiency_fn(
            args.judge_model,
            retries=args.provider_retries,
        )

    reranker = None
    selector = None
    try:
        reranker = _load_candidate_reranker(args, config)
        selector = _load_coverage_selector(args, config)
        report = run_sufficiency_audit(
            samples,
            config,
            benchmark=Path(args.sufficiency_audit).stem,
            max_samples=args.max_samples,
            ingest_fn=_attach_runtime_controls(
                _benchmark_ingest_fn(args, config),
                reranker=reranker,
                selector=selector,
            ),
            sufficiency_fn=sufficiency_fn,
        )
    finally:
        if reranker is not None:
            reranker.close()
        if selector is not None:
            selector.close()
        if local_judge is not None:
            local_judge.close()

    print_sufficiency_report(report)
    if args.csv:
        rows = [row.model_dump(mode="json") for row in report.questions]
        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                row["expected_source_ids"] = "|".join(row["expected_source_ids"])
                row["retrieved_source_ids"] = "|".join(row["retrieved_source_ids"])
                row["judge_usage"] = json.dumps(row["judge_usage"], sort_keys=True)
                writer.writerow(row)
        Path(args.csv).write_text(output.getvalue(), encoding="utf-8")
        print(f"Per-question sufficiency CSV written to {args.csv}")


def _validate_prepare_cache_args(
    args: argparse.Namespace,
    config: EvalConfig,
) -> None:
    """Reject options that could turn blind cache preparation into an eval."""

    if args.policy_manifest is None:
        raise ValueError("--prepare-cache-only requires --policy-manifest")
    if args.compiled_store_cache is None or args.causal_store_cache is None:
        raise ValueError(
            "--prepare-cache-only requires --compiled-store-cache and "
            "--causal-store-cache"
        )
    if config.retrieval.mode not in {"causal_consolidation", "causal_graph"}:
        raise ValueError(
            "--prepare-cache-only requires --mode causal_consolidation or "
            "causal_graph"
        )

    responder_requested = bool(
        getattr(args, "_responder_model_explicit", False)
        or args.responder_model != DEFAULT_RESPONDER_MODEL
        or args.local_qwen_model_dir
    )
    judge_requested = bool(
        getattr(args, "_judge_model_explicit", False)
        or args.judge_model != DEFAULT_JUDGE_MODEL
        or args.use_judge
    )
    provider_requested = bool(args.max_provider_calls or args.provider_retries)
    if responder_requested or judge_requested or provider_requested:
        raise ValueError(
            "--prepare-cache-only rejects responder, judge, --use-judge, "
            "and remote-provider options"
        )


def _validated_blind_cache_receipts(store) -> dict[str, list[dict[str, object]]]:
    """Copy the two exact, text-free receipts attached by the cache builder."""

    try:
        return validated_cache_receipts(
            getattr(store, "blind_cache_receipts", None)
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _causal_count_timing_metadata(store) -> dict[str, int | float]:
    """Whitelisted scalar causal-build diagnostics, with no event text/IDs."""

    stats = getattr(store, "causal_consolidation_stats", {})
    staging = stats.get("staging", {}) if isinstance(stats, dict) else {}
    learning = stats.get("learning", {}) if isinstance(stats, dict) else {}
    result: dict[str, int | float] = {}
    for source, fields, prefix in (
        (
            staging,
            (
                "source_turns",
                "events",
                "completed_episodes",
                "outcome_chunks_bound",
                "skipped_large_prompt",
                "skipped_insufficient_candidates",
                "elapsed_s",
            ),
            "staging_",
        ),
        (
            learning,
            ("events_offered", "events_applied", "elapsed_s"),
            "learning_",
        ),
    ):
        if not isinstance(source, dict):
            continue
        for field in fields:
            value = source.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[f"{prefix}{field}"] = value
    return result


def run_prepare_cache_only(args: argparse.Namespace) -> dict[str, object]:
    """Build locked compiled+causal caches without observing QA probes.

    Provenance is checked before parsing or ingesting the dataset. The only
    emitted record contains one-way hashes, aggregate counts, and timings; it
    deliberately has no sample IDs, paths, questions, answers, evidence, or
    retrieved content.
    """

    _reserve_embedding_device_for_transient_models(args)
    config = config_from_args(args)
    _validate_prepare_cache_args(args, config)

    dataset_hash = file_sha256(args.benchmark_file)
    implementation_hash = implementation_sha256()
    environment_lock_hash = environment_lock_sha256()
    policy_hash = _verified_policy_sha256(
        args.policy_manifest,
        config=config,
        dataset_sha256=dataset_hash,
        split_manifest=args.benchmark_split_manifest,
        active_split=args.benchmark_split,
        active_implementation_sha256=implementation_hash,
        active_environment_lock_sha256=environment_lock_hash,
        evaluation_identity=_benchmark_evaluation_identity(args, config),
        prepare_only=True,
    )
    split_manifest_hash = (
        file_sha256(args.benchmark_split_manifest)
        if args.benchmark_split_manifest
        else ""
    )

    samples = load_benchmark(args.benchmark_file, args.benchmark_format)
    if not samples:
        raise ValueError("no benchmark samples found")
    samples = _apply_sample_offset(
        args,
        _apply_locked_split(args, samples, verbose=False),
        verbose=False,
    )

    stress_tokens = getattr(args, "stress_context_tokens", None)
    actual_stress_tokens = 0
    if stress_tokens is not None:
        from memory_condense.eval.context_stress import (
            compose_context_stress_sample,
            transcript_tokens,
        )

        samples = [
            compose_context_stress_sample(
                samples,
                target_tokens=stress_tokens,
                max_questions=getattr(args, "stress_questions", 10),
                question_offset=getattr(args, "stress_question_offset", 0),
            )
        ]
        actual_stress_tokens = transcript_tokens(samples[0])

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("cache-preparation shard contains no samples")

    ingest_fn = _benchmark_ingest_fn(args, config, prepare_only=True)
    started = time.perf_counter()
    sample_rows: list[dict[str, object]] = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for index, sample in enumerate(samples):
                sample_started = time.perf_counter()
                store = ingest_fn(
                    sample,
                    config,
                    Path(tmpdir) / f"sample_{index}",
                )
                try:
                    digest = sample_sha256(sample)
                    receipts = _validated_blind_cache_receipts(store)
                    database_path = store.database_path
                    index_path = database_path.with_name("hnsw_index.bin")
                    database_digest = file_sha256(database_path)
                    index_digest = file_sha256(index_path)
                    causal_receipt = receipts["causal"][0]
                    if (
                        causal_receipt["database_sha256"] != database_digest
                        or causal_receipt["index_sha256"] != index_digest
                    ):
                        raise RuntimeError(
                            "active store hashes do not match its causal cache receipt"
                        )
                    row: dict[str, object] = {
                        "sample_sha256": digest,
                        "turn_count": len(sample.turns),
                        "source_count": len(
                            {
                                source_id
                                for source_id in sample.turn_source_ids
                                if source_id is not None
                            }
                        ),
                        "database_sha256": database_digest,
                        "index_sha256": index_digest,
                    }
                    row.update(_causal_count_timing_metadata(store))
                finally:
                    store.close()
                row["elapsed_s"] = time.perf_counter() - sample_started
                row["compiled_cache_entries"] = receipts["compiled"]
                row["causal_cache_entries"] = receipts["causal"]
                sample_rows.append(row)
    finally:
        release_embedder = getattr(ingest_fn, "release_embedder", None)
        if callable(release_embedder):
            release_embedder()

    _assert_implementation_unchanged(implementation_hash)
    report: dict[str, object] = {
        "dataset_sha256": dataset_hash,
        "split_manifest_sha256": split_manifest_hash,
        "policy_manifest_sha256": policy_hash,
        "implementation_sha256": implementation_hash,
        "environment_lock_sha256": environment_lock_hash,
        "sample_count": len(sample_rows),
        "turn_count": sum(int(row["turn_count"]) for row in sample_rows),
        "source_count": sum(int(row["source_count"]) for row in sample_rows),
        "stress_context_tokens": actual_stress_tokens,
        "samples": sample_rows,
        "elapsed_s": time.perf_counter() - started,
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return report


def run_benchmark_mode(args: argparse.Namespace) -> None:
    print(f"Loading benchmark from {args.benchmark_file}...")
    samples = load_benchmark(args.benchmark_file, args.benchmark_format)
    if not samples:
        print("No benchmark samples found.")
        sys.exit(1)

    samples = _apply_sample_offset(args, _apply_locked_split(args, samples))
    stress_tokens = getattr(args, "stress_context_tokens", None)
    if stress_tokens is not None:
        from memory_condense.eval.context_stress import (
            compose_context_stress_sample,
            transcript_tokens,
        )

        samples = [
            compose_context_stress_sample(
                samples,
                target_tokens=stress_tokens,
                max_questions=getattr(args, "stress_questions", 10),
                question_offset=getattr(args, "stress_question_offset", 0),
            )
        ]
        actual_tokens = transcript_tokens(samples[0])
        print(
            f"Context stress memory: {actual_tokens:,} tokens, "
            f"{len(samples[0].turns):,} turns, "
            f"{len(samples[0].questions)} questions"
        )
    questions = sum(len(s.questions) for s in samples)
    print(f"Loaded {len(samples)} samples / {questions} questions")

    planned_provider_calls = _planned_provider_calls(
        samples,
        max_samples=args.max_samples,
        local_answerer=bool(args.local_qwen_model_dir),
        use_judge=args.use_judge,
        provider_retries=args.provider_retries,
    )
    if planned_provider_calls > args.max_provider_calls:
        raise ValueError(
            f"planned remote provider calls ({planned_provider_calls}) exceed "
            f"--max-provider-calls ({args.max_provider_calls}); explicit "
            "authorization is required"
        )

    if args.max_samples is None:
        print(
            "\nWARNING: no --max-samples set. A full run ingests every haystack "
            "through bge-m3 and makes one LLM call per question"
            f"{' (doubled by --use-judge)' if args.use_judge else ''}. "
            "Start with --max-samples 10.\n"
        )

    local_answerer = None
    if (
        args.coverage_selector_local_model_dir
        or args.coverage_selector_qwen_prefix_model_dir
        or args.coverage_selector_cross_encoder_model_dir
    ) and args.local_qwen_model_dir:
        raise ValueError(
            "the transient coverage selector and local responder cannot share "
            "this process"
        )
    _reserve_embedding_device_for_transient_models(args)
    if args.local_qwen_model_dir:
        from memory_condense.eval.local_qwen import LocalQwenAnswerer

        # Keep the 8 GiB GPU available for the offloaded full responder. BGE
        # remains functional on CPU and is unloaded with the benchmark run.
        if args.embedding_device is None:
            args.embedding_device = "cpu"
        print(f"Loading local responder from {args.local_qwen_model_dir}...")
        local_answerer = LocalQwenAnswerer(
            args.local_qwen_model_dir,
            max_new_tokens=args.local_qwen_max_new_tokens,
            gpu_memory=args.local_qwen_gpu_memory,
            cpu_memory=args.local_qwen_cpu_memory,
            dtype=args.local_qwen_dtype,
        )
        args.responder_model = (
            f"local/{args.local_qwen_model_dir.name}:{local_answerer.dtype_name}"
        )

    config = config_from_args(args)
    dataset_hash = file_sha256(args.benchmark_file)
    split_manifest_hash = (
        file_sha256(args.benchmark_split_manifest)
        if args.benchmark_split_manifest
        else ""
    )
    implementation_hash = implementation_sha256()
    environment_lock_hash = environment_lock_sha256()
    evaluation_identity = _benchmark_evaluation_identity(args, config)
    policy_hash = _verified_policy_sha256(
        args.policy_manifest,
        config=config,
        dataset_sha256=dataset_hash,
        split_manifest=args.benchmark_split_manifest,
        active_split=args.benchmark_split,
        active_implementation_sha256=implementation_hash,
        active_environment_lock_sha256=environment_lock_hash,
        evaluation_identity=evaluation_identity,
    )
    reranker = None
    selector = None
    try:
        reranker = _load_candidate_reranker(args, config)
        selector = _load_coverage_selector(args, config)
        result = run_benchmark(
            samples,
            config,
            answer_fn=(
                local_answerer
                or _make_answer_fn(
                    args.responder_model,
                    retries=args.provider_retries,
                )
            ),
            judge_fn=(
                _make_judge_fn(
                    args.judge_model,
                    retries=args.provider_retries,
                )
                if args.use_judge
                else None
            ),
            max_samples=args.max_samples,
            # Label the run with the dataset, not the --benchmark-format flag, which
            # defaults to "auto" and would name every report benchmark_auto_*.json.
            benchmark=Path(args.benchmark_file).stem,
            ingest_fn=_attach_runtime_controls(
                _benchmark_ingest_fn(args, config),
                reranker=reranker,
                selector=selector,
            ),
            verbose=True,
            dataset_sha256=dataset_hash,
            split_manifest_sha256=split_manifest_hash,
            benchmark_split=args.benchmark_split or "",
            implementation_sha256=implementation_hash,
            environment_lock_sha256=environment_lock_hash,
            policy_manifest_sha256=policy_hash,
            evaluation_protocol=evaluation_identity,
        )
    finally:
        if reranker is not None:
            reranker.close()
        if selector is not None:
            selector.close()
        if local_answerer is not None:
            print(
                f"Local responder: {local_answerer.calls} calls in "
                f"{local_answerer.elapsed_s:.1f}s"
            )
            local_answerer.close()
    _assert_implementation_unchanged(implementation_hash)
    print_benchmark_summary(result)
    path = save_benchmark_report(result, args.results_dir)
    print(f"\nBenchmark report saved to {path}")


def run_replay_mode(args: argparse.Namespace) -> None:
    print(f"Loading conversations from {args.conversation_dir}...")
    conversations = load_directory(args.conversation_dir)
    if not conversations:
        print("No conversations found.")
        sys.exit(1)
    print(f"Found {len(conversations)} conversations")

    config = config_from_args(args)
    if config.retrieval.coverage_selection:
        raise ValueError(
            "coverage selection is currently measured through benchmark/recall "
            "packing, not self-replay"
        )

    if args.sweep:
        report = run_sweep(config, conversations)
        print_sweep_table(report)
        path = save_sweep_report(report, args.results_dir)
        print(f"\nSweep report saved to {path}")
        return

    print("\nRunning single eval...")
    result = run_eval(config, conversations)
    print_run_summary(result)
    path = save_run_result(result, args.results_dir)
    print(f"\nResult saved to {path}")

    if args.csv:
        Path(args.csv).write_text(to_csv(result), encoding="utf-8")
        print(f"Per-turn CSV written to {args.csv}")


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
