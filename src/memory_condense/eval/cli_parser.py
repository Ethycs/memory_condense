"""CLI argument contracts for every evaluation mode."""

from __future__ import annotations

import argparse
from pathlib import Path

from memory_condense.eval.schemas import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_RESPONDER_MODEL,
)

CLI_EPILOG = """Four modes, selected by flags:

    # 1. Self-replay on your own exported conversations (the default)
    pixi run python -m memory_condense.eval --conversation-dir <path>

    # 2. Parameter sweep over chunker/retrieval settings
    pixi run python -m memory_condense.eval --conversation-dir <path> --sweep

    # 3. Public benchmark (LongMemEval / LoCoMo) QA probes
    pixi run python -m memory_condense.eval --benchmark-file longmemeval_oracle.json

    # 4. Offline analysis of two saved runs (no API calls, no cost)
    pixi run python -m memory_condense.eval --compare baseline.json treatment.json
"""

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
        epilog=CLI_EPILOG,
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
