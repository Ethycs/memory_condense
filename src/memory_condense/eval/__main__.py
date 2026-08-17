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
import io
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from memory_condense.eval.analysis import (
    compare_runs,
    load_run,
    print_comparison,
    to_csv,
)
from memory_condense.eval.benchmark import (
    build_judge_prompt,
    ingest_sample,
    print_benchmark_summary,
    run_benchmark,
    save_benchmark_report,
)
from memory_condense.eval.compiled_cache import compiled_store_ingest_fn
from memory_condense.eval.judge import JUDGE_MAX_TOKENS
from memory_condense.eval.locked_split import load_split_manifest, select_locked_split
from memory_condense.eval.recall import print_recall_report, run_recall
from memory_condense.eval.reproducibility import (
    environment_lock_sha256,
    file_sha256,
    implementation_sha256,
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
from memory_condense.loader import load_benchmark, load_directory


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
        "--judge-model", default=DEFAULT_JUDGE_MODEL, help="LLM model for judging"
    )
    parser.add_argument(
        "--responder-model",
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
        help="Hard responder prompt-content token cap per question (default 8000)",
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

    return parser


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    # --hybrid predates --mode and is kept so the commands in
    # `docs/02 - Implementation/01` keep working.
    if args.qwen_feedback and not args.qwen_rerank_model_dir:
        raise ValueError("--qwen-feedback requires --qwen-rerank-model-dir")
    mode = "hybrid" if args.hybrid and args.mode == "dense" else args.mode
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


def _make_answer_fn(model: str, *, retries: int = 0):
    """Answer a benchmark question. Short, deterministic answers — F1/EM depend on it."""
    import litellm

    central_dev_client = None
    api_base = os.environ.get("OPENAI_API_BASE", "")
    if "codex_sdk/" in model and "central-dev.zt" in api_base:
        # central-dev terminates TLS with the trusted internal Caddy CA.
        # Give OpenAI/LiteLLM an explicit transport backed by the Windows trust
        # store; their default certifi transport cannot see that installed CA.
        import ssl

        import httpx
        import truststore
        from openai import OpenAI

        ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        central_dev_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=api_base,
            http_client=httpx.Client(verify=ssl_context),
        )

    def answer_fn(
        messages: list[dict[str, str]],
    ) -> tuple[str, UsageStats]:
        started = time.perf_counter()
        request = {
            "model": model,
            "messages": messages,
            "max_tokens": 256,
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
        return _content(response), UsageStats.from_litellm(
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

    def judge_fn(
        question: str,
        gold: str,
        prediction: str,
    ) -> tuple[bool, str, UsageStats]:
        started = time.perf_counter()
        response = litellm.completion(
            model=model,
            messages=build_judge_prompt(question, gold, prediction),
            max_tokens=JUDGE_MAX_TOKENS,
            num_retries=retries,
        )
        text = _content(response)
        return (
            text.upper().startswith("CORRECT"),
            text,
            UsageStats.from_litellm(response, time.perf_counter() - started),
        )

    return judge_fn


def _make_sufficiency_fn(model: str, *, retries: int = 0):
    """Judge whether excerpts can derive the gold answer, not an answer string."""

    import litellm

    from memory_condense.eval.sufficiency import build_sufficiency_prompt

    def sufficiency_fn(
        question: str,
        gold: str,
        context: list[str],
    ) -> tuple[bool, str, UsageStats]:
        started = time.perf_counter()
        response = litellm.completion(
            model=model,
            messages=build_sufficiency_prompt(question, gold, context),
            max_tokens=JUDGE_MAX_TOKENS,
            num_retries=retries,
        )
        verdict = _content(response)
        return (
            verdict.upper().startswith("SUFFICIENT"),
            verdict,
            UsageStats.from_litellm(response, time.perf_counter() - started),
        )

    return sufficiency_fn


def _apply_locked_split(args: argparse.Namespace, samples):
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
    print(
        f"Locked split {split!r}: {len(selected)} / {len(samples)} samples "
        f"(dataset sha256 {manifest.dataset_sha256[:12]}...)"
    )
    return selected


def _apply_sample_offset(args: argparse.Namespace, samples):
    offset = int(args.sample_offset)
    if offset < 0:
        raise ValueError("--sample-offset must be non-negative")
    if offset >= len(samples) and offset:
        raise ValueError(
            f"--sample-offset {offset} is outside the {len(samples)} samples"
        )
    if offset:
        print(f"Sample shard starts at locked-split offset {offset}")
        return samples[offset:]
    return samples


def _planned_provider_calls(
    samples,
    *,
    max_samples: int | None,
    local_answerer: bool,
    use_judge: bool,
) -> int:
    selected = samples[:max_samples] if max_samples is not None else samples
    questions = sum(len(sample.questions) for sample in selected)
    return (0 if local_answerer else questions) + (
        questions if use_judge else 0
    )


def _verified_policy_sha256(
    path: Path | None,
    *,
    config: EvalConfig,
    dataset_sha256: str,
    split_manifest: str | None,
) -> str:
    if path is None:
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = str(payload.get("status", ""))
    if not status or status.startswith("superseded"):
        raise ValueError(f"policy manifest is not active: {path}")
    if payload.get("dataset_sha256") != dataset_sha256:
        raise ValueError("policy manifest dataset SHA-256 mismatch")
    if split_manifest is None or payload.get("split_manifest") != Path(
        split_manifest
    ).name:
        raise ValueError("policy manifest locked-split identity mismatch")
    retrieval = payload.get("retrieval", {})
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
    if retrieval != expected:
        raise ValueError(
            f"policy manifest retrieval config mismatch: expected {retrieval}, "
            f"got {expected}"
        )
    return file_sha256(path)


def _benchmark_ingest_fn(args: argparse.Namespace, config: EvalConfig):
    """Select an isolated benchmark writer without mutating compiled stores."""

    if config.retrieval.mode in {"causal_consolidation", "causal_graph"}:
        from memory_condense.eval.causal_benchmark import (
            causal_consolidation_ingest_fn,
        )

        return causal_consolidation_ingest_fn(
            args.compiled_store_cache,
            causal_cache_root=args.causal_store_cache,
            device=config.embedding_device,
        )
    if args.compiled_store_cache:
        return compiled_store_ingest_fn(
            args.compiled_store_cache,
            device=config.embedding_device,
        )
    return ingest_sample


def _load_candidate_reranker(args: argparse.Namespace, config: EvalConfig):
    """Load one shared, bounded Qwen control plane for a benchmark run."""

    if not (config.retrieval.qwen_rerank or config.retrieval.qwen_feedback):
        return None
    if args.qwen_rerank_model_dir is None:
        raise ValueError("Qwen attention requires --qwen-rerank-model-dir")
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
    )


def _attach_candidate_reranker(ingest_fn, reranker):
    if reranker is None:
        return ingest_fn

    def attached(sample, config, data_dir):
        condenser = ingest_fn(sample, config, data_dir)
        condenser.set_source_candidate_reranker(reranker)
        return condenser

    return attached


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
            )
        ]
        actual_tokens = transcript_tokens(samples[0])
        print(
            f"Context stress memory: {actual_tokens:,} tokens, "
            f"{len(samples[0].turns):,} turns, "
            f"{len(samples[0].questions)} questions"
        )
    if args.qwen_rerank_model_dir and args.embedding_device is None:
        # Keep the GPU for the attention slice; BGE remains functional on CPU.
        args.embedding_device = "cpu"
    config = config_from_args(args)
    print(
        f"{len(samples)} sample(s); measuring "
        f"{args.max_samples or len(samples)} in {config.retrieval.mode} mode. "
        "No API calls will be made."
    )
    reranker = _load_candidate_reranker(args, config)
    try:
        report = run_recall(
            samples,
            config,
            benchmark=Path(args.answer_recall).stem,
            max_samples=1 if stress_tokens is not None else args.max_samples,
            ingest_fn=_attach_candidate_reranker(
                _benchmark_ingest_fn(args, config),
                reranker,
            ),
        )
    finally:
        if reranker is not None:
            reranker.close()
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
    planned_calls = 2 * labeled_questions if args.use_judge else 0
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
    if args.qwen_rerank_model_dir and args.embedding_device is None:
        args.embedding_device = "cpu"
    config = config_from_args(args)
    policy_hash = _verified_policy_sha256(
        args.policy_manifest,
        config=config,
        dataset_sha256=file_sha256(args.sufficiency_audit),
        split_manifest=args.benchmark_split_manifest,
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
    try:
        reranker = _load_candidate_reranker(args, config)
        report = run_sufficiency_audit(
            samples,
            config,
            benchmark=Path(args.sufficiency_audit).stem,
            max_samples=args.max_samples,
            ingest_fn=_attach_candidate_reranker(
                _benchmark_ingest_fn(args, config),
                reranker,
            ),
            sufficiency_fn=sufficiency_fn,
        )
    finally:
        if reranker is not None:
            reranker.close()
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
    if args.qwen_rerank_model_dir and args.embedding_device is None:
        args.embedding_device = "cpu"
    if args.local_qwen_model_dir:
        from memory_condense.eval.local_qwen import LocalQwenAnswerer

        # Keep the 8 GiB GPU available for the offloaded full responder. BGE
        # remains functional on CPU and is unloaded with the benchmark run.
        if args.embedding_device is None:
            args.embedding_device = "cpu"
        args.responder_model = f"local/{args.local_qwen_model_dir.name}:bf16"
        print(f"Loading local responder from {args.local_qwen_model_dir}...")
        local_answerer = LocalQwenAnswerer(
            args.local_qwen_model_dir,
            max_new_tokens=args.local_qwen_max_new_tokens,
            gpu_memory=args.local_qwen_gpu_memory,
            cpu_memory=args.local_qwen_cpu_memory,
        )

    config = config_from_args(args)
    dataset_hash = file_sha256(args.benchmark_file)
    split_manifest_hash = (
        file_sha256(args.benchmark_split_manifest)
        if args.benchmark_split_manifest
        else ""
    )
    policy_hash = _verified_policy_sha256(
        args.policy_manifest,
        config=config,
        dataset_sha256=dataset_hash,
        split_manifest=args.benchmark_split_manifest,
    )
    reranker = None
    try:
        reranker = _load_candidate_reranker(args, config)
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
            ingest_fn=_attach_candidate_reranker(
                _benchmark_ingest_fn(args, config),
                reranker,
            ),
            verbose=True,
            dataset_sha256=dataset_hash,
            split_manifest_sha256=split_manifest_hash,
            benchmark_split=args.benchmark_split or "",
            implementation_sha256=implementation_sha256(),
            environment_lock_sha256=environment_lock_sha256(),
            policy_manifest_sha256=policy_hash,
        )
    finally:
        if reranker is not None:
            reranker.close()
        if local_answerer is not None:
            print(
                f"Local responder: {local_answerer.calls} calls in "
                f"{local_answerer.elapsed_s:.1f}s"
            )
            local_answerer.close()
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

    if args.compare:
        run_compare(args)
    elif args.answer_recall:
        run_answer_recall(args)
    elif args.sufficiency_audit:
        run_sufficiency_mode(args)
    elif args.benchmark_file:
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
