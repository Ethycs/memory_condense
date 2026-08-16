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
            "or bounded source-local expansion around hybrid anchors"
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

    return parser


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    # --hybrid predates --mode and is kept so the commands in
    # `docs/02 - Implementation/01` keep working.
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
            neighbor_radius=args.neighbor_radius,
            neighbor_slots=args.neighbor_slots,
            neighbor_replacement_slots=args.neighbor_replacement_slots,
            neighbor_direction=args.neighbor_direction,
        ),
        judge_model=args.judge_model,
        responder_model=args.responder_model,
        embedding_device=args.embedding_device,
        conversation_dir=args.conversation_dir or args.benchmark_file or "",
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

    def answer_fn(
        messages: list[dict[str, str]],
    ) -> tuple[str, UsageStats]:
        started = time.perf_counter()
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
            num_retries=retries,
        )
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


def _apply_locked_split(args: argparse.Namespace, samples):
    manifest_path = args.benchmark_split_manifest
    split = args.benchmark_split
    if bool(manifest_path) != bool(split):
        raise ValueError(
            "--benchmark-split-manifest and --benchmark-split must be used together"
        )
    if not manifest_path:
        return samples
    dataset_path = args.answer_recall or args.benchmark_file
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
    if config.retrieval.mode in {"hybrid_source", "hybrid_graph"}:
        expected.update(
            {
                "source_slots": config.retrieval.source_slots,
                "source_activation_k": (
                    config.retrieval.source_activation_k or config.retrieval.k
                ),
                "source_candidate_pool": config.retrieval.source_candidate_pool,
            }
        )
    if config.retrieval.mode == "hybrid_graph":
        expected["neighbor_direction"] = config.retrieval.neighbor_direction
    if retrieval != expected:
        raise ValueError(
            f"policy manifest retrieval config mismatch: expected {retrieval}, "
            f"got {expected}"
        )
    return file_sha256(path)


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

    samples = _apply_locked_split(args, samples)
    config = config_from_args(args)
    print(
        f"{len(samples)} sample(s); measuring "
        f"{args.max_samples or len(samples)} in {config.retrieval.mode} mode. "
        "No API calls will be made."
    )
    report = run_recall(
        samples,
        config,
        benchmark=Path(args.answer_recall).stem,
        max_samples=args.max_samples,
        ingest_fn=(
            compiled_store_ingest_fn(
                args.compiled_store_cache,
                device=config.embedding_device,
            )
            if args.compiled_store_cache
            else ingest_sample
        ),
    )
    print_recall_report(report)

    if args.csv:
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            [
                "question_id",
                "category",
                "in_context",
                "best_f1",
                "in_header",
                "in_expansions",
                "context_tokens",
                "evidence_source_recall",
                "all_evidence_sources",
                "retrieved_source_ids",
            ]
        )
        for question in report.questions:
            writer.writerow(
                [
                    question.question_id,
                    question.category,
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
                ]
            )
        Path(args.csv).write_text(output.getvalue(), encoding="utf-8")
        print(f"Per-question CSV written to {args.csv}")


def run_benchmark_mode(args: argparse.Namespace) -> None:
    print(f"Loading benchmark from {args.benchmark_file}...")
    samples = load_benchmark(args.benchmark_file, args.benchmark_format)
    if not samples:
        print("No benchmark samples found.")
        sys.exit(1)

    samples = _apply_locked_split(args, samples)
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
    try:
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
            ingest_fn=(
                compiled_store_ingest_fn(
                    args.compiled_store_cache,
                    device=config.embedding_device,
                )
                if args.compiled_store_cache
                else ingest_sample
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
        bool(args.benchmark_file),
        bool(args.conversation_dir),
    ]
    if sum(modes) > 1:
        parser.error(
            "--compare, --answer-recall, --benchmark-file, and --conversation-dir "
            "are mutually exclusive"
        )

    if args.compare:
        run_compare(args)
    elif args.answer_recall:
        run_answer_recall(args)
    elif args.benchmark_file:
        run_benchmark_mode(args)
    elif args.conversation_dir:
        run_replay_mode(args)
    else:
        parser.error(
            "one of --conversation-dir, --benchmark-file, --compare, or "
            "--answer-recall is required"
        )


if __name__ == "__main__":
    main()
